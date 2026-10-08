"""Medical intensity preparation and RTV priors, before masking.

Raw images are supplied through JSON manifest records. Raster PNG/JPEG inputs
are already prepared data: lost HU and volume statistics cannot be recovered
from them. Network tensors retain the repository's [-1, 1] convention.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy import sparse
from scipy.sparse.linalg import spsolve


def ct_window(hu, width=400.0, level=40.0):
    hu = np.asarray(hu, dtype=np.float32)
    if width <= 0 or not np.isfinite(hu).all():
        raise ValueError('CT input must contain finite HU values and positive width')
    return np.clip((hu - (level - width / 2.0)) / width, 0, 1)


def mri_volume_normalize(volume):
    """Foreground z-score and min-max scaling, jointly over one volume."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('MRI raw input must be a finite three-dimensional volume')
    foreground = volume != 0
    out = np.zeros_like(volume)
    if not foreground.any():
        return out
    values = volume[foreground]
    std = float(values.std())
    if std == 0:
        return out
    z = (values - float(values.mean())) / std
    span = float(z.max() - z.min())
    if span > 0:
        out[foreground] = (z - z.min()) / span
    return out


@lru_cache(maxsize=2)
def _prepared_volume(path, modality, width, level):
    if str(path).lower().endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        arr = nib.load(path).get_fdata(dtype=np.float32)
    else:
        arr = np.load(path, allow_pickle=False)
    return mri_volume_normalize(arr) if modality == 'MRI' else ct_window(arr, width, level)


def load_medical_image(entry, config):
    """Return a prepared RGB PIL image from a raster path or raw manifest entry.

    Manifest record: {path, modality: CT|MRI, slice_index, slice_axis}.
    CT NPY/NIfTI arrays must already be HU; DICOM applies slope/intercept first.
    MRI DICOM volumes use {path: directory, modality: MRI, ...}; the directory
    must contain exactly one series. Volume normalization never uses other
    subjects, partitions or volumes.
    """
    if not isinstance(entry, dict):
        return Image.open(entry).convert('RGB')
    path = str(entry['path'])
    modality = entry['modality'].upper().replace('MR', 'MRI') if entry['modality'].upper() == 'MR' else entry['modality'].upper()
    if modality not in ('CT', 'MRI'):
        raise ValueError('Raw image modality must be CT or MRI')
    width = float(config.CT_WINDOW_WIDTH or 400)
    level = float(config.CT_WINDOW_LEVEL if config.CT_WINDOW_LEVEL is not None else 40)
    if Path(path).is_dir() or path.lower().endswith('.dcm'):
        import pydicom
        from pydicom.pixels import apply_modality_lut
        if Path(path).is_dir():
            records = [pydicom.dcmread(str(p)) for p in Path(path).glob('*.dcm')]
            if not records or len({str(r.SeriesInstanceUID) for r in records}) != 1:
                raise ValueError('Provide one nonempty DICOM volume/series per manifest entry')
            records.sort(key=lambda r: float(r.ImagePositionPatient[2]) if hasattr(r, 'ImagePositionPatient') else int(r.InstanceNumber))
            arr = np.stack([apply_modality_lut(r.pixel_array, r) for r in records])
            prepared = mri_volume_normalize(arr) if modality == 'MRI' else ct_window(arr, width, level)
            image = np.take(prepared, int(entry['slice_index']), axis=int(entry.get('slice_axis', 0)))
        else:
            record = pydicom.dcmread(path)
            arr = apply_modality_lut(record.pixel_array, record)
            if modality == 'MRI':
                if arr.ndim != 3:
                    raise ValueError('MRI must supply a complete volume, not an independently normalized slice')
                prepared = mri_volume_normalize(arr)
                image = np.take(prepared, int(entry['slice_index']), axis=int(entry.get('slice_axis', 0)))
            else:
                prepared = ct_window(arr, width, level)
                image = np.take(prepared, int(entry['slice_index']), axis=int(entry.get('slice_axis', 0))) if prepared.ndim == 3 else prepared
    else:
        arr = _prepared_volume(path, modality, width, level)
        image = np.take(arr, int(entry['slice_index']), axis=int(entry.get('slice_axis', 0))) if arr.ndim == 3 else arr
    if image.ndim != 2:
        raise ValueError('Manifest must select a two-dimensional slice')
    return Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).convert('RGB')


def read_manifest(path):
    manifest = Path(path)
    records = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('Medical manifest must contain a JSON list of image records')
    for record in records:
        source = Path(record['path'])
        record['path'] = str(source if source.is_absolute() else manifest.parent / source)
    return records


def rtv_structure(image, lam=0.015, sigma=3.0, iterations=30, sharpness=0.001):
    """RTV reweighted smoothing, mirroring the supplied MATLAB tsmooth solver.

    Receives the unmasked image. There is no missing-region mask in this step.
    The MATLAB implementation remains the offline reference implementation.
    """
    original = np.asarray(image, dtype=np.float64) / 255.0
    if original.ndim == 2:
        original = original[..., None]
    height, width, channels = original.shape
    current = original.copy()
    indices = np.arange(height * width).reshape(height, width)
    for _ in range(int(iterations)):
        dx = np.diff(current, axis=1, append=current[:, -1:, :])
        dy = np.diff(current, axis=0, append=current[-1:, :, :])
        total = np.maximum(np.sqrt(dx * dx + dy * dy).mean(axis=2), sharpness) ** -1
        length = int(np.floor(5 * sigma + 0.5)) | 1
        grid = np.arange(length) - length // 2
        kernel = np.exp(-grid ** 2 / (2 * sigma ** 2)); kernel /= kernel.sum()
        smooth = convolve1d(convolve1d(current, kernel, axis=1, mode='constant'), kernel, axis=0, mode='constant')
        sx = np.diff(smooth, axis=1, append=smooth[:, -1:, :])
        sy = np.diff(smooth, axis=0, append=smooth[-1:, :, :])
        wx = total / np.maximum(np.abs(sx).mean(axis=2), 0.001)
        wy = total / np.maximum(np.abs(sy).mean(axis=2), 0.001)
        first = np.concatenate([indices[:, :-1].ravel(), indices[:-1, :].ravel()])
        second = np.concatenate([indices[:, 1:].ravel(), indices[1:, :].ravel()])
        weights = float(lam) / 2 * np.concatenate([wx[:, :-1].ravel(), wy[:-1, :].ravel()])
        diagonal = np.ones(height * width)
        np.add.at(diagonal, first, weights); np.add.at(diagonal, second, weights)
        matrix = sparse.coo_matrix((np.concatenate([-weights, -weights, diagonal]),
                                  (np.concatenate([first, second, indices.ravel()]),
                                   np.concatenate([second, first, indices.ravel()]))),
                                  shape=(height * width, height * width)).tocsc()
        current = spsolve(matrix, original.reshape(-1, channels)).reshape(original.shape)
        sigma = max(sigma / 2, 0.5)
    result = np.round(np.clip(current, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result[..., 0] if channels == 1 else result).convert('RGB')
