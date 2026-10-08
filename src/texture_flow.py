from torch.autograd import Variable, grad
import numpy as np
import csv
import json
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import glob
import torchvision.utils as vutils
import math
import shutil
import tensorboardX
from itertools import islice
from torch.utils.data import DataLoader
from .data import Dataset
from .utils import Progbar, write_2images, write_2tensorboard, create_dir, imsave
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from .models_last import TextureFlowModel


class TextureFlow():
    def __init__(self, config):
        self.config = config
        if config.MODE == 'train' and config.RESUME_ALL:
            raise ValueError('Formal training starts from initialization; checkpoint resumption is disabled.')

        if self.config.MODEL == 1:
            self.stage_name = 'structure'
        elif self.config.MODEL == 2:
            self.stage_name = 'texture'
        elif self.config.MODEL == 3:
            self.stage_name = 'fusion'

        self.debug = False
        self.inpaint_model = TextureFlowModel(config).to(config.DEVICE)
        self.samples_path = os.path.join(config.PATH, config.NAME, 'images')
        self.checkpoints_path = os.path.join(config.PATH, config.NAME, 'checkpoints')
        self.test_image_path = os.path.join(config.PATH, config.NAME, 'test_result')

        if self.config.MODE == 'train' and not self.config.RESUME_ALL:
            if config.MODEL == 3:
                for name, path in (('s_gen', config.SR_CHECKPOINT), ('t_gen', config.TR_CHECKPOINT)):
                    if not path or not os.path.isfile(path):
                        raise ValueError('Formal fusion training requires SR_CHECKPOINT and TR_CHECKPOINT.')
                    state = torch.load(path, map_location=config.DEVICE, weights_only=True)
                    getattr(self.inpaint_model, name).load_state_dict(state, strict=True)
        else:
            self.inpaint_model.load(self.config.WHICH_ITER)
        if config.MODEL == 3:
            self.inpaint_model.s_gen.requires_grad_(False).eval()
            self.inpaint_model.t_gen.requires_grad_(False).eval()

    def train(self):
        train_writer = self.obtain_log(self.config)
        # 从这里进入纹理提取过程（LBP）
        train_dataset = Dataset(
            self.config.DATA_TRAIN_GT,
            self.config.DATA_TRAIN_STRUCTURE,
            self.config,
            self.config.DATA_MASK_FILE
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        val_dataset = Dataset(self.config.DATA_VAL_GT, self.config.DATA_VAL_STRUCTURE,
                              self.config, self.config.DATA_VAL_MASK, evaluation=True)
        sample_iterator = val_dataset.create_iterator(self.config.SAMPLE_SIZE)

        iterations = self.inpaint_model.iterations
        total = len(train_dataset)
        epoch = math.floor(iterations * self.config.TRAIN_BATCH_SIZE / total)
        keep_training = True
        model = self.config.MODEL
        max_iterations = int(float(self.config.MAX_ITERS))
        if len(train_loader)==0: raise ValueError('Training set must contain at least one full batch.')
        if iterations>=max_iterations: raise ValueError('Checkpoint already reached the configured budget.')

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                # input_image, structure_image, texture_image, gt_image, inpaint_map
                inputs, smooths, lbps, gts, masks = self.cuda(*items)

                # texture model
                if model == 1:
                    logs = self.inpaint_model.update_structure(inputs, smooths, masks)
                    iterations = self.inpaint_model.iterations
                # flow modelW
                elif model == 2:
                    logs = self.inpaint_model.update_texture(inputs, lbps, masks)
                    iterations = self.inpaint_model.iterations
                # flow with structure model
                elif model == 3:
                    with torch.no_grad():
                        smooth_stage_1, texture_stage_2 = self.inpaint_model.reconstruct_priors(inputs, smooths, lbps, masks)
                    logs = self.inpaint_model.update_inpaint(inputs, smooth_stage_1.detach(), texture_stage_2.detach(),
                                                             gts, masks, self.inpaint_model.use_correction_loss,
                                                             self.inpaint_model.use_vgg_loss)
                    iterations = self.inpaint_model.iterations


                # print(logs)
                logs = [
                           ("epoch", epoch),
                           ("iter", iterations),
                       ] + logs

                progbar.add(len(inputs),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model
                if self.config.LOG_INTERVAL and iterations % self.config.LOG_INTERVAL == 0:
                    self.write_loss(logs, train_writer)
                # sample model
                if self.config.SAMPLE_INTERVAL and iterations % self.config.SAMPLE_INTERVAL == 0:
                    items = next(sample_iterator)
                    inputs, smooths, lbps, gts, masks = self.cuda(*items)
                    result = self.inpaint_model.sample(inputs, smooths, lbps, gts, masks)
                    self.write_image(result, train_writer, iterations, 'image')
                # evaluate model
                if self.config.EVAL_INTERVAL and iterations % self.config.EVAL_INTERVAL == 0:
                    self.inpaint_model.eval()
                    print('\nstart eval...\n')
                    self.eval(writer=train_writer)
                    self.inpaint_model.train()

                # save the latest model
                if self.config.SAVE_LATEST and iterations % self.config.SAVE_LATEST == 0:
                    print('\nsaving the latest model (total_steps %d)\n' % (iterations))
                    self.inpaint_model.save('latest')

                # save the model
                if self.config.SAVE_INTERVAL and iterations % self.config.SAVE_INTERVAL == 0:
                    print('\nsaving the model of iterations %d\n' % iterations)
                    self.inpaint_model.save(iterations)
                if iterations >= max_iterations:
                    self.inpaint_model.save(iterations)
                    self.inpaint_model.save('latest')
                    keep_training = False
                    break
        train_writer.close()
        print('\nEnd training....')

    def eval(self, writer=None):
        val_dataset = Dataset(self.config.DATA_VAL_GT,
                              self.config.DATA_VAL_STRUCTURE,
                              self.config,
                              self.config.DATA_VAL_MASK, evaluation=True)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=False
        )
        model = self.config.MODEL
        total = len(val_dataset)
        iterations = self.inpaint_model.iterations

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        psnr_list = []

        # TODO: add fid score to evaluate
        with torch.no_grad():
            # for items in val_loader:
            for j, items in enumerate(val_loader):

                logs = []
                iteration += 1
                inputs, smooths, lbps, gts, masks = self.cuda(*items)
                if model == 1:
                    outputs_structure = self.inpaint_model.structure_forward(inputs, smooths, masks)
                    psnr, ssim, l1 = self.metrics(outputs_structure, smooths)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.extend([psnr.item()] * len(inputs))

                # inpaint model
                elif model == 2:
                    outputs_texture = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    psnr, ssim, l1 = self.metrics(outputs_texture, lbps)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.extend([psnr.item()] * len(inputs))


                # inpaint with structure model
                elif model == 3:
                    smooth_stage_1, texture_stage_2 = self.inpaint_model.reconstruct_priors(inputs, smooths, lbps, masks)
                    outputs, lbp = self.inpaint_model.inpaint_forward(inputs, smooth_stage_1.detach(),
                                                                      texture_stage_2.detach(), masks)
                    psnr, ssim, l1 = self.metrics(outputs, gts)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.extend([psnr.item()] * len(inputs))

                logs = [("it", iteration), ] + logs
                progbar.add(len(inputs), values=logs)

        avg_psnr = np.average(psnr_list)

        if writer is not None:
            writer.add_scalar('eval_psnr', avg_psnr, iterations)

        print('model eval at iterations:%d' % iterations)
        print('average psnr:%f' % avg_psnr)

    def test(self):
        self.inpaint_model.eval()

        model = self.config.MODEL
        print(self.config.DATA_TEST_RESULTS)
        create_dir(self.config.DATA_TEST_RESULTS)
        test_dataset = Dataset(self.config.DATA_TEST_GT, self.config.DATA_TEST_STRUCTURE,
                               self.config, self.config.DATA_TEST_MASK, evaluation=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.TEST_BATCH_SIZE or 16,
        )

        index = 0
        metric_rows = []
        raw_dir = self.config.DATA_TEST_RESULTS
        completed_dir = raw_dir + '_completed_display'
        tensor_dir = raw_dir + '_raw_tensors'
        create_dir(completed_dir)
        create_dir(tensor_dir)
        with torch.no_grad():
            for items in test_loader:
                # input_image, structure_image, texture_image, gt_image, inpaint_map
                inputs, smooths, lbps, gts, masks = self.cuda(*items)

                # structure model
                if model == 1:
                    outputs = self.inpaint_model.structure_forward(inputs, smooths, masks)
                    outputs_merged = (outputs * masks) + (smooths * (1 - masks))

                # texture model
                elif model == 2:
                    outputs = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    outputs_merged = (outputs * masks) + (lbps * (1 - masks))


                # inpaint with structure model / joint model
                else:
                    smooth_stage_1, texture_stage_2 = self.inpaint_model.reconstruct_priors(inputs, smooths, lbps, masks)
                    outputs, lbp = self.inpaint_model.inpaint_forward(inputs, smooth_stage_1.detach(),
                                                                   texture_stage_2.detach(), masks)
                    outputs_merged = (outputs * masks) + (gts * (1 - masks))

                # Preserve continuous raw predictions for pixel metrics before any
                # display clipping, quantization or known-region compositing.
                raw_arrays = outputs.detach().cpu().numpy()
                references = smooths if model == 1 else lbps if model == 2 else gts
                reference_arrays = references.detach().cpu().numpy()
                mask_arrays = masks.detach().cpu().numpy()
                raw_display = self.postprocess(outputs) * 255.0
                outputs_merged = self.postprocess(outputs_merged) * 255.0
                inputs_show = inputs + masks

                for i in range(outputs_merged.size(0)):
                    name = test_dataset.load_name(index, self.debug)
                    print(index, name)
                    path = os.path.join(self.config.DATA_TEST_RESULTS, name)
                    imsave(raw_display[i:i+1], path)
                    imsave(outputs_merged[i:i+1], os.path.join(completed_dir, name))
                    np.save(os.path.join(tensor_dir, name + '.npy'), raw_arrays[i])
                    error = np.abs(raw_arrays[i] - reference_arrays[i]).mean(axis=0)
                    missing = mask_arrays[i, 0] > 0
                    ratio = float(missing.mean())
                    whole = float(error.mean() * 100)
                    inside = float(error[missing].mean() * 100)
                    outside = float(error[~missing].mean() * 100)
                    metric_rows.append([name, ratio, whole, inside, outside,
                        whole - ratio * inside - (1-ratio) * outside])
                    mask_dir = self.config.DATA_TEST_RESULTS + '_masks'
                    create_dir(mask_dir)
                    imsave(masks[i:i+1].repeat(1,3,1,1) * 255, os.path.join(mask_dir, name))
                    index += 1

                    if self.debug and model == 3:
                        smooth_ = self.postprocess(smooth_stage_1[i, :, :, :].unsqueeze(0)) * 255.0
                        texture_ = self.postprocess(texture_stage_2[i, :, :, :].unsqueeze(0)) * 255.0
                        inputs_ = self.postprocess(inputs_show[i, :, :, :].unsqueeze(0)) * 255.0
                        gts_ = self.postprocess(gts[i, :, :, :].unsqueeze(0)) * 255.0
                        print(path)
                        fname, fext = os.path.splitext(path)
                        imsave(smooth_, fname + '_smooth.' + fext)
                        imsave(texture_, fname + '_texture.' + fext)
                        imsave(inputs_, fname + '_inputs.' + fext)
                        imsave(gts_, fname + '_gts.' + fext)

        with open(os.path.join(raw_dir, 'raw_pixel_metrics.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'mask_ratio', 'L1_whole_percent',
                'L1_mask_percent', 'L1_outside_percent', 'identity_residual_percent'])
            writer.writerows(metric_rows)
        with open(os.path.join(raw_dir, 'prediction_protocol.json'), 'w', encoding='utf-8') as f:
            json.dump({'prediction_kind': 'raw', 'raw_tensors': tensor_dir,
                'completed_display': completed_dir, 'pixel_metrics': 'continuous raw tensors',
                'feature_images': 'raw output clipped to [0,1] and exported to image files',
                'group_identity': 'mean(r_i*Lmask_i + (1-r_i)*Loutside_i)'}, f, indent=2)
        print('\nEnd test....')

    def obtain_log(self, config):
        log_dir = os.path.join(config.PATH, config.NAME, self.stage_name + '_log')
        if os.path.exists(log_dir) and config.REMOVE_LOG:
            shutil.rmtree(log_dir)
        train_writer = tensorboardX.SummaryWriter(log_dir)
        return train_writer

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def write_loss(self, logs, train_writer):
        iteration = [x[1] for x in logs if x[0] == 'iter']
        for x in logs:
            if x[0].startswith('l_'):
                train_writer.add_scalar(x[0], x[1], iteration[-1])

    def write_image(self, result, train_writer, iterations, label):
        if result:
            name = '%s/model%d_sample_%08d' % (self.samples_path, self.config.MODEL, iterations) + label + '.jpg'
            write_2images(result, self.config.SAMPLE_SIZE, name)
            write_2tensorboard(iterations, result, train_writer, self.config.SAMPLE_SIZE, label)

    def postprocess(self, x):
        return x.clamp(0, 1)

    def metrics(self, inputs, gts):
        # Evaluate continuous raw tensors; data_range is the reference range.
        psnr, ssim = [], []
        for pred, target in zip(inputs, gts):
            pred = pred.detach().cpu().numpy().transpose(1,2,0)
            target = target.detach().cpu().numpy().transpose(1,2,0)
            if min(pred.shape[:2]) < 11:
                raise ValueError('SSIM requires images at least 11 pixels in each dimension.')
            psnr.append(compare_psnr(target, pred, data_range=1))
            ssim.append(compare_ssim(target, pred, data_range=1, win_size=11, channel_axis=-1))
        return np.float64(np.mean(psnr)), np.float64(np.mean(ssim)), torch.mean(torch.abs(inputs-gts))
