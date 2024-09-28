from torch.autograd import Variable, grad
import numpy as np
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

        if self.config.MODEL == 1:
            self.stage_name = 'structure'
        elif self.config.MODEL == 2:
            self.stage_name = 'texture'
        elif self.config.MODEL == 3:
            self.stage_name = 'joint_train'

        self.debug = False
        self.inpaint_model = TextureFlowModel(config).to(config.DEVICE)
        self.samples_path = os.path.join(config.PATH, config.NAME, 'images')
        self.checkpoints_path = os.path.join(config.PATH, config.NAME, 'checkpoints')
        self.test_image_path = os.path.join(config.PATH, config.NAME, 'test_result')

        if self.config.MODE == 'train' and not self.config.RESUME_ALL:
            pass
        else:
            self.inpaint_model.load(self.config.WHICH_ITER)

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
                              self.config, self.config.DATA_MASK_FILE)
        sample_iterator = val_dataset.create_iterator(self.config.SAMPLE_SIZE)

        iterations = self.inpaint_model.iterations
        total = len(train_dataset)
        epoch = math.floor(iterations * self.config.TRAIN_BATCH_SIZE / total)
        keep_training = True
        model = self.config.MODEL
        max_iterations = int(float(self.config.MAX_ITERS))

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
                        smooth_stage_1 = self.inpaint_model.structure_forward(inputs, smooths, masks)
                        texture_stage_2 = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    logs = self.inpaint_model.update_inpaint(inputs, smooth_stage_1.detach(), texture_stage_2.detach(),
                                                             gts, masks, self.inpaint_model.use_correction_loss,
                                                             self.inpaint_model.use_vgg_loss)
                    iterations = self.inpaint_model.iterations

                if iterations >= max_iterations:
                    keep_training = False
                    break

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
        print('\nEnd training....')

    def eval(self, writer=None):
        val_dataset = Dataset(self.config.DATA_VAL_GT,
                              self.config.DATA_VAL_STRUCTURE,
                              self.config,
                              self.config.DATA_VAL_MASK)
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
            for j, items in enumerate(islice(val_loader, 50)):

                logs = []
                iteration += 1
                inputs, smooths, lbps, gts, masks = self.cuda(*items)
                if model == 1:
                    outputs_structure = self.inpaint_model.structure_forward(inputs, smooths, masks)
                    psnr, ssim, l1 = self.metrics(outputs_structure, smooths)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.append(psnr.item())

                # inpaint model
                elif model == 2:
                    outputs_texture = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    psnr, ssim, l1 = self.metrics(outputs_texture, lbps)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.append(psnr.item())


                # inpaint with structure model
                elif model == 3:
                    smooth_stage_1 = self.inpaint_model.structure_forward(inputs, smooths, masks)
                    texture_stage_2 = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    outputs, lbp = self.inpaint_model.inpaint_forward(inputs, smooth_stage_1.detach(),
                                                                      texture_stage_2.detach(), masks)
                    psnr, ssim, l1 = self.metrics(outputs, gts)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.append(psnr.item())

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
                               self.config, self.config.DATA_TEST_MASK)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=16,
        )

        index = 0
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
                    smooth_stage_1 = self.inpaint_model.structure_forward(inputs, smooths, masks)
                    texture_stage_2 = self.inpaint_model.texture_forward(inputs, lbps, masks)
                    outputs, lbp = self.inpaint_model.inpaint_forward(inputs, smooth_stage_1.detach(),
                                                                   texture_stage_2.detach(), masks)
                    outputs_merged = (outputs * masks) + (gts * (1 - masks))

                outputs_merged = self.postprocess(outputs_merged) * 255.0
                inputs_show = inputs + masks

                for i in range(outputs_merged.size(0)):
                    name = test_dataset.load_name(index, self.debug)
                    print(index, name)
                    path = os.path.join(self.config.DATA_TEST_RESULTS, name)
                    imsave(outputs_merged[i, :, :, :].unsqueeze(0), path)
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
        x = (x + 1) / 2
        x.clamp_(0, 1)
        return x

    def metrics(self, inputs, gts):
        inputs = self.postprocess(inputs)
        gts = self.postprocess(gts)
        psnr_value = []
        l1_value = torch.mean(torch.abs(inputs - gts))

        [b, n, w, h] = inputs.size()
        inputs = (inputs * 255.0).int().float() / 255.0
        gts = (gts * 255.0).int().float() / 255.0

        for i in range(inputs.size(0)):
            inputs_p = inputs[i, :, :, :].cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            gts_p = gts[i, :, :, :].cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            psnr_value.append(compare_psnr(inputs_p, gts_p, data_range=1))

        psnr_value = np.average(psnr_value)
        inputs = inputs.view(b * n, w, h).cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        gts = gts.view(b * n, w, h).cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        ssim_value = compare_ssim(inputs, gts, data_range=1, win_size=51, multichannel=True)
        return psnr_value, ssim_value, l1_value
