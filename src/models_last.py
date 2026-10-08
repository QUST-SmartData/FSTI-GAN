import torch
import torch.nn as nn
from .base_model import BaseModel
from .paper_networks import TextureGen
from .paper_networks import StructureGen
from .paper_networks import InpaintingGen
from .paper_networks import SMPatchDiscriminator
from .loss import StyleLoss, PerceptualLoss
from .soft_mask_loss import SoftMaskAdversarialLoss

# ---------只改models里面的from就可以了

class TextureFlowModel(BaseModel):
    def __init__(self, config):
        super(TextureFlowModel, self).__init__('TextureFlow', config)
        self.config = config
        if config.DIS_GAN_LOSS != 'smgan':
            raise ValueError('Formal manuscript configuration requires SM-PatchGAN soft-mask loss.')
        self.net_name = ['s_gen', 's_dis', 't_gen', 't_dis', 'i_gen', 'i_dis']

        self.structure_param = {'input_dim': 3, 'dim': 64, 'n_res': 1, 'activ': 'relu',
                                'norm': 'in', 'pad_type': 'reflect', 'use_sn': True}
        self.texture_param = {'input_dim': 3, 'dim': 64, 'n_res': 1, 'activ': 'relu',
                              'norm': 'in', 'pad_type': 'reflect', 'use_sn': True}
        self.inpaint_param = {'input_dim': 3, 'dim': 64, 'n_res': 2, 'activ': 'relu',
                              'norm_conv': 'ln', 'norm_flow': 'in', 'pad_type': 'reflect', 'use_sn': False,
                              'fst_blocks': config.FSTI_BLOCKS or 8}
        self.dis_param1 = {'input_dim': 3, 'dim': 64, 'n_layers': 3,
                           'norm': 'none', 'activ': 'lrelu', 'pad_type': 'reflect', 'use_sn': True}
        self.dis_param2 = {'input_dim': 1, 'dim': 64, 'n_layers': 3,
                           'norm': 'none', 'activ': 'lrelu', 'pad_type': 'reflect', 'use_sn': True}

        l1_loss = nn.L1Loss()
        adversarial_loss = SoftMaskAdversarialLoss()
        if config.USE_CORRECTION_LOSS:
            raise ValueError('The paper architecture does not include a flow-correctness branch.')
        self.use_correction_loss = False
        self.use_vgg_loss = config.MODEL == 3
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        if self.use_vgg_loss:
            self.add_module('vgg_style', StyleLoss())
            self.add_module('vgg_content', PerceptualLoss())

        self.build_model()

    def build_model(self):
        self.iterations = 0
        # structure model
        if self.config.MODEL == 1:
            self.s_gen = StructureGen(**self.structure_param)
            self.s_dis = SMPatchDiscriminator(**self.dis_param1)
            # self.t_dis = NLayerDiscriminator()
        # flow model with true input smooth
        elif self.config.MODEL == 2:
            self.t_gen = TextureGen(**self.texture_param)
            self.t_dis = SMPatchDiscriminator(**self.dis_param2)
        # flow model with fake input smooth
        elif self.config.MODEL == 3:
            self.s_gen = StructureGen(**self.structure_param)
            self.t_gen = TextureGen(**self.texture_param)
            self.i_gen = InpaintingGen(**self.inpaint_param)
            self.i_dis = SMPatchDiscriminator(**self.dis_param1)

        self.define_optimizer()
        self.init()

    def structure_forward(self, inputs, smooths, masks):
        smooths_input = smooths * (1 - masks)
        outputs = self.s_gen(torch.cat((inputs, smooths_input, masks), dim=1))
        return outputs

    def texture_forward(self, inputs, lbps, masks):
        lbps_input = lbps * (1 - masks)
        outputs = self.t_gen(torch.cat((inputs, lbps_input, masks), dim=1))
        return outputs

    def inpaint_forward(self, inputs, smooths_stage_1, lbps_stage_2, masks):
        outputs, lbps = self.i_gen(torch.cat((inputs, smooths_stage_1, masks), dim=1), smooths_stage_1, lbps_stage_2)
        return outputs, lbps

    def train(self, mode=True):
        super().train(mode)
        if self.config.MODEL == 3:
            self.s_gen.eval()
            self.t_gen.eval()
        return self

    def reconstruct_priors(self, inputs, smooths, lbps, masks):
        with torch.no_grad():
            sr = self.structure_forward(inputs, smooths, masks)
            tr = self.texture_forward(inputs, lbps, masks)
            return sr * masks + smooths * (1 - masks), tr * masks + lbps * (1 - masks)

    def sample(self, inputs, smooths, lbps, gts, masks):
        with torch.no_grad():
            if self.config.MODEL == 1:
                outputs = self.structure_forward(inputs, smooths, masks)
                result = [inputs, smooths, gts, masks, outputs]

            elif self.config.MODEL == 2:
                outputs = self.texture_forward(inputs, lbps, masks)
                result = [inputs, lbps, gts, masks, outputs]

            elif self.config.MODEL == 3:
                smooth_stage_1, texture_stage_1 = self.reconstruct_priors(inputs, smooths, lbps, masks)
                outputs, lbp = self.inpaint_forward(inputs, smooth_stage_1, texture_stage_1, masks)
                result = [inputs, smooths, lbps, gts, masks, smooth_stage_1, texture_stage_1, outputs]

        return result

    def update_structure(self, inputs, smooths, masks):
        self.iterations += 1

        self.s_gen.zero_grad()
        self.s_dis.zero_grad()
        outputs = self.structure_forward(inputs, smooths, masks)
        completed = outputs * masks + smooths * (1-masks)

        dis_loss = 0
        dis_fake_input = completed.detach()
        dis_real_input = smooths
        fake_labels = self.s_dis(dis_fake_input)
        real_labels = self.s_dis(dis_real_input)
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], masks, True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], masks, False, True)
            dis_loss += dis_real_loss + dis_fake_loss
        self.structure_adv_dis_loss = dis_loss / len(fake_labels)

        self.structure_adv_dis_loss.backward()
        self.s_dis_opt.step()
        if self.s_dis_scheduler is not None:
            self.s_dis_scheduler.step()

        dis_gen_loss = 0
        fake_labels = self.s_dis(completed)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], masks, True, False)
            dis_gen_loss += dis_fake_loss
        self.structure_adv_gen_loss = dis_gen_loss / len(fake_labels) * self.config.STRUCTURE_ADV_GEN
        self.structure_l1_loss = self.l1_loss(outputs, smooths) * self.config.STRUCTURE_L1
        self.structure_gen_loss = self.structure_l1_loss + self.structure_adv_gen_loss

        self.structure_gen_loss.backward()
        self.s_gen_opt.step()
        if self.s_gen_scheduler is not None:
            self.s_gen_scheduler.step()

        logs = [
            ("l_s_adv_dis", self.structure_adv_dis_loss.item()),
            ("l_s_l1", self.structure_l1_loss.item()),
            ("l_s_adv_gen", self.structure_adv_gen_loss.item()),
            ("l_s_gen", self.structure_gen_loss.item()),
        ]
        return logs

    def update_texture(self, inputs, lbps, maps):
        self.iterations += 1
        self.t_dis.zero_grad()
        self.t_gen.zero_grad()
        outputs = self.texture_forward(inputs, lbps, maps)
        completed = outputs * maps + lbps * (1-maps)
        fake_labels = self.t_dis(completed.detach())
        real_labels = self.t_dis(lbps)
        self.texture_adv_dis_loss = sum(
            self.adversarial_loss(real,maps,True,True)+self.adversarial_loss(fake,maps,False,True)
            for real,fake in zip(real_labels,fake_labels))/len(fake_labels)
        self.texture_adv_dis_loss.backward()
        self.t_dis_opt.step()
        if self.t_dis_scheduler is not None: self.t_dis_scheduler.step()
        fake_labels = self.t_dis(completed)
        self.texture_adv_gen_loss = sum(self.adversarial_loss(fake,maps,True,False)
            for fake in fake_labels)/len(fake_labels)*self.config.TR_ADV_GEN
        # Eq. 7: extract phi_l from G2, conditioned on the same Id and M.
        # Replacing the texture input by Tr or Tg isolates the texture difference.
        _, fake_features = self.t_gen(torch.cat((inputs,completed,maps),1),return_features=True)
        with torch.no_grad():
            _, real_features = self.t_gen(torch.cat((inputs,lbps,maps),1),return_features=True)
        self.texture_l1_loss = torch.linalg.vector_norm((completed-lbps).flatten(1),dim=1).mean()*self.config.TR_RECON
        self.texture_multilevel_loss = sum(torch.linalg.vector_norm((fake-real.detach()).flatten(1),dim=1).mean()
            for fake,real in zip(fake_features,real_features))*self.config.TR_MULTILEVEL
        self.texture_gen_loss = self.texture_l1_loss+self.texture_adv_gen_loss+self.texture_multilevel_loss
        self.texture_gen_loss.backward()
        self.t_gen_opt.step()
        if self.t_gen_scheduler is not None: self.t_gen_scheduler.step()
        return [('l_t_adv_dis',self.texture_adv_dis_loss.item()),('l_t_l2',self.texture_l1_loss.item()),
                ('l_t_multilevel',self.texture_multilevel_loss.item()),('l_t_adv_gen',self.texture_adv_gen_loss.item()),
                ('l_t_gen',self.texture_gen_loss.item())]

    def update_inpaint(self, inputs, smooths, lbps, gts, masks, use_correction_loss, use_vgg_loss):
        self.iterations += 1

        self.i_dis.zero_grad()
        self.i_gen.zero_grad()
        outputs, lbp_maps = self.inpaint_forward(inputs, smooths, lbps, masks)
        outputs = outputs * masks + gts * (1-masks)

        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = gts
        fake_labels = self.i_dis(dis_fake_input)
        real_labels = self.i_dis(dis_real_input)
        # self.flow_adv_dis_loss = dis_real_loss + dis_fake_loss
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], masks, True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], masks, False, True)
            dis_loss += dis_real_loss + dis_fake_loss
        self.lbp_adv_dis_loss = dis_loss / len(fake_labels)

        self.lbp_adv_dis_loss.backward()
        self.i_dis_opt.step()
        if self.i_dis_scheduler is not None:
            self.i_dis_scheduler.step()

        dis_gen_loss = 0
        fake_labels = self.i_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], masks, True, False)
            dis_gen_loss += dis_fake_loss
        self.lbp_adv_gen_loss = dis_gen_loss / len(fake_labels) * self.config.FLOW_ADV_GEN
        self.lbp_l1_loss = self.l1_loss(outputs, gts) * self.config.FLOW_L1
        self.lbp_correctness_loss = self.correctness_loss(gts, inputs, lbp_maps, masks) * \
                                    self.config.FLOW_CORRECTNESS if use_correction_loss else 0

        if use_vgg_loss:
            self.vgg_loss_style = self.vgg_style(outputs * masks, gts * masks) * self.config.VGG_STYLE
            self.vgg_loss_content = self.vgg_content(outputs, gts) * self.config.VGG_CONTENT
            self.vgg_loss = self.vgg_loss_style + self.vgg_loss_content
        else:
            self.vgg_loss = 0

        self.lbp_loss = self.lbp_adv_gen_loss + self.lbp_l1_loss + self.lbp_correctness_loss + self.vgg_loss

        self.lbp_loss.backward()
        self.i_gen_opt.step()

        if self.i_gen_scheduler is not None:
            self.i_gen_scheduler.step()

        logs = [
            ("l_lbp_adv_dis", self.lbp_adv_dis_loss.item()),
            ("l_lbp_adv_gen", self.lbp_adv_gen_loss.item()),
            ("l_lbp_l1_gen", self.lbp_l1_loss.item()),
            ("l_lbp_total_gen", self.lbp_loss.item()),
        ]
        if use_correction_loss:
            logs = logs + [("l_lbp_correctness_gen", self.lbp_correctness_loss.item())]
        if use_vgg_loss:
            logs = logs + [("l_lbp_vgg_style", self.vgg_loss_style.item())]
            logs = logs + [("l_lbp_vgg_content", self.vgg_loss_content.item())]
        return logs
