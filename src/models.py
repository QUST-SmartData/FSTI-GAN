import torch
import torch.nn as nn
from .base_model import BaseModel
from .network_GatedConv2d import TextureGen
from .network_GatedConv2d import StructureGen
from .network_GatedConv2d import InpaintingGen
from .network_GatedConv2d import MultiDiscriminator
from .loss import AdversarialLoss, PerceptualCorrectness, StyleLoss, PerceptualLoss

# ---------只改models里面的from就可以了

class TextureFlowModel(BaseModel):
    def __init__(self, config):
        super(TextureFlowModel, self).__init__('TextureFlow', config)
        self.config = config
        self.net_name = ['s_gen', 's_dis', 't_gen', 't_dis', 'i_gen', 'i_dis']

        self.structure_param = {'input_dim': 3, 'dim': 64, 'n_res': 4, 'activ': 'relu',
                                'norm': 'in', 'pad_type': 'reflect', 'use_sn': True}
        self.texture_param = {'input_dim': 3, 'dim': 64, 'n_res': 4, 'activ': 'relu',
                              'norm': 'in', 'pad_type': 'reflect', 'use_sn': True}
        self.inpaint_param = {'input_dim': 3, 'dim': 64, 'n_res': 2, 'activ': 'relu',
                              'norm_conv': 'ln', 'norm_flow': 'in', 'pad_type': 'reflect', 'use_sn': False}
        self.dis_param1 = {'input_dim': 3, 'dim': 64, 'n_layers': 3,
                           'norm': 'none', 'activ': 'lrelu', 'pad_type': 'reflect', 'use_sn': True}
        self.dis_param2 = {'input_dim': 1, 'dim': 64, 'n_layers': 3}

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.DIS_GAN_LOSS)
        correctness_loss = PerceptualCorrectness()
        criterion_loss = GANLoss(gan_type='vanilla')
        criterionL2_loss = torch.nn.MSELoss()
        vgg_style = StyleLoss()
        vgg_content = PerceptualLoss()
        self.use_correction_loss = True
        self.use_vgg_loss = True if self.config.MODEL == 3 else False

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('correctness_loss', correctness_loss)
        self.add_module('criterion_loss', criterion_loss)
        self.add_module('criterionL2_loss', criterionL2_loss)
        self.add_module('vgg_style', vgg_style)
        self.add_module('vgg_content', vgg_content)

        self.build_model()

    def build_model(self):
        self.iterations = 0
        # structure model
        if self.config.MODEL == 1:
            self.s_gen = StructureGen(**self.structure_param)
            self.s_dis = MultiDiscriminator(**self.dis_param1)
            # self.t_dis = NLayerDiscriminator()
        # flow model with true input smooth
        elif self.config.MODEL == 2:
            self.t_gen = define_LBP(self.config)
            self.t_dis = define_D(1, 64, device=self.config.DEVICE)
        # flow model with fake input smooth
        elif self.config.MODEL == 3:
            self.s_gen = StructureGen(**self.structure_param)
            self.t_gen = define_LBP(**self.texture_param)
            self.i_gen = InpaintingGen(**self.inpaint_param)
            self.i_dis = MultiDiscriminator(**self.dis_param1)

        self.define_optimizer()
        self.init()

    def structure_forward(self, inputs, smooths, masks):
        smooths_input = smooths * (1 - masks)
        outputs = self.s_gen(torch.cat((inputs, smooths_input, masks), dim=1))
        return outputs

    # def texture_forward(self, inputs, lbps, masks):
    #     lbps_input = lbps * (1 - masks)
    #     outputs = self.t_gen(torch.cat((inputs, lbps_input, masks), dim=1))
    #     return outputs

    def texture_forward(self, lbps, masks):
        outputs, outputs_features = self.t_gen(lbps, masks)
        return outputs, outputs_features

    def inpaint_forward(self, inputs, smooths_stage_1, lbps_stage_2, masks):
        outputs, lbps = self.i_gen(torch.cat((inputs, smooths_stage_1, masks), dim=1), smooths_stage_1, lbps_stage_2)
        return outputs, lbps

    def sample(self, inputs, smooths, lbps, gts, masks):
        with torch.no_grad():
            if self.config.MODEL == 1:
                outputs = self.structure_forward(inputs, smooths, masks)
                result = [inputs, smooths, gts, masks, outputs]

            elif self.config.MODEL == 2:
                outputs = self.texture_forward(lbps, masks)
                result = [inputs, lbps, gts, masks, outputs]

            elif self.config.MODEL == 3:
                smooth_stage_1 = self.structure_forward(inputs, smooths, masks)
                texture_stage_1 = self.texture_forward(inputs, lbps, masks)
                outputs, lbp = self.inpaint_forward(inputs, smooth_stage_1, texture_stage_1, masks)
                result = [inputs, smooths, lbps, gts, masks, smooth_stage_1, texture_stage_1, outputs]

        return result

    def update_structure(self, inputs, smooths, masks):
        self.iterations += 1

        self.s_gen.zero_grad()
        self.s_dis.zero_grad()
        outputs = self.structure_forward(inputs, smooths, masks)

        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = smooths
        fake_labels = self.s_dis(dis_fake_input)
        real_labels = self.s_dis(dis_real_input)
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
        self.structure_adv_dis_loss = dis_loss / len(fake_labels)

        self.structure_adv_dis_loss.backward()
        self.s_dis_opt.step()
        if self.s_dis_scheduler is not None:
            self.s_dis_scheduler.step()

        dis_gen_loss = 0
        fake_labels = self.s_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], True, False)
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

    def update_texture(self, lbps, maps):
        self.iterations += 1

        # 清零梯度
        self.t_dis.zero_grad()
        self.t_gen.zero_grad()

        # 前向传播
        outputs, outputs_features = self.texture_forward(lbps, maps)

        dis_fake_input = outputs.detach()
        dis_real_input = lbps
        fake_labels = self.t_dis(dis_fake_input)
        real_labels = self.t_dis(dis_real_input)

        # 生成器损失计算
        self.texture_loss_G_GAN = self.criterion_loss(fake_labels, True) * 0.2
        self.texture_loss_G_L2 = self.criterionL2_loss(outputs, dis_real_input) * 100
        self.texture_loss_G = self.texture_loss_G_GAN + self.texture_loss_G_L2

        # 反向传播生成器损失并更新参数
        self.texture_loss_G.backward()
        self.t_gen_opt.step()

        # 判别器损失计算（使用先前计算的 fake_labels 和 real_labels）
        self.texture_loss_D_fake = self.criterion_loss(fake_labels, False)
        self.texture_loss_D_real = self.criterion_loss(real_labels, True)
        self.texture_loss_D = (self.texture_loss_D_fake + self.texture_loss_D_real) * 0.5

        # 反向传播判别器损失并更新参数
        self.texture_loss_D.backward()
        self.t_dis_opt.step()

        # 记录损失
        logs = [
            ("texture_loss_G_GAN", self.texture_loss_G_GAN.item()),
            ("texture_loss_G_L2", self.texture_loss_G_L2.item()),
            ("texture_loss_G", self.texture_loss_G.item()),
            ("texture_loss_D", self.texture_loss_D.item()),
        ]
        return logs

    def update_inpaint(self, inputs, smooths, lbps, gts, masks, use_correction_loss, use_vgg_loss):
        self.iterations += 1

        self.i_dis.zero_grad()
        self.i_gen.zero_grad()
        outputs, lbp_maps = self.inpaint_forward(inputs, smooths, lbps, masks)

        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = gts
        fake_labels = self.i_dis(dis_fake_input)
        real_labels = self.i_dis(dis_real_input)
        # self.flow_adv_dis_loss = (dis_real_loss + dis_fake_loss) / 2
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
        self.lbp_adv_dis_loss = dis_loss / len(fake_labels)

        self.lbp_adv_dis_loss.backward()
        self.i_dis_opt.step()
        if self.i_dis_scheduler is not None:
            self.i_dis_scheduler.step()

        dis_gen_loss = 0
        fake_labels = self.i_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], True, False)
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
