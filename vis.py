import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv.runner.checkpoint import _load_checkpoint
import copy

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


class ModuleOutputsRecorder:

    def __init__(self, sources):
        super().__init__()
        self.recording = True
        self.data_buffer = dict()
        self.sources = sources

    def prepare_from_model(self, model):
        self.module2name = {}

        for module_name, module in model.named_modules():
            self.module2name[module] = module_name
        self.name2module = dict(model.named_modules())

        for module_name in self.sources:
            self.data_buffer[module_name] = list()
            module = self.name2module[module_name]
            module.register_forward_hook(self.forward_output_hook)

    def reset_data_buffer(self):
        for key in self.data_buffer.keys():
            self.data_buffer[key] = list()

    def forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        print(module)
        if self.recording:
            module_name = self.module2name[module]
            self.data_buffer[module_name].append(outputs)

def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / std
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

def to255(feat, mmin=None, mmax=None):
    if mmin is None:
        mmax = np.max(feat)
        mmin = np.min(feat)
    # mmax, mmin = 10, -10
    k = (255 - 0) / (mmax - mmin)
    normed = 0 + k * (feat - mmin)
    return np.clip(normed.astype(int), 0, 255)
    # return torch.clamp(normed.int(), 0, 255).cpu().numpy()


def convert_overlay_heatmap(feat_map, img, alpha = 0.5, mmin=None, mmax=None):
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    if mmax is None:
        norm_img = np.zeros(feat_map.shape)
        norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
        # print(norm_img)
        # print(feat_map.min(), feat_map.max())
    else:
        norm_img = to255(feat_map, mmin, mmax)
        # print(norm_img)
    print(norm_img.max())
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


def resize(feat, ori_size, img_size, pad_size):
    ph, pw = pad_size[:2]
    ih, iw = img_size[:2]
    fh, fw = feat.shape[-2:]
    rh, rw = ih / ph, iw / pw
    h, w = round(fh * rh), round(fw * rw)
    feat = feat[..., :h, :w]
    return F.interpolate(feat, ori_size[:2], mode='bilinear')


def vis(checkpoint_file, cfg_path, place='neck', use_same_minmax=True, use_norm=True):
    device = 'cpu'
    img_path = 'demo/demo.jpg'

    model = init_detector(cfg_path, checkpoint_file, device=device)
    recorder = ModuleOutputsRecorder([place])
    recorder.prepare_from_model(model)
    print(model.bbox_head.multi_level_conv_cls)
    result, img_metas = inference_detector(model, img_path)
    # show_result_pyplot(model, img_path, result, score_thr=0.3)
    ori_shape = img_metas[0]['ori_shape']
    img_shape = img_metas[0]['img_shape']
    pad_shape = img_metas[0]['pad_shape']
    print(img_metas)
    print(recorder.data_buffer)
    outs = list(recorder.data_buffer[place][0])
    for i in range(len(outs)):
        outs[i] = outs[i].detach()

    mmin, mmax = 100, -100
    img = cv2.imread(img_path)
    size = img.shape[:2]
    for i, out in enumerate(outs):
        if use_norm:
            out = norm(out)
        out = resize(out, ori_shape, img_shape, pad_shape)
        act_max = torch.max(out, dim=1)[0]
        mmin = min(mmin, act_max.min())
        mmax = max(mmax, act_max.max())
    print(mmin, mmax)

    for i, out in enumerate(outs):
        if use_norm:
            out = norm(out)
        out = resize(out, ori_shape, img_shape, pad_shape)
        act_max = torch.max(out, dim=1)[0]
        if use_same_minmax:
            act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6,
                                              mmin=mmin.item(),
                                              mmax=mmax.item())
        else:
            act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6,
                                              mmin=act_max.min().item(),
                                              mmax=act_max.max().item())
        plt.axis('off')
        plt.imshow(act_max, cmap='Reds')
        plt.show()
        # plt.savefig(f'gfl2/gfl_r101_fpn{i}.png', bbox_inches='tight',
        #             pad_inches=0)


# checkpoint_file = r'G:\projects\research\checkpoint\mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
# model = init_detector('configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

# checkpoint_file = r'G:\projects\research\checkpoint\yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
# cfg_path = 'configs/yolox/yolox_x_8x8_300e_coco.py'
# model = init_detector('configs/yolox/yolox_x_8x8_300e_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

checkpoint_file = r'G:\projects\research\checkpoint\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
cfg_path = 'configs/yolox/yolox_s_8x8_300e_coco.py'

# checkpoint_file = r'G:\projects\research\checkpoint\yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
# cfg_path = 'configs/yolox/yolox_tiny_8x8_300e_coco.py'
vis(checkpoint_file, cfg_path, place='bbox_head', use_same_minmax=False, use_norm=False)
# input()

# checkpoint_file = r'G:\projects\research\checkpoint\faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
# model = init_detector('configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py', checkpoint_file, device=device)
# result_frcnn = inference_detector(model, img)

# config_file = 'configs/gfl/gfl_r50_fpn_1x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
# vis(checkpoint_file, config_file)
input()
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
#

# checkpoint_file = r'G:\projects\research\checkpoint\faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
# model = init_detector('configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py', None, device=device)
# result = inference_detector(model, img)

# checkpoint_file = r'G:\projects\research\checkpoint\faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
# model = init_detector('configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)
#
# config_file = 'configs/tood/tood_x101_64x4d_fpn_mstrain_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/tood/tood_x101_64x4d_fpn_mstrain_2x_coco/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)

# config_file = 'configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
#
# config_file = 'configs/fcos/fcos_r50_fpn_gn-head_1x_coco_fix_img_norm.py'
# checkpoint_file = 'epoch_12.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
#
# config_file = 'configs/finetune_head/fcos_r50_retina_1x1_finetune.py'
# checkpoint_file = 'fcos_retina_finetune.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)

# config_file = 'configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)

config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
checkpoint_file = 'retina_r50_frcnn_distill.pth'
model = init_detector(config_file, checkpoint_file, device=device)
result_retina = inference_detector(model, img)

# config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
# checkpoint_file = 'retina_r50_fcos_distill.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result_retina = inference_detector(model, img)
# r50 = [0, 1, 19, 117, 613, 3683, 20234, 113776, 493418, 1272914, 1362976, 506045, 97168, 16587, 3040, 498, 88, 22, 1, 0]
# show_result_pyplot(model, img, result, score_thr=0.3)

# config_file = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
# checkpoint_file = r'G:\projects\research\checkpoint\retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)

# config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)


# config_file = 'configs/retinanet/retinanet_r101_fpn_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
# r101 = [0, 10, 57, 329, 1703, 7584, 34291, 165683, 575532, 1175573, 1139809, 582280, 165769, 34602, 6420, 1274, 234, 43, 6, 1]
# show_result_pyplot(model, img, result, score_thr=0.3)

# config_file = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result_fcos = inference_detector(model, img)
# fx101 = [0, 0, 2, 26, 152, 1011, 5341, 32296, 242077, 1629713, 1695624, 245846, 32826, 5321, 841, 111, 11, 1, 1, 0]
# show_result_pyplot(model, img, result, score_thr=0.3)


# import matplotlib.pyplot as plt
# import numpy as np
#
# r50 = [8686, 8593, 17927, 35360, 67877, 123579, 215760, 343218, 484529, 599246, 628961, 529387, 367524, 223313, 119836, 59533, 29335, 14238, 7070, 7228]
# r101 = [17590, 14413, 28441, 53491, 95722, 159811, 249151, 363313, 464398, 514432, 503411, 447297, 352130, 253586, 165665, 96697, 52615, 28403, 14954, 15680]
# fx101 = [0, 0, 2, 26, 152, 1011, 5341, 32296, 242077, 1629713, 1695624, 245846, 32826, 5321, 841, 111, 11, 1, 1, 0]
#
#
# plt.bar(np.arange(-2., 2., 0.2), r50)
# plt.yticks(np.arange(0, 1.8e6, 2e5), fontsize=12)
# plt.xlabel('feature map ranges', fontsize=18)
# plt.ylabel('number of elements', fontsize=18)
# plt.savefig('r50_distribution.jpg')
