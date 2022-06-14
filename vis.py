import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv.runner.checkpoint import _load_checkpoint
import copy
# ckpt = _load_checkpoint('epoch_12.pth', map_location='cpu')
# for key in ckpt['state_dict'].keys():
#     print(key)
# new_ckpt = dict()
# new_ckpt['meta'] = ckpt['meta']
# new_ckpt['optimizer'] = ckpt['optimizer']
# new_ckpt['state_dict'] = dict()
# for key, val in ckpt['state_dict'].items():
#     if 'teacher' in key:
#         continue
#     print(key)
#     kl = '.'.join(key.split('.')[2:])
#     print(kl)
#     new_ckpt['state_dict'][kl] = val
#     break
# torch.save(new_ckpt, 'fcos_retina_finetune.pth')
# input()

device = 'cpu'
img = 'demo/demo.jpg'

# checkpoint_file = r'G:\projects\research\checkpoint\mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
# model = init_detector('configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

# checkpoint_file = r'G:\projects\research\checkpoint\yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
# model = init_detector('configs/yolox/yolox_x_8x8_300e_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

# checkpoint_file = r'G:\projects\research\checkpoint\yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
# model = init_detector('configs/yolox/yolox_tiny_8x8_300e_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

# checkpoint_file = r'G:\projects\research\checkpoint\faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
# model = init_detector('configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py', checkpoint_file, device=device)
# result = inference_detector(model, img)

# config_file = 'configs/gfl/gfl_r50_fpn_1x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
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

# config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
# checkpoint_file = 'retina_r50_fcos_distill.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
# r50 = [0, 1, 19, 117, 613, 3683, 20234, 113776, 493418, 1272914, 1362976, 506045, 97168, 16587, 3040, 498, 88, 22, 1, 0]
# show_result_pyplot(model, img, result, score_thr=0.3)

# config_file = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
# checkpoint_file = r'G:\projects\research\checkpoint\retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)

config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
model = init_detector(config_file, checkpoint_file, device=device)
result = inference_detector(model, img)

# config_file = 'configs/retinanet/retinanet_r101_fpn_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
# r101 = [0, 10, 57, 329, 1703, 7584, 34291, 165683, 575532, 1175573, 1139809, 582280, 165769, 34602, 6420, 1274, 234, 43, 6, 1]
# show_result_pyplot(model, img, result, score_thr=0.3)

# config_file = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
# checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth'
# model = init_detector(config_file, checkpoint_file, device=device)
# result = inference_detector(model, img)
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
