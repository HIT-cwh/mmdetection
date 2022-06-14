_base_ = './gfl_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='s3://caoweihan/pretrained/cls/resnet101-63fe2227.pth')))

checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='s3://caoweihan/mmdet')
