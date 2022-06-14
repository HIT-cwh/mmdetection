_base_ = './gfl_r101_fpn_1x_coco.py'

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
