_base_ = './maskdino_r50_8xb2-lsj-50e_coco-panoptic.py'

custom_imports = None
model = dict(
    data_preprocessor=None,  # no clue why `'DetDataPreprocessor is not in the model registry` would be raised
    panoptic_head=dict(decoder=dict(dn='no')))
