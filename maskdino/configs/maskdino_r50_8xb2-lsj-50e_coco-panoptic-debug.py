_base_ = './maskdino_r50_8xb2-lsj-50e_coco-panoptic.py'

train_dataloader = dict(
    dataset=dict(
        ann_file=_base_.val_dataloader.dataset.ann_file,
        data_prefix=_base_.val_dataloader.dataset.data_prefix))
