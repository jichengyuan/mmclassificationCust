_base_ = [
    '../_base_/datasets/mixed_datasets.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

dataset_type = 'MixedDatasets'
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
classes = ('tench', 'cassette_player', 'church', 'garbage_truck', 'golf_ball', 'parachute')
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='./train',
        ann_file='./meta/train.txt',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        data_prefix='./val',
        ann_file='./meta/val.txt',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(

        type=dataset_type,
        data_prefix='./val',
        ann_file='./val.txt',
        pipeline=test_pipeline,
        classes=classes))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=1, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = None
resume_from = None
workflow = [('train', 1)]
