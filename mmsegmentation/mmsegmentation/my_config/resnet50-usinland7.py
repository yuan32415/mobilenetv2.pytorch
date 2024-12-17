norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    # size=None,#512,512输出图像的大小。默认为None，表示不进行缩放
    size=(280,280),#512,512输出图像的大小。默认为None，表示不进行缩放
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    pretrained=None,
    backbone=dict(
        type='MY_ResNet',
        depth=50,
        num_stages=4,
        cbam_block=[True, True, True, True],  # 是否使用SE模块
        # cbam_block=[False, False, False, False],  # 是否使用SE模块
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # decode_head=dict(
    #     # type='ASPPHead',
    #     type='Myhead',
    #     # in_channels=2048,
    #     in_channels=[256,512,1024,2048],
    #     in_index=3,
    #     channels=512,
    #     dilations=(1, 12, 24, 36),
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
            # type='Myhead_noASPP',
            type='Myhead',
            # in_channels=2048,
            in_channels=[256,512,1024,2048],
            in_index=[0,1,2,3],
            channels=512,
            # dilations=(1, 12, 24, 36),
            dilations=(1,6,12,18),
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset settings
# dataset_type = 'ChaseDB1Dataset'
# data_root = 'data/USVInIand'
dataset_type = 'Waterbankset'
# data_root = 'data/water_bank'
data_root = 'data/USVInland'

img_scale = (640,400)

crop_size = (280, 280)
train_pipeline = [
    dict(type='LoadImageFromFile'),#从文件中加载图像。
    dict(type='LoadAnnotations'),#加载与图像对应的标注信息
    dict(
        type='RandomResize',#随机调整图像大小
        scale=img_scale,#目标图像大小
        ratio_range=(0.7, 2.0),#图像大小随机调整的范围
        keep_ratio=True),#在缩放过程中保持图像的宽高比。
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),#随机裁剪图像。裁剪后图像的大小crop_size，在进行随机裁剪时，裁剪区域最大可以占到原标注区域的75%。
    dict(type='RandomFlip', prob=0.5),#每次处理图像时有50%的概率进行翻转。
    dict(type='PhotoMetricDistortion'),#通过不同的光照条件增强图像，以提高模型的泛化能力。
    dict(type='PackSegInputs')#将处理后的图像和标注信息打包成适合模型输入的格式。
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    # batch_size=4,
    batch_size=8,
    num_workers=4,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=20000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                # img_path='images/training',
                # seg_map_path='annotations/training'
                img_path='img_dir/train', seg_map_path='ann_dir/train'
            ),
            pipeline=train_pipeline
                )))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='images/validation',
            # seg_map_path='annotations/validation'
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'
        ),
        pipeline=test_pipeline
    ))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='images/validation',
            # seg_map_path='annotations/validation'
            img_path='img_dir/test',
            seg_map_path='ann_dir/test'
        ),
        pipeline=test_pipeline))

# val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005) # 优化器种类，学习率，动量，权重衰减
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None) # 优化器包装器(Optimizer wrapper)，# optimizer用于更新模型参数的优化器(Optimizer)# 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数。
# learning policy 学习策略
param_scheduler = [
    dict(
        type='PolyLR',# 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等
        eta_min=1e-4,# 训练结束时的最小学习率
        power=0.9,# 多项式衰减 (polynomial decay) 的幂
        begin=0, # 开始更新参数的时间步(step)
        end=20000, # 停止更新参数的时间步(step)
        by_epoch=False) # 是否按照 epoch 计算训练时间
]

# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)# 训练配置，包括训练循环的类型，最大迭代次数，验证间隔
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(# 默认钩子(hook)配置
    timer=dict(type='IterTimerHook'),# 记录迭代过程中花费的时间
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False), # 从'Runner'的不同组件收集和写入日志
    param_scheduler=dict(type='ParamSchedulerHook'),# 更新优化器中的一些超参数，例如学习率
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000), # 定期保存检查点(checkpoint)
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200,max_keep_ckpts=3,save_best='mIoU'), # 定期保存检查点(checkpoint)
    sampler_seed=dict(type='DistSamplerSeedHook'),# 用于分布式训练的数据加载采样器
    visualization=dict(type='SegVisualizationHook')# 可视化训练过程中的预测结果
)

default_scope = 'mmseg'# 将注册表的默认范围设置为mmseg
env_cfg = dict(
    cudnn_benchmark=True,#使用cuDNN的基准测试来寻找最佳的算法实现，以提高性能。
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),#多进程配置，指定了多进程启动方法为fork，并将OpenCV的线程数设置为0，以避免多线程导致的性能下降。
    dist_cfg=dict(backend='nccl'),#分布式训练配置，指定了使用的后端为nccl，即NVIDIA的集体通信库。
)
vis_backends = [dict(type='LocalVisBackend')]#定义了一个可视化后端列表，这里只包含了一个本地可视化后端。
# visualizer = dict(#配置了可视化组件，
#     type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')#指定了可视化器的类型为SegLocalVisualizer，这可能是用于语义分割的特定可视化器。
visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])#配置了可视化组件TensorboardVisBackend，
log_processor = dict(by_epoch=False)#配置了日志处理器，其中by_epoch=False表示日志处理是按迭代次数而不是按时代数。
log_level = 'INFO'#设置了日志记录的级别为INFO，这意味着只有信息性的消息和以上级别的消息（如警告和错误）会被记录。
load_from = None#指定了从哪个文件加载预训练模型，这里设置为None，意味着不从文件加载。
resume = False
tta_model = dict(type='SegTTAModel')#配置了测试时增强（TTA）模型，type='SegTTAModel'表示使用了一种用于语义分割的TTA模型。
