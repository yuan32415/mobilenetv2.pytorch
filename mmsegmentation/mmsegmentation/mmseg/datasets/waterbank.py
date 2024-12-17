from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Waterbankset(BaseSegDataset):
    # METAINFO = dict(
    #     CLASSES=('background', 'oysterrow', 'land'),
    #     PALETTE = [[0, 0, 255], [215, 255, 255], [0, 255, 0]])
    #
    # def __init__(self,
    #              img_suffix='.png',
    #              seg_map_suffix='.png',
    #              **kwargs) -> None:
    #     super().__init__(
    #         img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    METAINFO = dict(
        # 写你实际的类别名就好了，跟生成mask是映射的数字顺序一致即可，有背景不需要改没有背景记得与生成mask时一样一定要在第一个加上background
        classes=('background', 'water'),
        # 调色板，这个数量与上面个数对应就好了,只是最后的预测每个类别对应的mask颜色
        # 分别是蓝色、白色、绿色
        palette=[[0, 0, 0], [128, 0, 0]]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 # reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            # reduce_zero_label=reduce_zero_label,
            **kwargs)
