
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class TKDataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        'classes':['background', 'zhuzun', '金刚杵', '宝瓶', '钵', '法轮', '莲花', '经书', '金刚铃', '嘎布拉碗','剑', '伞盖', '喀章嘎'],
        'palette':[[127,127,127], [0,0,50], [0,0,100], [0,0,150], [0,50,0],[0,100,0], [0,200,0], [50,0,0], [100,0,0], [150,0,0], [200,0,0], [30,30,30], [0,150,0]]
        # 'classes':['background', 'zhuzun', '金刚杵', '八辐法轮', '钵', '宝剑', '经书', '胜利幢', '金刚铃', '嘎巴拉碗','剑', '羯磨杵', '金刚钺刀', '莲花'],
        # 'palette':[[127,127,127], [0,0,50], [0,0,100], [0,0,150], [0,50,0],[0,100,0], [0,200,0], [50,0,0], [100,0,0], [150,0,0], [200,0,0], [30,30,30], [0,150,0], [0,0,200]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)