# # Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
# from collections import OrderedDict
# from typing import Dict, List, Optional, Sequence

# import numpy as np
# import torch
# from mmengine.dist import is_main_process
# from mmengine.evaluator import BaseMetric
# from mmengine.logging import MMLogger, print_log
# from mmengine.utils import mkdir_or_exist
# from PIL import Image
# from prettytable import PrettyTable

# from mmseg.registry import METRICS


# @METRICS.register_module()
# class IoUMetric(BaseMetric):
#     """IoU evaluation metric.

#     Args:
#         ignore_index (int): Index that will be ignored in evaluation.
#             Default: 255.
#         iou_metrics (list[str] | str): Metrics to be calculated, the options
#             includes 'mIoU', 'mDice' and 'mFscore'.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#         beta (int): Determines the weight of recall in the combined score.
#             Default: 1.
#         collect_device (str): Device name used for collecting results from
#             different ranks during distributed training. Must be 'cpu' or
#             'gpu'. Defaults to 'cpu'.
#         output_dir (str): The directory for output prediction. Defaults to
#             None.
#         format_only (bool): Only format result for results commit without
#             perform evaluation. It is useful when you want to save the result
#             to a specific format and submit it to the test server.
#             Defaults to False.
#         prefix (str, optional): The prefix that will be added in the metric
#             names to disambiguate homonymous metrics of different evaluators.
#             If prefix is not provided in the argument, self.default_prefix
#             will be used instead. Defaults to None.
#     """

#     def __init__(self,
#                  ignore_index: int = 255,
#                  iou_metrics: List[str] = ['mIoU'],
#                  nan_to_num: Optional[int] = None,
#                  beta: int = 1,
#                  collect_device: str = 'cpu',
#                  output_dir: Optional[str] = None,
#                  format_only: bool = False,
#                  prefix: Optional[str] = None,
#                  **kwargs) -> None:
#         super().__init__(collect_device=collect_device, prefix=prefix)

#         self.ignore_index = ignore_index
#         self.metrics = iou_metrics
#         self.nan_to_num = nan_to_num
#         self.beta = beta
#         self.output_dir = output_dir
#         if self.output_dir and is_main_process():
#             mkdir_or_exist(self.output_dir)
#         self.format_only = format_only

#     def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
#         """Process one batch of data and data_samples.

#         The processed results should be stored in ``self.results``, which will
#         be used to compute the metrics when all batches have been processed.

#         Args:
#             data_batch (dict): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of outputs from the model.
#         """
#         num_classes = len(self.dataset_meta['classes'])
#         for data_sample in data_samples:
#             pred_label = data_sample['pred_sem_seg']['data'].squeeze()
#             # format_only always for test dataset without ground truth
#             if not self.format_only:
#                 label = data_sample['gt_sem_seg']['data'].squeeze().to(
#                     pred_label)
#                 self.results.append(
#                     self.intersect_and_union(pred_label, label, num_classes,
#                                              self.ignore_index))
#             # format_result
#             if self.output_dir is not None:
#                 basename = osp.splitext(osp.basename(
#                     data_sample['img_path']))[0]
#                 png_filename = osp.abspath(
#                     osp.join(self.output_dir, f'{basename}.png'))
#                 output_mask = pred_label.cpu().numpy()
#                 # The index range of official ADE20k dataset is from 0 to 150.
#                 # But the index range of output is from 0 to 149.
#                 # That is because we set reduce_zero_label=True.
#                 if data_sample.get('reduce_zero_label', False):
#                     output_mask = output_mask + 1
#                 output = Image.fromarray(output_mask.astype(np.uint8))
#                 output.save(png_filename)

#     def compute_metrics(self, results: list) -> Dict[str, float]:
#         """Compute the metrics from processed results.

#         Args:
#             results (list): The processed results of each batch.

#         Returns:
#             Dict[str, float]: The computed metrics. The keys are the names of
#                 the metrics, and the values are corresponding results. The key
#                 mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
#                 mRecall.
#         """
#         logger: MMLogger = MMLogger.get_current_instance()
#         if self.format_only:
#             logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
#             return OrderedDict()
#         # convert list of tuples to tuple of lists, e.g.
#         # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
#         # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
#         results = tuple(zip(*results))
#         assert len(results) == 4

#         total_area_intersect = sum(results[0])
#         total_area_union = sum(results[1])
#         total_area_pred_label = sum(results[2])
#         total_area_label = sum(results[3])
#         ret_metrics = self.total_area_to_metrics(
#             total_area_intersect, total_area_union, total_area_pred_label,
#             total_area_label, self.metrics, self.nan_to_num, self.beta)

#         class_names = self.dataset_meta['classes']

#         # summary table
#         ret_metrics_summary = OrderedDict({
#             ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
#             for ret_metric, ret_metric_value in ret_metrics.items()
#         })
#         metrics = dict()
#         for key, val in ret_metrics_summary.items():
#             if key == 'aAcc':
#                 metrics[key] = val
#             else:
#                 metrics['m' + key] = val

#         # each class table
#         ret_metrics.pop('aAcc', None)
#         ret_metrics_class = OrderedDict({
#             ret_metric: np.round(ret_metric_value * 100, 2)
#             for ret_metric, ret_metric_value in ret_metrics.items()
#         })
#         ret_metrics_class.update({'Class': class_names})
#         ret_metrics_class.move_to_end('Class', last=False)
#         class_table_data = PrettyTable()
#         for key, val in ret_metrics_class.items():
#             class_table_data.add_column(key, val)

#         print_log('per class results:', logger)
#         print_log('\n' + class_table_data.get_string(), logger=logger)

#         return metrics

#     @staticmethod
#     def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
#                             num_classes: int, ignore_index: int):
#         """Calculate Intersection and Union.

#         Args:
#             pred_label (torch.tensor): Prediction segmentation map
#                 or predict result filename. The shape is (H, W).
#             label (torch.tensor): Ground truth segmentation map
#                 or label filename. The shape is (H, W).
#             num_classes (int): Number of categories.
#             ignore_index (int): Index that will be ignored in evaluation.

#         Returns:
#             torch.Tensor: The intersection of prediction and ground truth
#                 histogram on all classes.
#             torch.Tensor: The union of prediction and ground truth histogram on
#                 all classes.
#             torch.Tensor: The prediction histogram on all classes.
#             torch.Tensor: The ground truth histogram on all classes.
#         """

#         mask = (label != ignore_index)
#         pred_label = pred_label[mask]
#         label = label[mask]

#         intersect = pred_label[pred_label == label]
#         area_intersect = torch.histc(
#             intersect.float(), bins=(num_classes), min=0,
#             max=num_classes - 1).cpu()
#         area_pred_label = torch.histc(
#             pred_label.float(), bins=(num_classes), min=0,
#             max=num_classes - 1).cpu()
#         area_label = torch.histc(
#             label.float(), bins=(num_classes), min=0,
#             max=num_classes - 1).cpu()
#         area_union = area_pred_label + area_label - area_intersect
#         return area_intersect, area_union, area_pred_label, area_label

#     @staticmethod
#     def total_area_to_metrics(total_area_intersect: np.ndarray,
#                               total_area_union: np.ndarray,
#                               total_area_pred_label: np.ndarray,
#                               total_area_label: np.ndarray,
#                               metrics: List[str] = ['mIoU'],
#                               nan_to_num: Optional[int] = None,
#                               beta: int = 1):
#         """Calculate evaluation metrics
#         Args:
#             total_area_intersect (np.ndarray): The intersection of prediction
#                 and ground truth histogram on all classes.
#             total_area_union (np.ndarray): The union of prediction and ground
#                 truth histogram on all classes.
#             total_area_pred_label (np.ndarray): The prediction histogram on
#                 all classes.
#             total_area_label (np.ndarray): The ground truth histogram on
#                 all classes.
#             metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
#                 'mDice'.
#             nan_to_num (int, optional): If specified, NaN values will be
#                 replaced by the numbers defined by the user. Default: None.
#             beta (int): Determines the weight of recall in the combined score.
#                 Default: 1.
#         Returns:
#             Dict[str, np.ndarray]: per category evaluation metrics,
#                 shape (num_classes, ).
#         """

#         def f_score(precision, recall, beta=1):
#             """calculate the f-score value.

#             Args:
#                 precision (float | torch.Tensor): The precision value.
#                 recall (float | torch.Tensor): The recall value.
#                 beta (int): Determines the weight of recall in the combined
#                     score. Default: 1.

#             Returns:
#                 [torch.tensor]: The f-score value.
#             """
#             score = (1 + beta**2) * (precision * recall) / (
#                 (beta**2 * precision) + recall)
#             return score

#         if isinstance(metrics, str):
#             metrics = [metrics]
#         allowed_metrics = ['mIoU', 'mDice', 'mFscore']
#         if not set(metrics).issubset(set(allowed_metrics)):
#             raise KeyError(f'metrics {metrics} is not supported')

#         all_acc = total_area_intersect.sum() / total_area_label.sum()
#         ret_metrics = OrderedDict({'aAcc': all_acc})
#         for metric in metrics:
#             if metric == 'mIoU':
#                 iou = total_area_intersect / total_area_union
#                 acc = total_area_intersect / total_area_label
#                 ret_metrics['IoU'] = iou
#                 ret_metrics['Acc'] = acc
#             elif metric == 'mDice':
#                 dice = 2 * total_area_intersect / (
#                     total_area_pred_label + total_area_label)
#                 acc = total_area_intersect / total_area_label
#                 ret_metrics['Dice'] = dice
#                 ret_metrics['Acc'] = acc
#             elif metric == 'mFscore':
#                 precision = total_area_intersect / total_area_pred_label
#                 recall = total_area_intersect / total_area_label
#                 f_value = torch.tensor([
#                     f_score(x[0], x[1], beta) for x in zip(precision, recall)
#                 ])
#                 ret_metrics['Fscore'] = f_value
#                 ret_metrics['Precision'] = precision
#                 ret_metrics['Recall'] = recall

#         ret_metrics = {
#             metric: value.numpy()
#             for metric, value in ret_metrics.items()
#         }
#         if nan_to_num is not None:
#             ret_metrics = OrderedDict({
#                 metric: np.nan_to_num(metric_value, nan=nan_to_num)
#                 for metric, metric_value in ret_metrics.items()
#             })
#         return ret_metrics




import os.path as osp
import numpy as np
import torch
import cv2
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU', 'mBoundaryFscore'],  # 新增指标
                 boundary_width: int = 2,  # 新增边界宽度参数
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.format_only = format_only
        self.boundary_width = boundary_width  # 存储边界宽度
        
        # 类别相关属性
        self.class_names = kwargs.get('dataset_meta', {}).get('classes', [])
        self.n_classes = len(self.class_names)
        self.kernel = self._create_kernel()  # 预先生成膨胀核

        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def _create_kernel(self) -> np.ndarray:
        """创建膨胀核"""
        kernel_size = self.boundary_width * 2 + 1
        return np.ones((kernel_size, kernel_size), np.uint8)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """处理单批次数据，存储预测结果和边界指标所需数据"""
        self.class_names = self.dataset_meta.get('classes', [])
        self.n_classes = len(self.class_names)
        
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label.device)

            # 保存预测掩码
            if self.output_dir is not None:
                self._save_pred_mask(data_sample, pred_label)

            if not self.format_only:
                # 计算各分类的边界指标数据
                boundary_results = self.calculate_boundary_metrics(pred_label, label)
                
                # 计算IoU指标数据
                iou_results = self.calculate_iou_metrics(pred_label, label)
                
                # 存储边界指标所需的交集/并集
                self.results.append({
                    'boundary_intersect': boundary_results['intersect'],
                    'boundary_union': boundary_results['union'],
                    'iou_intersect': iou_results['intersect'],
                    'iou_pred_area': iou_results['pred_area'],
                    'iou_label_area': iou_results['label_area']
                })

    def calculate_boundary_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Dict[str, np.ndarray]:
        """计算每个类别的边界交集和并集"""
        intersect = np.zeros(self.n_classes, dtype=np.float64)
        union = np.zeros(self.n_classes, dtype=np.float64)

        for cls in range(self.n_classes):
            # 生成单类别掩码
            pred_cls = (pred == cls).cpu().numpy().astype(np.uint8)
            label_cls = (label == cls).cpu().numpy().astype(np.uint8)
            
            # 提取边界区域
            pred_boundary = self._get_boundary(pred_cls)
            label_boundary = self._get_boundary(label_cls)
            
            # 计算交集和并集
            intersect[cls] = np.sum(pred_boundary * label_boundary)
            union[cls] = np.sum(pred_boundary) + np.sum(label_boundary) - intersect[cls]

        return {'intersect': intersect, 'union': union}

    def calculate_iou_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Dict[str, np.ndarray]:
        """计算每个类别的IoU相关指标"""
        intersect = np.zeros(self.n_classes, dtype=np.float64)
        pred_area = np.zeros(self.n_classes, dtype=np.float64)
        label_area = np.zeros(self.n_classes, dtype=np.float64)

        # 计算每个类别的交集和面积
        for cls in range(self.n_classes):
            pred_cls = (pred == cls)
            label_cls = (label == cls)
            
            intersect[cls] = torch.logical_and(pred_cls, label_cls).sum().cpu().numpy()
            pred_area[cls] = pred_cls.sum().cpu().numpy()
            label_area[cls] = label_cls.sum().cpu().numpy()

        return {'intersect': intersect, 'pred_area': pred_area, 'label_area': label_area}

    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """通过膨胀操作获取边界"""
        dilated = cv2.dilate(mask, self.kernel, iterations=1)
        return dilated - mask

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """计算最终评估指标（包含边界指标和IoU指标）"""
        if not results:
            print_log("No evaluation results available.", 'current', level='WARNING')
            return {}

        # 初始化边界指标累计值
        total_boundary_intersect = np.zeros(self.n_classes, dtype=np.float64)
        total_boundary_union = np.zeros(self.n_classes, dtype=np.float64)
        
        # 初始化IoU指标累计值
        total_iou_intersect = np.zeros(self.n_classes, dtype=np.float64)
        total_iou_pred_area = np.zeros(self.n_classes, dtype=np.float64)
        total_iou_label_area = np.zeros(self.n_classes, dtype=np.float64)

        # 累加边界指标数据
        for result in results:
            total_boundary_intersect += result['boundary_intersect']
            total_boundary_union += result['boundary_union']
            total_iou_intersect += result['iou_intersect']
            total_iou_pred_area += result['iou_pred_area']
            total_iou_label_area += result['iou_label_area']

        # 计算边界相关指标
        boundary_metrics = self.calculate_boundary_fscore(
            total_boundary_intersect,
            total_boundary_union,
            to_percent=True
        )
        
        # 计算IoU相关指标
        iou_metrics = self.calculate_iou(
            total_iou_intersect,
            total_iou_pred_area,
            total_iou_label_area,
            to_percent=True
        )

        # 生成最终结果字典
        final_results = {
            'mBoundaryFscore': boundary_metrics['mean'],
            **{f'class_{cls}_BoundaryFscore': val for cls, val in enumerate(boundary_metrics['classwise'])},
            'mIoU': iou_metrics['miou'],
            **{f'class_{cls}_IoU': val for cls, val in enumerate(iou_metrics['iou'])}
        }

        # 打印指标
        self._print_boundary_metrics(boundary_metrics)
        self._print_iou_metrics(iou_metrics)
        return final_results

    def calculate_boundary_fscore(self, intersect: np.ndarray, union: np.ndarray, to_percent: bool = False) -> Dict[str, np.ndarray]:
        """计算边界F-score"""
        epsilon = 1e-6
        boundary_iou = np.divide(intersect, union + epsilon, out=np.zeros_like(intersect), where=union!=0)
        boundary_fscore = (2 * boundary_iou) / (boundary_iou + 1 + epsilon)
        
        if to_percent:
            boundary_fscore *= 100
        
        return {
            'classwise': boundary_fscore,
            'mean': np.nanmean(boundary_fscore)
        }

    def calculate_iou(self, intersect: np.ndarray, pred_area: np.ndarray, label_area: np.ndarray, to_percent: bool = False) -> Dict[str, np.ndarray]:
        """计算IoU指标"""
        epsilon = 1e-6
        union = pred_area + label_area - intersect
        iou = np.divide(intersect, union + epsilon, out=np.zeros_like(intersect), where=union!=0)
        accuracy = np.divide(intersect, pred_area + epsilon, out=np.zeros_like(intersect), where=pred_area!=0)
        
        if to_percent:
            iou *= 100
            accuracy *= 100
        
        return {
            'iou': iou,
            'accuracy': accuracy,
            'miou': np.nanmean(iou),
            'mAcc': np.nanmean(accuracy),
            'aAcc': np.divide(np.sum(intersect), np.sum(label_area) + epsilon) * 100 if to_percent else np.divide(np.sum(intersect), np.sum(label_area) + epsilon)
        }

    def _print_boundary_metrics(self, metrics: Dict[str, np.ndarray]) -> None:
        """打印边界指标表格"""
        table = PrettyTable()
        table.field_names = ['Class'] + ['BoundaryFscore (%)']
        
        for cls_idx in range(self.n_classes):
            class_name = self.class_names[cls_idx]
            table.add_row([class_name, f"{metrics['classwise'][cls_idx]:.2f}"])
        
        table.add_row(['Mean', f"{metrics['mean']:.2f}"])
        print_log("\nBoundary Metrics:", 'current')
        print_log(str(table), 'current')

    def _print_iou_metrics(self, metrics: Dict[str, np.ndarray]) -> None:
        """打印IoU指标表格"""
        table = PrettyTable()
        table.field_names = ['Class'] + ['IoU (%)']
        
        for cls_idx in range(self.n_classes):
            class_name = self.class_names[cls_idx]
            table.add_row([class_name, f"{metrics['iou'][cls_idx]:.2f}"])
        
        table.add_row(['Mean', f"{metrics['miou']:.2f}"])
        print_log("\nIoU Metrics:", 'current')
        print_log(str(table), 'current')

    # 保留原有文件保存和其他指标计算逻辑（可根据需要扩展）
    def _save_pred_mask(self, data_sample: dict, pred_label: torch.Tensor) -> None:
        basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
        png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
        
        output_mask = pred_label.cpu().numpy()
        if data_sample.get('reduce_zero_label', False):
            output_mask = output_mask + 1
        
        Image.fromarray(output_mask.astype(np.uint8)).save(png_filename)