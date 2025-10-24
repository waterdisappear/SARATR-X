import os
import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
import matplotlib.pyplot as plt


class LargeImageDetector:
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        """
        初始化大图检测器

        Args:
            config_file: 模型配置文件路径
            checkpoint_file: 模型权重文件路径
            device: 推理设备
        """
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.device = device
        self.class_names = self.model.CLASSES

    def sliding_window_inference(self,
                                 image,
                                 window_size=(1024, 1024),
                                 stride=(512, 512),
                                 confidence_threshold=0.3,
                                 iou_threshold=0.5):
        """
        滑动窗口推理

        Args:
            image: 输入图像 (H, W, C)
            window_size: 窗口大小 (width, height)
            stride: 滑动步长 (stride_x, stride_y)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值

        Returns:
            results: 检测结果列表
        """
        orig_h, orig_w = image.shape[:2]
        window_w, window_h = window_size
        stride_x, stride_y = stride

        all_results = []

        # 遍历所有窗口
        for y in range(0, orig_h, stride_y):
            for x in range(0, orig_w, stride_x):
                # 计算当前窗口位置
                x1 = x
                y1 = y
                x2 = min(x + window_w, orig_w)
                y2 = min(y + window_h, orig_h)

                # 如果窗口太小则跳过
                if (x2 - x1) < window_w // 4 or (y2 - y1) < window_h // 4:
                    continue

                # 提取窗口区域
                window = image[y1:y2, x1:x2]

                # 推理
                result = inference_detector(self.model, window)

                # 处理检测结果
                for class_id, bboxes in enumerate(result):
                    for bbox in bboxes:
                        if bbox[4] >= confidence_threshold:  # 置信度过滤
                            # 转换坐标到原图
                            bbox_orig = [
                                bbox[0] + x1,  # x1
                                bbox[1] + y1,  # y1
                                bbox[2] + x1,  # x2
                                bbox[3] + y1,  # y2
                                bbox[4],  # score
                                class_id  # class
                            ]
                            all_results.append(bbox_orig)

        # 应用NMS
        if len(all_results) > 0:
            all_results = self.nms(all_results, iou_threshold)

        return all_results

    def nms(self, detections, iou_threshold):
        """
        非极大值抑制

        Args:
            detections: 检测结果列表
            iou_threshold: IOU阈值

        Returns:
            filtered_detections: 过滤后的检测结果
        """
        if len(detections) == 0:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        filtered_detections = []

        while detections:
            # 取置信度最高的检测框
            best = detections.pop(0)
            filtered_detections.append(best)

            # 计算与剩余框的IOU
            remaining = []
            for detection in detections:
                iou = self.calculate_iou(best, detection)
                if iou < iou_threshold:
                    remaining.append(detection)
            detections = remaining

        return filtered_detections

    def calculate_iou(self, box1, box2):
        """
        计算两个框的IOU

        Args:
            box1: 框1 [x1, y1, x2, y2, score, class]
            box2: 框2 [x1, y1, x2, y2, score, class]

        Returns:
            iou: IOU值
        """
        # 提取坐标
        x1_1, y1_1, x1_2, y1_2 = box1[:4]
        x2_1, y2_1, x2_2, y2_2 = box2[:4]

        # 计算交集区域
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 计算并集面积
        box1_area = (x1_2 - x1_1) * (y1_2 - y1_1)
        box2_area = (x2_2 - x2_1) * (y2_2 - y2_1)
        union_area = box1_area + box2_area - intersection_area

        # 计算IOU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def visualize_results(self, image, results, output_path=None):
        """
        可视化检测结果

        Args:
            image: 原图像
            results: 检测结果
            output_path: 输出路径
        """
        # 创建副本
        vis_image = image.copy()

        # 定义颜色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for result in results:
            x1, y1, x2, y2, score, class_id = result
            color = colors[class_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(vis_image,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          color, 2)

            # 添加标签
            label = f'{self.class_names[class_id]}: {score:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(vis_image,
                          (int(x1), int(y1) - label_size[1] - 10),
                          (int(x1) + label_size[0], int(y1)),
                          color, -1)

            cv2.putText(vis_image, label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, vis_image)

        return vis_image


def main():
    # 配置参数
    config_file = 'D:\\2023_SARatrX_1\MIM\detection_hivit\work_dirs\\SOC50\hivit_base_SOC50.py'
    checkpoint_file = 'D:\\2023_SARatrX_1\MIM\detection_hivit\work_dirs\SOC50\\best_bbox_mAP_epoch_34.pth'
    image_path = 'D:\\Code\\utils\\img_test\\imgs\\XXX.tif'
    output_dir = 'outputs'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化检测器
    detector = LargeImageDetector(config_file, checkpoint_file)

    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 滑动窗口推理
    print("开始大图推理...")
    results = detector.sliding_window_inference(
        image_rgb,
        window_size=(200, 200),
        stride=(100, 100),
        confidence_threshold=0.6,
        iou_threshold=0.5
    )

    print(f"检测到 {len(results)} 个目标")

    # 可视化结果
    output_path = os.path.join(output_dir, 'detection_result_3.jpg')
    vis_image = detector.visualize_results(image, results, output_path)

    # 保存检测结果到文件
    results_file = os.path.join(output_dir, 'detection_results_3.txt')
    with open(results_file, 'w') as f:
        for result in results:
            x1, y1, x2, y2, score, class_id = result
            class_name = detector.class_names[class_id]
            f.write(f'{class_name} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n')

    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
