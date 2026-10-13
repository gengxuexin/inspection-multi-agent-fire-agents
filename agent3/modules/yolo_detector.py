# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent3")

3.2.1 基于 YOLO 的物品识别
封装 YOLO 推理，输出结构化检测结果
"""
import os
from pathlib import Path
from ultralytics import YOLO
from config import YOLO_WEIGHTS
import logging


class YoloDetector:
    def __init__(self, weights=None):
        weights = weights or YOLO_WEIGHTS
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO 权重不存在：{weights}")
        self.model = YOLO(weights)
        self.class_names = self.model.names   # {0: 'fire_extinguisher', ...}
        logger.info(f"✅ YOLO 已加载：{weights}")
        logger.info(f"   类别：{list(self.class_names.values())}")

    def detect(self, source, conf=0.25, imgsz=320):
        """
        对图片或视频进行检测。

        Args:
            source: 图片/视频路径
            conf: 置信度阈值
            imgsz: 输入尺寸（与训练一致）

        Returns:
            dict: {
                "source": 路径,
                "counts": {类别: 数量},
                "detections": [{cls, conf, bbox, ...}],
                "frames_analyzed": N
            }
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
            stream=False,
        )

        # 汇总所有帧/图片的检测
        counts = {}
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                counts[cls_name] = counts.get(cls_name, 0) + 1
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf_val, 3),
                    "bbox": [round(v, 1) for v in xyxy],
                })

        return {
            "source": str(source),
            "counts": counts,
            "total_objects": len(detections),
            "detections": detections[:100],   # 限制条数避免报告过长
            "frames_analyzed": len(results),
        }


# ========== 测试 ==========
if __name__ == "__main__":
    det = YoloDetector()

    # ★ 改成有消防设施的图片
    test_img = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\1\frames\1\frame_0009.jpg"

    # 如果上面那张也没有消防设施，就从训练集里抽一张：
    # test_img = r"D:\AI\ollama_deepseek\yolov8\fire_mini\images\train\随便一张.jpg"

    r = det.detect(test_img)
    logger.info("\n检测结果：")
    logger.info(f"  总目标数：{r['total_objects']}")
    logger.info(f"  分类统计：{r['counts']}")

    if r['total_objects'] > 0:
        logger.info("\n详细检测：")
        for d in r['detections'][:10]:
            logger.info(f"  {d['class']} (conf={d['confidence']})")
    else:
        logger.info("\n⚠️ 未检测到任何目标，请确认图片中有消防设施")