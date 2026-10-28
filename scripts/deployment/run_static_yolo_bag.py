#!/usr/bin/env python3
"""Run a fixed-class Ultralytics YOLO-World RKNN model on a ROS image topic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from semantic_map_online.rknn_runtime import RknnSession


CLASSES = [
    "chair",
    "sofa",
    "table",
    "cabinet",
    "shelf",
    "monitor",
    "computer",
    "television",
    "plant",
    "trash can",
    "door",
    "desk",
]


def letterbox(image: np.ndarray, size: int = 640):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, left, top


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = order[np.flatnonzero(iou <= threshold) + 1]
    return np.asarray(keep, dtype=np.int64)


def postprocess(outputs, confidence, iou, scale, left, top, width, height):
    if len(outputs) == 1:
        combined = np.asarray(outputs[0], dtype=np.float32)
        expected_channels = 4 + len(CLASSES)
        if combined.ndim != 3 or combined.shape[1] != expected_channels:
            raise RuntimeError(
                f"expected FP16 output [1,{expected_channels},N], got {combined.shape}"
            )
        boxes_output = combined[:, :4, :]
        scores_output = combined[:, 4:, :]
    elif len(outputs) == 2:
        boxes_output, scores_output = (
            np.asarray(value, dtype=np.float32) for value in outputs
        )
        if boxes_output.shape[1] != 4:
            boxes_output, scores_output = scores_output, boxes_output
    else:
        raise RuntimeError(f"expected one or two RKNN outputs, got {len(outputs)}")
    boxes = boxes_output[0].T
    scores = scores_output[0].T
    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    valid = confidences >= confidence
    boxes = xywh_to_xyxy(boxes[valid])
    class_ids = class_ids[valid]
    confidences = confidences[valid]
    detections = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        for local_index in nms(boxes[indices], confidences[indices], iou):
            index = indices[local_index]
            box = boxes[index].copy()
            box[[0, 2]] = (box[[0, 2]] - left) / scale
            box[[1, 3]] = (box[[1, 3]] - top) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            detections.append((float(confidences[index]), int(class_id), box))
    return sorted(detections, reverse=True, key=lambda item: item[0])


class Detector(Node):
    def __init__(self, args):
        super().__init__("static_int8_yolo_bag_test")
        self.args = args
        self.output = args.output.expanduser()
        self.output.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.model = RknnSession(args.model, backend="lite", core_mask=args.core)
        self.received = 0
        self.processed = 0
        self.total_seconds = 0.0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Image, args.topic, self.on_image, qos)
        self.get_logger().info(
            f"model={args.model} topic={args.topic} output={self.output} "
            f"conf={args.confidence} iou={args.iou} every={args.every}"
        )

    def on_image(self, message: Image):
        self.received += 1
        if (self.received - 1) % self.args.every:
            return
        bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        model_input, scale, left, top = letterbox(rgb)
        started = time.perf_counter()
        outputs = self.model.inference([model_input[None]], data_format="nhwc")
        elapsed = time.perf_counter() - started
        detections = postprocess(
            outputs,
            self.args.confidence,
            self.args.iou,
            scale,
            left,
            top,
            bgr.shape[1],
            bgr.shape[0],
        )
        draw = bgr.copy()
        for score, class_id, box in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                draw,
                f"{CLASSES[class_id]} {score:.2f}",
                (x1, max(22, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        stamp = message.header.stamp
        filename = f"yolo_{self.processed:06d}_{stamp.sec * 1_000_000_000 + stamp.nanosec}.jpg"
        cv2.imwrite(str(self.output / filename), draw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.processed += 1
        self.total_seconds += elapsed
        if self.processed == 1 or self.processed % 25 == 0:
            average_ms = self.total_seconds * 1000 / self.processed
            self.get_logger().info(
                f"saved={self.processed} detections={len(detections)} "
                f"inference={elapsed * 1000:.1f}ms average={average_ms:.1f}ms"
            )
        if self.args.max_images and self.processed >= self.args.max_images:
            self.get_logger().info(f"reached max-images={self.args.max_images}")
            rclpy.shutdown()

    def destroy_node(self):
        self.model.close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--core", default="0_1_2")
    args = parser.parse_args()
    rclpy.init()
    node = Detector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
