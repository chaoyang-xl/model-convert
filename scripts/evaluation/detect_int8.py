from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = str(ROOT / "models/onnx/yolov8s-worldv2_indoor_split.onnx")
DATASET = str(ROOT / "data/calibration/dataset.txt")
IMAGE = str(ROOT / "data/samples/test.jpg")
OUTPUT_IMAGE = str(ROOT / "results/test_int8_result.jpg")

IMG_SIZE = 640
CONF = 0.25
IOU = 0.7

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


def letterbox(image, size=640):
    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8,
    )

    left = (size - new_w) // 2
    top = (size - new_h) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    return canvas, scale, left, top


def xywh_to_xyxy(boxes):
    result = boxes.copy()

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h

        union = (
            areas[i]
            + areas[order[1:]]
            - inter
        )

        iou = inter / np.maximum(union, 1e-6)

        indices = np.where(iou <= threshold)[0]
        order = order[indices + 1]

    return keep


def postprocess(
    boxes_output,
    scores_output,
    scale,
    left,
    top,
    original_width,
    original_height,
):
    # boxes_output:  [1, 4, 8400]
    # scores_output: [1,12, 8400]

    boxes = boxes_output[0].T
    scores = scores_output[0].T

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= CONF

    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    boxes = xywh_to_xyxy(boxes)

    results = []

    # 分类别做 NMS
    for class_id in range(len(CLASSES)):

        class_mask = class_ids == class_id

        class_boxes = boxes[class_mask]
        class_scores = confidences[class_mask]

        if len(class_boxes) == 0:
            continue

        keep = nms(
            class_boxes,
            class_scores,
            IOU,
        )

        for index in keep:
            box = class_boxes[index].copy()

            # letterbox -> 原图
            box[0] = (box[0] - left) / scale
            box[1] = (box[1] - top) / scale
            box[2] = (box[2] - left) / scale
            box[3] = (box[3] - top) / scale

            box[0] = np.clip(box[0], 0, original_width - 1)
            box[2] = np.clip(box[2], 0, original_width - 1)
            box[1] = np.clip(box[1], 0, original_height - 1)
            box[3] = np.clip(box[3], 0, original_height - 1)

            results.append({
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "confidence": float(class_scores[index]),
                "box": box.tolist(),
            })

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return results


def main():

    # --------------------------------------------------
    # 图片
    # --------------------------------------------------

    image_bgr = cv2.imread(IMAGE)

    if image_bgr is None:
        raise FileNotFoundError(IMAGE)

    original_h, original_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    input_image, scale, left, top = letterbox(
        image_rgb,
        IMG_SIZE,
    )

    input_image = np.expand_dims(
        input_image,
        axis=0,
    )

    print("原图:", original_w, "x", original_h)
    print("RKNN输入:", input_image.shape)

    # --------------------------------------------------
    # RKNN INT8
    # --------------------------------------------------

    rknn = RKNN(verbose=False)

    ret = rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )

    if ret != 0:
        raise RuntimeError(
            f"config failed: {ret}"
        )

    ret = rknn.load_onnx(
        model=ONNX_MODEL
    )

    if ret != 0:
        raise RuntimeError(
            f"load_onnx failed: {ret}"
        )

    print("构建 INT8 模型...")

    ret = rknn.build(
        do_quantization=True,
        dataset=DATASET,
    )

    if ret != 0:
        raise RuntimeError(
            f"build failed: {ret}"
        )

    ret = rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(
            f"init_runtime failed: {ret}"
        )

    print("开始推理...")

    outputs = rknn.inference(
        inputs=[input_image],
        data_format=["nhwc"],
    )

    rknn.release()

    print("输出数量:", len(outputs))

    for i, output in enumerate(outputs):
        print(
            f"output[{i}]",
            output.shape,
            "min=",
            float(output.min()),
            "max=",
            float(output.max()),
        )

    boxes_output = outputs[0]
    scores_output = outputs[1]

    # --------------------------------------------------
    # 后处理
    # --------------------------------------------------

    results = postprocess(
        boxes_output,
        scores_output,
        scale,
        left,
        top,
        original_w,
        original_h,
    )

    print()
    print("========== INT8检测结果 ==========")

    if not results:
        print("没有检测到目标")

    for result in results:
        print(
            result["class_name"],
            f'{result["confidence"]:.6f}',
            [
                round(v, 2)
                for v in result["box"]
            ],
        )

    # --------------------------------------------------
    # 画框
    # --------------------------------------------------

    draw = image_bgr.copy()

    for result in results:

        x1, y1, x2, y2 = [
            int(round(v))
            for v in result["box"]
        ]

        label = (
            f'{result["class_name"]} '
            f'{result["confidence"]:.2f}'
        )

        cv2.rectangle(
            draw,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            draw,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        OUTPUT_IMAGE,
        draw,
    )

    print()
    print("检测结果图片已保存：")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
