from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = str(ROOT / "models/onnx/yolov8s-worldv2_indoor_split.onnx")
DATASET = str(ROOT / "data/calibration/dataset.txt")
IMAGE = str(ROOT / "data/samples/test.jpg")

CONF = 0.25
IOU = 0.7
IMG_SIZE = 640

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
    output,
    scale,
    left,
    top,
    original_width,
    original_height,
):
    # output:
    # [1, 16, 8400]
    #
    # 转成：
    # [8400, 16]

    pred = output[0].T

    boxes = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    )

    confidences = np.max(
        class_scores,
        axis=1,
    )

    mask = confidences >= CONF

    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    boxes = xywh_to_xyxy(boxes)

    final_boxes = []
    final_scores = []
    final_classes = []

    # YOLO NMS 应按类别分别执行
    for class_id in range(len(CLASSES)):

        class_mask = (
            class_ids == class_id
        )

        class_boxes = boxes[class_mask]
        class_confidences = confidences[class_mask]

        if len(class_boxes) == 0:
            continue

        keep = nms(
            class_boxes,
            class_confidences,
            IOU,
        )

        for index in keep:

            box = class_boxes[index].copy()

            # 从640x640 letterbox恢复到原图
            box[0] = (box[0] - left) / scale
            box[1] = (box[1] - top) / scale
            box[2] = (box[2] - left) / scale
            box[3] = (box[3] - top) / scale

            box[0] = np.clip(
                box[0],
                0,
                original_width,
            )

            box[2] = np.clip(
                box[2],
                0,
                original_width,
            )

            box[1] = np.clip(
                box[1],
                0,
                original_height,
            )

            box[3] = np.clip(
                box[3],
                0,
                original_height,
            )

            final_boxes.append(box)
            final_scores.append(
                float(class_confidences[index])
            )
            final_classes.append(class_id)

    if len(final_boxes) == 0:
        return []

    # 按置信度从高到低
    order = np.argsort(
        final_scores
    )[::-1]

    results = []

    for index in order:

        results.append(
            {
                "class_id": int(
                    final_classes[index]
                ),
                "class_name": CLASSES[
                    final_classes[index]
                ],
                "confidence": float(
                    final_scores[index]
                ),
                "box": final_boxes[
                    index
                ].tolist(),
            }
        )

    return results


def print_results(name, results):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    if not results:
        print("没有检测结果")
        return

    for result in results:

        print(
            result["class_name"],
            f'{result["confidence"]:.6f}',
            [
                round(x, 2)
                for x in result["box"]
            ],
        )


def main():
    image_bgr = cv2.imread(IMAGE)

    if image_bgr is None:
        raise FileNotFoundError(IMAGE)

    original_height, original_width = (
        image_bgr.shape[:2]
    )

    # OpenCV:
    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    input_image, scale, left, top = letterbox(
        image_rgb,
        IMG_SIZE,
    )

    print("原图尺寸：")
    print(
        original_width,
        "x",
        original_height,
    )

    print()

    print("letterbox:")
    print("scale =", scale)
    print("left =", left)
    print("top =", top)

    # ==================================================
    # ONNX Runtime
    # ==================================================

    print()
    print("========== ONNX ==========")

    session = ort.InferenceSession(
        ONNX_MODEL,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_info = session.get_inputs()[0]

    print(
        "ONNX input:",
        input_info.name,
        input_info.shape,
    )

    onnx_input = (
        input_image
        .astype(np.float32)
        / 255.0
    )

    # NHWC -> NCHW
    onnx_input = np.transpose(
        onnx_input,
        (2, 0, 1),
    )

    # CHW -> NCHW
    onnx_input = np.expand_dims(
        onnx_input,
        axis=0,
    )

    onnx_outputs = session.run(
        None,
        {
            input_info.name: onnx_input
        },
    )

    onnx_boxes = onnx_outputs[0]
    onnx_scores = onnx_outputs[1]

    print("ONNX boxes:", onnx_boxes.shape)
    print("ONNX scores:", onnx_scores.shape)

    onnx_output = np.concatenate(
        [
            onnx_boxes,
            onnx_scores,
        ],
        axis=1,
    )

    print(
        "ONNX output:",
        onnx_output.shape,
    )

    # ==================================================
    # RKNN Simulator
    # ==================================================

    print()
    print("========== RKNN ==========")

    rknn = RKNN(verbose=False)

    ret = rknn.config(
        target_platform="rk3588",
        mean_values=[
            [0, 0, 0]
        ],
        std_values=[
            [255, 255, 255]
        ],
    )

    if ret != 0:
        raise RuntimeError(
            f"rknn.config failed: {ret}"
        )

    ret = rknn.load_onnx(
        model=ONNX_MODEL
    )

    if ret != 0:
        raise RuntimeError(
            f"load_onnx failed: {ret}"
        )

    ret = rknn.build(
        do_quantization=True,
        dataset=DATASET,
    )

    if ret != 0:
        raise RuntimeError(
            f"rknn.build failed: {ret}"
        )

    ret = rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(
            f"init_runtime failed: {ret}"
        )

    # 显式增加batch维
    rknn_input = np.expand_dims(
        input_image,
        axis=0,
    )

    print(
        "RKNN input:",
        rknn_input.shape,
        rknn_input.dtype,
    )

    rknn_outputs = rknn.inference(
        inputs=[rknn_input],
        data_format=["nhwc"],
    )

    print("RKNN输出数量:", len(rknn_outputs))

    for i, x in enumerate(rknn_outputs):
        print(
            f"RKNN output[{i}]:",
            x.shape,
            x.dtype,
            "min=",
            float(x.min()),
            "max=",
            float(x.max()),
        )

    rknn_boxes = rknn_outputs[0]
    rknn_scores = rknn_outputs[1]

    rknn_output = np.concatenate(
        [
            rknn_boxes,
            rknn_scores,
        ],
        axis=1,
    )

    rknn.release()

    print(
        "RKNN output:",
        rknn_output.shape,
    )

    # ==================================================
    # RAW OUTPUT DIFFERENCE
    # ==================================================

    print()
    print(
        "========== RAW DIFFERENCE =========="
    )

    diff = np.abs(
        onnx_output.astype(np.float32)
        -
        rknn_output.astype(np.float32)
    )

    print(
        "全部输出 MAE:",
        float(np.mean(diff)),
    )

    print(
        "全部输出 Max Error:",
        float(np.max(diff)),
    )

    # bbox部分
    bbox_diff = diff[:, :4, :]

    print()
    print(
        "BBox MAE:",
        float(np.mean(bbox_diff)),
    )

    print(
        "BBox Max Error:",
        float(np.max(bbox_diff)),
    )

    # 类别分数部分
    score_diff = diff[:, 4:, :]

    print()
    print(
        "Class Score MAE:",
        float(np.mean(score_diff)),
    )

    print(
        "Class Score Max Error:",
        float(np.max(score_diff)),
    )

    # ==================================================
    # POSTPROCESS
    # ==================================================

    onnx_results = postprocess(
        onnx_output,
        scale,
        left,
        top,
        original_width,
        original_height,
    )

    rknn_results = postprocess(
        rknn_output,
        scale,
        left,
        top,
        original_width,
        original_height,
    )

    print_results(
        "ONNX检测结果",
        onnx_results,
    )

    print_results(
        "RKNN FP检测结果",
        rknn_results,
    )
    print("\n========== CLASS SCORE DEBUG ==========")

    onnx_scores = onnx_output[0, 4:, :]
    rknn_scores = rknn_output[0, 4:, :]

    print("ONNX class score max:", onnx_scores.max())
    print("RKNN class score max:", rknn_scores.max())

    print("\n每个类别最大分数：")

    for i, name in enumerate(CLASSES):
        print(
            f"{i:2d} {name:12s} "
            f"ONNX={onnx_scores[i].max():.6f} "
            f"RKNN={rknn_scores[i].max():.6f}"
        )


if __name__ == "__main__":
    main()
