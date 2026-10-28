from pathlib import Path

import cv2
import numpy as np

from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / "models/onnx/yolov8s-worldv2_indoor.onnx")
IMAGE = str(ROOT / "data/samples/test.jpg")


def check(ret, name):
    if ret != 0:
        raise RuntimeError(f"{name} failed, ret={ret}")


def letterbox(image, size=640):
    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8
    )

    left = (size - nw) // 2
    top = (size - nh) // 2

    canvas[
        top:top + nh,
        left:left + nw
    ] = resized

    return canvas


rknn = RKNN(verbose=True)

print("========== CONFIG ==========")

ret = rknn.config(
    target_platform="rk3588",
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
)
check(ret, "config")


print("========== LOAD ONNX ==========")

ret = rknn.load_onnx(
    model=MODEL
)
check(ret, "load_onnx")


print("========== BUILD ==========")

ret = rknn.build(
    do_quantization=False
)
check(ret, "build")


print("========== INIT SIMULATOR ==========")

# 注意：
# PC模拟器这里不要加 target='rk3588'
ret = rknn.init_runtime()
check(ret, "init_runtime")


print("========== LOAD IMAGE ==========")

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(IMAGE)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = letterbox(image, 640)

print("input shape:", image.shape)
print("input dtype:", image.dtype)
print("input min/max:", image.min(), image.max())


print("========== INFERENCE ==========")

outputs = rknn.inference(
    inputs=[image],
    data_format=["nhwc"],
)


print()
print("========== OUTPUTS ==========")

print("RKNN输出数量:", len(outputs))

for i, output in enumerate(outputs):

    print()
    print(f"output[{i}]")

    print("shape:", output.shape)
    print("dtype:", output.dtype)
    print("min:", float(output.min()))
    print("max:", float(output.max()))

    # 看一点实际数值
    flat = output.reshape(-1)

    print(
        "first values:",
        flat[:20]
    )


rknn.release()
