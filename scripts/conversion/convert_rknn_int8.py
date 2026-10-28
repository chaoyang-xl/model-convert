from pathlib import Path
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = ROOT / "models/onnx/yolov8s-worldv2_indoor_split.onnx"
DATASET = ROOT / "data/calibration/dataset.txt"
RKNN_MODEL = ROOT / "models/rknn/yolov8s-worldv2_indoor_split_int8.rknn"


def check(ret, name):
    if ret != 0:
        raise RuntimeError(
            f"{name} failed, ret={ret}"
        )


def main():

    if not ONNX_MODEL.exists():
        raise FileNotFoundError(
            ONNX_MODEL.resolve()
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            DATASET.resolve()
        )

    print("ONNX:")
    print(ONNX_MODEL.resolve())

    print()
    print("Calibration dataset:")
    print(DATASET.resolve())

    rknn = RKNN(
        verbose=True
    )

    print()
    print("========== CONFIG ==========")

    ret = rknn.config(
        target_platform="rk3588",

        mean_values=[
            [0, 0, 0]
        ],

        std_values=[
            [255, 255, 255]
        ],
    )

    check(
        ret,
        "config"
    )

    print()
    print("========== LOAD ONNX ==========")

    ret = rknn.load_onnx(
        model=str(ONNX_MODEL)
    )

    check(
        ret,
        "load_onnx"
    )

    print()
    print("========== BUILD INT8 ==========")

    ret = rknn.build(
        do_quantization=True,
        dataset=str(DATASET),
    )

    check(
        ret,
        "build"
    )

    print()
    print("========== EXPORT ==========")

    ret = rknn.export_rknn(
        str(RKNN_MODEL)
    )

    check(
        ret,
        "export_rknn"
    )

    rknn.release()

    print()
    print("INT8 RKNN生成完成：")
    print(
        RKNN_MODEL.resolve()
    )


if __name__ == "__main__":
    main()
