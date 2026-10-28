from pathlib import Path
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = ROOT / "models/onnx/yolov8s-worldv2_indoor.onnx"
RKNN_MODEL = ROOT / "models/rknn/yolov8s-worldv2_indoor_fp.rknn"


def check(ret, name):
    if ret != 0:
        raise RuntimeError(f"{name} failed, ret={ret}")


def main():
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(ONNX_MODEL.resolve())

    rknn = RKNN(verbose=True)

    print("========== CONFIG ==========")

    ret = rknn.config(
        target_platform="rk3588",

        # 原始YOLO输入最终需要 /255
        # 这样RKNN端可以直接输入 uint8 RGB
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )
    check(ret, "config")

    print("========== LOAD ONNX ==========")

    ret = rknn.load_onnx(
        model=str(ONNX_MODEL),
    )
    check(ret, "load_onnx")

    print("========== BUILD FP RKNN ==========")

    ret = rknn.build(
        do_quantization=False,
    )
    check(ret, "build")

    print("========== EXPORT RKNN ==========")

    ret = rknn.export_rknn(
        str(RKNN_MODEL)
    )
    check(ret, "export_rknn")

    rknn.release()

    print()
    print("转换完成：")
    print(RKNN_MODEL.resolve())


if __name__ == "__main__":
    main()
