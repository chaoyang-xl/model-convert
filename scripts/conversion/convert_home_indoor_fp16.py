#!/usr/bin/env python3
"""Build a fixed-class, non-quantized RKNN model for RK3588."""

from __future__ import annotations

from pathlib import Path

import onnx
from rknn.api import RKNN
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
CLASSES_FILE = Path(__file__).with_name("home_indoor_classes.txt")
SOURCE_MODEL = ROOT / "models/source/yolov8s-worldv2.pt"
FIXED_MODEL = ROOT / "models/fixed_pt/yolov8s-worldv2_home_indoor.pt"
ONNX_MODEL = ROOT / "models/onnx/yolov8s-worldv2_home_indoor.onnx"
RKNN_MODEL = ROOT / "models/rknn/yolov8s-worldv2_home_indoor_fp16.rknn"


def check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}")


def load_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    classes = [name for name in classes if name and not name.startswith("#")]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    duplicates = sorted({name for name in classes if classes.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate classes in {path}: {duplicates}")
    return classes


def export_onnx(classes: list[str]) -> None:
    print(f"Loading source model: {SOURCE_MODEL}")
    model = YOLO(str(SOURCE_MODEL))
    model.set_classes(classes)
    print(f"Fixed classes ({len(classes)}): {classes}")

    FIXED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FIXED_MODEL))

    fixed_model = YOLO(str(FIXED_MODEL))
    saved_classes = list(fixed_model.names.values())
    if saved_classes != classes:
        raise RuntimeError(
            "Classes changed after saving the fixed PT model:\n"
            f"expected={classes}\nactual={saved_classes}"
        )

    exported = Path(
        fixed_model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=12,
            half=False,
        )
    ).resolve()
    ONNX_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if exported != ONNX_MODEL:
        exported.replace(ONNX_MODEL)

    onnx_model = onnx.load(str(ONNX_MODEL), load_external_data=False)
    onnx.checker.check_model(onnx_model)
    output_dims = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    expected_channels = 4 + len(classes)
    if output_dims != [1, expected_channels, 8400]:
        raise RuntimeError(
            f"Unexpected ONNX output shape: {output_dims}; "
            f"expected [1, {expected_channels}, 8400]"
        )
    print(f"Validated ONNX output: {output_dims}")


def export_rknn() -> None:
    rknn = RKNN(verbose=True)
    try:
        check(
            rknn.config(
                target_platform="rk3588",
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
            ),
            "rknn.config",
        )
        check(rknn.load_onnx(model=str(ONNX_MODEL)), "rknn.load_onnx")
        check(rknn.build(do_quantization=False), "rknn.build")
        RKNN_MODEL.parent.mkdir(parents=True, exist_ok=True)
        check(rknn.export_rknn(str(RKNN_MODEL)), "rknn.export_rknn")
    finally:
        rknn.release()


def main() -> None:
    for path in (CLASSES_FILE, SOURCE_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    classes = load_classes(CLASSES_FILE)
    export_onnx(classes)
    export_rknn()
    print(f"FP16 RKNN ready: {RKNN_MODEL}")


if __name__ == "__main__":
    main()
