from pathlib import Path

from ultralytics import YOLO


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

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = ROOT / "models/source/yolov8s-worldv2.pt"
FIXED_MODEL = ROOT / "models/fixed_pt/yolov8s-worldv2_indoor.pt"
ONNX_MODEL = ROOT / "models/onnx/yolov8s-worldv2_indoor.onnx"


def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"找不到模型：{SOURCE_MODEL.resolve()}")

    print("加载原始模型：", SOURCE_MODEL)
    model = YOLO(str(SOURCE_MODEL))

    print("设置固定类别：")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")

    # 使用CLIP生成这12个类别的文本向量，并写入模型
    model.set_classes(CLASSES)

    print("当前模型类别：", model.names)

    # 保存包含固定类别向量的PT模型
    model.save(str(FIXED_MODEL))
    print("固定类别PT模型已保存：", FIXED_MODEL.resolve())

    # 重新加载，验证类别向量确实保存在PT中
    fixed_model = YOLO(str(FIXED_MODEL))

    print("重新加载后的类别：", fixed_model.names)

    if list(fixed_model.names.values()) != CLASSES:
        raise RuntimeError(
            "重新加载后的类别与预期不一致："
            f"\n实际：{fixed_model.names}"
            f"\n预期：{CLASSES}"
        )

    # 导出静态ONNX
    onnx_path = fixed_model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=12,
        half=False,
    )

    exported_onnx = Path(onnx_path).resolve()
    ONNX_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if exported_onnx != ONNX_MODEL:
        exported_onnx.replace(ONNX_MODEL)
    print("ONNX导出完成：", ONNX_MODEL)


if __name__ == "__main__":
    main()
