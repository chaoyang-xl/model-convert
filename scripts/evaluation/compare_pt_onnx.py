from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
IMAGE = str(ROOT / "data/samples/test.jpg")
PT_MODEL = str(ROOT / "models/fixed_pt/yolov8s-worldv2_indoor.pt")
ONNX_MODEL = str(ROOT / "models/onnx/yolov8s-worldv2_indoor.onnx")
OUTPUT_DIR = str(ROOT / "results/compare_results")
CONF = 0.25

pt_model = YOLO(PT_MODEL)
onnx_model = YOLO(ONNX_MODEL)

print("PT类别：", pt_model.names)
print("ONNX类别：", onnx_model.names)

pt_results = pt_model.predict(
    source=IMAGE,
    imgsz=640,
    conf=CONF,
    iou=0.7,
    save=True,
    project=OUTPUT_DIR,
    name="pt",
)

onnx_results = onnx_model.predict(
    source=IMAGE,
    imgsz=640,
    conf=CONF,
    iou=0.7,
    save=True,
    project=OUTPUT_DIR,
    name="onnx",
)

print("\nPT结果：")
for box in pt_results[0].boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].tolist()
    print(pt_model.names[class_id], confidence, coordinates)

print("\nONNX结果：")
for box in onnx_results[0].boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].tolist()
    print(onnx_model.names[class_id], confidence, coordinates)
