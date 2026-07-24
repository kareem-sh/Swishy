from ultralytics import YOLO

model = YOLO("models/basketball_yolov8n.pt")

output = model.export(
    format="engine",
    imgsz=640,
    dynamic=False,
    device=0,
)

print(f"TensorRT engine saved to: {output}")
