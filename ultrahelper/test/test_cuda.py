from ultralytics import YOLO
# Load a COCO-pretrained YOLOv8n model

model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="ultrahelper/cfg/coco-pose8.yaml", epochs=100, imgsz=640, device = 'cuda')

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("/Users/mahsaraeisinezhad/Documents/interviews/mentium/ultrahelper/IMG_4425.jpg")