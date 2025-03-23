import os
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Load pre-trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'object_detection_model', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'saved_model')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

model = tf.saved_model.load(MODEL_PATH)

# Label map for COCO dataset (90 classes)
LABELS = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
}

# Object detection function
def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    detection_scores = detections["detection_scores"].numpy()[0]
    detection_classes = detections["detection_classes"].numpy()[0].astype(int)
    detection_boxes = detections["detection_boxes"].numpy()[0]

    return detection_scores, detection_classes, detection_boxes

# Draw bounding boxes on image
def draw_boxes(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    scores, classes, boxes = detect_objects(image_path)

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            label = LABELS.get(class_id, "Unknown")
            box = boxes[i]

            y_min, x_min, y_max, x_max = box
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {scores[i]:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join('static', 'uploads', 'output.jpg')
    cv2.imwrite(output_path, img)
    return output_path

# Django View
def index(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location="static/uploads/")
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = os.path.join("static/uploads", file_path)

        # Run object detection
        output_image_path = draw_boxes(full_file_path)

        return render(request, 'index.html', {'image_url': output_image_path})

    return render(request, 'index.html')
