from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'object_detection_model', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'saved_model')
model = tf.saved_model.load(MODEL_PATH)

LABELS = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 21: "Cow",
    22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe", 27: "Backpack",
    28: "Umbrella", 31: "Handbag", 32: "Tie", 33: "Suitcase", 34: "Frisbee",
    35: "Skis", 36: "Snowboard", 37: "Sports Ball", 38: "Kite", 39: "Baseball Bat",
    40: "Baseball Glove", 41: "Skateboard", 42: "Surfboard", 43: "Tennis Racket",
    44: "Bottle", 46: "Wine Glass", 47: "Cup", 48: "Fork", 49: "Knife",
    50: "Spoon", 51: "Bowl", 52: "Banana", 53: "Apple", 54: "Sandwich",
    55: "Orange", 56: "Broccoli", 57: "Carrot", 58: "Hot Dog", 59: "Pizza",
    60: "Donut", 61: "Cake", 62: "Chair", 63: "Couch", 64: "Potted Plant",
    65: "Bed", 67: "Dining Table", 70: "Toilet", 72: "TV", 73: "Laptop",
    74: "Mouse", 75: "Remote", 76: "Keyboard", 77: "Cell Phone", 78: "Microwave",
    79: "Oven", 80: "Toaster", 81: "Sink", 82: "Refrigerator", 84: "Book",
    85: "Clock", 86: "Vase", 87: "Scissors", 88: "Teddy Bear", 89: "Hair Drier",
    90: "Toothbrush"
}



def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    detection_scores = detections["detection_scores"].numpy()[0]
    detection_classes = detections["detection_classes"].numpy()[0].astype(int)
    detection_boxes = detections["detection_boxes"].numpy()[0]

    detected_objects = [
        {"label": LABELS.get(cls, "Unknown"), "confidence": score * 100}
        for cls, score in zip(detection_classes, detection_scores) if score > 0.5
    ]

    return detected_objects, detection_boxes

def draw_boxes(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    detected_objects, boxes = detect_objects(image_path)

    for i, obj in enumerate(detected_objects):
        y_min, x_min, y_max, x_max = boxes[i]
        x_min, x_max = int(x_min * w), int(x_max * w)
        y_min, y_max = int(y_min * h), int(y_max * h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"{obj['label']} {obj['confidence']:.2f}%", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join("static/uploads", "output.jpg")
    cv2.imwrite(output_path, img)
    return output_path, detected_objects

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location="static/uploads/")
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = os.path.join("static/uploads", file_path)

        output_image_path, detected_objects = draw_boxes(full_file_path)

        return render(request, 'index.html', {'image_url': output_image_path, 'detected_objects': detected_objects})

    return render(request, 'index.html')
