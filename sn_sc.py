import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import math

st.title('Real-time Object Detection using YOLOv8')

model = YOLO("model_- 25 march 2024 2_53.pt")
classNames = ["Snak", "Scorpion"]

# Function to perform object detection on the input image
def detect_objects(img):
    img = np.array(img)
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return img

# Web Application
st.text('Using the camera.')

video = st.video('camera')

while video:
    frame = video
    output_frame = detect_objects(frame)
    st.image(output_frame, channels="BGR")

st.stop()
