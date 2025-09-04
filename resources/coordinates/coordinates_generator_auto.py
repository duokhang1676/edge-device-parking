import cv2
import numpy as np
from ultralytics import YOLO
from modules.utils import detect_objects
model = YOLO("resources/models/detect-parking-space-yolov8n.pt")
class CoordinatesGeneratorAuto:
    def __init__(self, image, output):
        self.output = output
        self.image = image

    def generate(self):
        frame = self.image
        detected_boxes, confidences, classes = detect_objects(model, frame, 0.7)
        for i, box in enumerate(detected_boxes):
            x_min, y_min, x_max, y_max = box
            x = int((x_min+x_max)/2)
            y = int((y_min+y_max)/2)
            self.output.write("- id: " + str(i) + "\n  coordinate: [" +
                         str(x) +", "+str(y)+ "]\n")
        # x_min, y_min, x_max, y_max = detected_boxes[0]
        # x = int((x_min+x_max)/2)
        # y = int((y_min+y_max)/2) 
        # x_min2, y_min2, x_max2, y_max2 = detected_boxes[7]
        # x2 = int((x_min2+x_max2)/2)
        # y2 = int((y_min2+y_max2)/2)
        # distance = np.sqrt((x-x2)**2 + (y-y2)**2)
        # print(distance)