import cv2
import numpy as np
import os
import pygame
import pyttsx3

# Initialize pygame mixer
pygame.mixer.init()


# Function to handle mouse click events
def click_button(event, x, y, flags, params):
    button_person=False
    if event == cv2.EVENT_LBUTTONDOWN:
        # Define the region of the button
        roi = {'x': 200, 'y': 100, 'width': 600, 'height': 600}
        if roi['x'] <= x <= roi['x'] + roi['width'] and \
           roi['y'] <= y <= roi['y'] + roi['height']:
            if button_person == False:
                engine = pyttsx3.init()
                engine.say(class_name +" is detected")
                engine.runAndWait()
                button_person = True
        

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
# Load class Lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects Lists")
print(classes)

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create Window
cv2.namedWindow('Frame')

cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cam.read()
    roi = {'x': 200, 'y': 100, 'width': 600, 'height': 600}
    roi_frame = frame[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']]
    
    
    
    # Create button
    cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 0), -1)
    cv2.putText(frame, "Click In the green region to detect !", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    # Object Detection
    (class_ids, scores, bboxes) = model.detect(roi_frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        x += roi['x']
        y += roi['y']
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 3)
    # Draw ROI rectangle
    cv2.rectangle(frame, (roi['x'], roi['y']), (roi['x']+roi['width'], roi['y']+roi['height']), (0, 255, 0), 2)
    
    # Add text "Put objects here" to the ROI
    cv2.putText(frame, "Put objects here", (roi['x'] + 10, roi['y'] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
    
    

    
    
    cv2.imshow("Frame", roi_frame)
    
    # Check for ESC key press to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release camera and close windows
cam.release()
cv2.destroyAllWindows()
