import cv2
import time
import numpy as np
from datetime import datetime, timedelta

import csv_writer
from common import *
from main import DEBUG_MODE

# Initialize video reader
cap = cv2.VideoCapture('LaGrange_KentuckyUSA_12hours.mp4')

TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
FPS = cap.get(cv2.CAP_PROP_FPS)
TAKE_EACH_NTH_FRAME = int(TAKE_FRAME_EACH_NTH_SECONDS * FPS)

video_current_datetime = datetime(2018, 12, 24, 21, 48, 00)

# Initialize YOLOv3
net = cv2.dnn.readNetFromDarknet('./yolov3.cfg', './yolov3.weights')

layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

## Colors to outline detected objects
COLORS = {key: np.random.randint(0, 255, size=3, dtype="uint8") for key in CLASSES.keys()}

## Detect params
CONFIDENCE = 0.6
THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

def try_get_frame():
    global video_current_datetime

    # Skip n frames
    for frame_count in range(TAKE_EACH_NTH_FRAME):
        cap.grab()

    video_current_datetime += timedelta(seconds=TAKE_FRAME_EACH_NTH_SECONDS)
    return cap.retrieve()

def detect_on_frame(frame: cv2.Mat):
    # Process frame
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layersOutputs = net.forward(layer_names)

    boxes = []
    confidences = []
    classIDs = []
    (H, W) = frame.shape[:2]

    # Iterate over outputs
    for output in layersOutputs:
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Skip not needed classes
            if CLASSES.get(classID) is None:
                continue

            # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Remove unnecessary boxes using non-maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, THRESHOLD, NMS_THRESHOLD)

    # Draw debug images
    if DEBUG_MODE and len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Count each class objects
    counted_classes = {key: 0 for key in CLASSES.keys()}
    if len(idxs) > 0:
        for i in idxs.flatten():
            counted_classes[classIDs[i]] += 1

    return list(counted_classes.values())

def close_all():
    csv_writer.close()
    cap.release()
    cv2.destroyAllWindows()

def process():
    local_start_time = time.time()
    last_print_time = time.time()
    processed_frames_count = 0

    while True:
        ret, frame = try_get_frame()

        if not ret:
            break

        counted_classes = detect_on_frame(frame)
        csv_writer.write(video_current_datetime, counted_classes)

        # Estimate processing time left
        processed_frames_count += 1
        local_current_time = time.time()
        if local_current_time - last_print_time > 60:
            processed_video_frames_count = processed_frames_count*(TAKE_EACH_NTH_FRAME + 1)
            avg_frame_processing_speed = processed_video_frames_count/(local_current_time - local_start_time)
            frames_left = TOTAL_FRAMES - processed_video_frames_count
            seconds_left = frames_left/avg_frame_processing_speed
            hours_left = int(seconds_left//3600)
            minutes_left = int(seconds_left//60 - hours_left*60)
            seconds_left = int(seconds_left % 60)
            print(f'{datetime.now().strftime("%H:%M:%S")}\t'
                  f'Processed {100*processed_video_frames_count/TOTAL_FRAMES:.2f}% of the video. '
                  f'Approximate time left: {hours_left}h {minutes_left}m {seconds_left}s')
            last_print_time = local_current_time

        # Print image for debug purposes
        if DEBUG_MODE:
            cv2.imshow('Video', frame)

            # Wait key
            if cv2.waitKey(10) == ord('q'):
                break

    # Close everything
    close_all()