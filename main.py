import cv2
import RPi.GPIO as GPIO
import time
import os

# Define GPIO pins
TRIG = 23
ECHO = 24

def distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    return distance

def object_detection():
    # Initialize object detection
    classNames = []
    classFile = "/home/bhavy/Desktop/Object_Detection_Files/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    configPath = "/home/bhavy/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "/home/bhavy/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Capture frames and perform object detection
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()

        # Perform object detection
        classIds, confs, bbox = net.detect(img, confThreshold=0.45, nmsThreshold=0.2)
        if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1]
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Display the output image with object detection
        cv2.imshow("Output", img)
        cv2.waitKey(1)

        # Get distance from ultrasonic sensor
        dist = distance()
        print("Distance: {} cm".format(dist))

        # Speak based on distance
        if dist < 30:
            os.system('espeak "Stop"')  # Speak "Stop"
        else:
            os.system('espeak "Move on"')  # Speak "Move on"

# Execute the main function
if _name_ == "_main_":
    # Initialize ultrasonic sensor
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    # Run object detection
    object_detection()

    # Cleanup GPIO
    GPIO.cleanup()