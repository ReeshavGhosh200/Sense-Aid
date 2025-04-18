import cv2
import numpy as np
import urllib.request
import requests
import time
import os

# Check if required files exist
required_files = ['yolov3.cfg', 'yolov3.weights', 'coco.names']
for file in required_files:
    if not os.path.exists(file):
        print(f"ERROR: Required file '{file}' not found in the current directory.")
        exit(1)

# Configuration
camera_url = 'http://192.168.1.13/cam-hi.jpg'  # Your camera URL
esp32_ip = '192.168.1.14'  # Replace with your ESP32's IP address
esp32_port = '80'  # Default HTTP port

# YOLO configuration
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Load class names
classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(f"Loaded {len(classNames)} classes from coco.names")

# Load YOLO model
print("Loading YOLO model...")
modelConfig = './yolov3.cfg'
modelWeights = './yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("YOLO model loaded successfully")

def send_to_esp32(data):
    """Send detection data to ESP32"""
    try:
        # Convert the data dict to a query string format
        query_params = '&'.join([f"{key}={'1' if value else '0'}" for key, value in data.items()])
        request_url = f"http://{esp32_ip}:{esp32_port}/update?{query_params}"
        
        # Send the GET request to ESP32
        response = requests.get(request_url, timeout=1)
        
        if response.status_code == 200:
            print(f"Successfully sent data to ESP32: {query_params}")
        else:
            print(f"Failed to send data to ESP32. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"Error sending data to ESP32: {e}")

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    
    # Create boolean flags for each object of interest
    carDetected = False
    busDetected = False
    dogDetected = False
    catDetected = False
    motorbikeDetected = False
    trainDetected = False
    truckDetected = False
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    # Process detected objects
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            class_name = classNames[classIds[i]]
            confidence = confs[i]
            
            # Check for objects of interest
            if class_name == 'car':
                carDetected = True
                print(f"CAR DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'bus':
                busDetected = True
                print(f"BUS DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'dog':
                dogDetected = True
                print(f"DOG DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'cat':
                catDetected = True
                print(f"CAT DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'motorbike':
                motorbikeDetected = True
                print(f"MOTORBIKE DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'train':
                trainDetected = True
                print(f"TRAIN DETECTED with confidence {int(confidence*100)}%")
            elif class_name == 'truck':
                truckDetected = True
                print(f"TRUCK DETECTED with confidence {int(confidence*100)}%")
            
            # Draw rectangle for detected objects
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{class_name.upper()} {int(confidence*100)}%', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Create a dictionary with all detection results
    objectsDetected = {
        'car': carDetected,
        'bus': busDetected,
        'dog': dogDetected,
        'cat': catDetected,
        'motorbike': motorbikeDetected,
        'train': trainDetected,
        'truck': truckDetected
    }
    
    return objectsDetected

def main():
    print("Starting object detection program...")
    print(f"Camera URL: {camera_url}")
    print(f"ESP32 address: http://{esp32_ip}:{esp32_port}")
    print("Press 'q' to quit")
    
    # Ask user if they want to send data to ESP32
    esp32_enabled = input("Do you want to send data to ESP32? (y/n): ").lower() == 'y'
    
    if esp32_enabled:
        print("ESP32 communication enabled")
    else:
        print("ESP32 communication disabled")
    
    while True:
        try:
            # Get image from camera
            img_resp = urllib.request.urlopen(camera_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)
            
            if im is None:
                print("Error: Unable to read image from URL.")
                time.sleep(1)
                continue

            # Process image with YOLO
            blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            
            # Get output layer names
            layernames = net.getLayerNames()
            # Handle different OpenCV versions
            if isinstance(net.getUnconnectedOutLayers()[0], (list, tuple)):
                outputNames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            else:
                outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
            
            # Detect objects
            outputs = net.forward(outputNames)
            objectsDetected = findObject(outputs, im)
            
            # Send detection data to ESP32 if enabled
            if esp32_enabled:
                send_to_esp32(objectsDetected)
            
            # Display image with detections
            cv2.imshow('Object Detection', im)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)  # Wait before trying again

    # Cleanup
    cv2.destroyAllWindows()
    print("Program terminated")


if __name__ == "__main__":
    main()