import cv2
import numpy as np
import urllib.request
import requests
import time

# Configuration
camera_url = 'http://192.168.134.171/cam-hi.jpg'
esp32_ip = '192.168.134.184'  # Replace with your WiFi ESP32's IP

# YOLO configuration
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Load class names
classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def calculate_distance(box, frame_width):
    x, y, w, h = box
    return int((frame_width / w) * 10)  # Simple distance estimation

def send_to_esp32(detections, distance):
    try:
        url = f"http://{esp32_ip}/update?distance={distance}"
        
        # Add detection parameters
        for obj, detected in detections.items():
            url += f"&{obj}={1 if detected else 0}"
        
        print(f"[NETWORK] Sending to ESP32: {url}")
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            print("[NETWORK] Data sent successfully")
        else:
            print(f"[NETWORK] Failed to send data. Status: {response.status_code}")
    except Exception as e:
        print(f"[NETWORK] Error sending to ESP32: {e}")

def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    detections = {
        'bus': False,
        'car': False,
        'truck': False,
        'train': False,
        'banana': False,
        'bed': False,
        'bench': False,
        'bicycle': False,
        'chair': False,
        'motorcycle': False
    }
    distance = 200  # Default distance
    
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
    
    # Fix for the flatten error
    if len(bbox) > 0:
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        
        if len(indices) > 0:
            # Handle both single and multiple detections
            if isinstance(indices, (tuple, list)):
                indices = indices[0]
            
            for i in indices:
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                class_name = classNames[classIds[i]].lower()
                
                if class_name in detections:
                    detections[class_name] = True
                    distance = calculate_distance(box, wT)
                    
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(img, f'{class_name} {distance}cm', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                    print(f"[DETECTION] Found {class_name} at {distance}cm")
    
    return detections, distance

def main():
    print("[SYSTEM] Starting object detection...")
    print(f"[CAMERA] Using URL: {camera_url}")
    print(f"[NETWORK] Target ESP32 IP: {esp32_ip}")
    
    while True:
        try:
            # Get camera frame
            print("[CAMERA] Capturing frame...")
            img_resp = urllib.request.urlopen(camera_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)
            
            if img is None:
                print("[CAMERA] Failed to get image")
                time.sleep(1)
                continue
            
            # Process with YOLO
            print("[AI] Processing image...")
            blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
            net.setInput(blob)
            
            layerNames = net.getLayerNames()
            outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            
            # Detect objects
            detections, distance = find_objects(outputs, img)
            print(f"[AI] Detection complete. Distance: {distance}cm")
            
            # Send to ESP32
            send_to_esp32(detections, distance)
            
            # Display
            cv2.imshow('Object Detection', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)
    
    print("[SYSTEM] Shutting down...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()