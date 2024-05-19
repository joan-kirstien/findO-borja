from ultralytics import YOLO
import cv2
import math

class ObjectDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0        
        
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.classNames = ["person", "bicycle", "car", "wallet", "keys", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        
        self.cm_to_feet = 0.0328084  # 1 centimeter = 0.0328084 feet

    def detect_objects(self, frame):
        results = self.model(frame, stream=True)
        detected_objects_info = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                confidence = math.ceil((box.conf[0]*100))/100
                
                obj_info = {
                    "class": self.classNames[cls],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "distance": None  # Optional: calculate distance based on bbox and camera parameters
                }
                
                detected_objects_info.append(obj_info)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f"{self.classNames[cls]}: {confidence*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return detected_objects_info

    def get_near_instructions(self, object_name):
        detected_objects_info = self.detect_objects(None)  # Use None as a placeholder for the frame argument
        
        for obj_info in detected_objects_info:
            if obj_info['class'] == object_name:
                target_x, target_y = obj_info['bbox'][0], obj_info['bbox'][1]
                current_x, current_y = 100, 100  # Starting position
                
                dx = target_x - current_x
                dy = target_y - current_y
                
                if dx > 0:
                    direction_x = "right"
                elif dx < 0:
                    direction_x = "left"
                else:
                    direction_x = "straight"
                
                if dy > 0:
                    direction_y = "up"
                elif dy < 0:
                    direction_y = "down"
                else:
                    direction_y = "straight"
                
                return f"Move {abs(dx)} units {direction_x} and {abs(dy)} units {direction_y} to get near the {object_name}."
        
        return "Object not found."

    def start_detecting(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            detected_objects_info = self.detect_objects(img)
            
            # Process detected objects here, e.g., display them or use get_near_instructions
            
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()