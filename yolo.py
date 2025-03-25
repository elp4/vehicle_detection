import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch

# Check GPU availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables
model = None
cap = None
lines = []
line_vehicle_counts = {}
class_vehicle_counts = {}
tracked_vehicles = {}
next_vehicle_id = 0

# Parameters
MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "D:/videos/camera4/video1.mp4"
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.1
PROCESS_EVERY_NTH_FRAME = 1
BUFFER_ZONE = 25
SAVE_VIDEO = False  # Whether to save the video
SAVE_PATH = "D:/videos/camera4/output/conf_" + str(CONFIDENCE_THRESHOLD) + "/iou_" + str(IOU_THRESHOLD) + "/yolov8m_1.mp4"

# Predefined lines
#lines = [[(738, 1252), (766, 1378)], [(2790, 1388), (2941, 1504)], [(1446, 1994), (2011, 1849)]]

# Ensure the folder exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Line and vehicle colors
line_colors = [
    (255, 190, 220),  # Lavender
    (128, 128, 0),    # Teal
    (75, 25, 230),    # Red (Crimson)
    (255, 0, 0),      # Blue
    (0, 0, 128),      # Maroon
    (0, 255, 0),      # Green
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (0, 128, 255),    # Orange
    (255, 0, 128),    # Purple
    (212, 190, 250),  # Pink
    (255, 255, 0),    # Cyan
    (0, 128, 128)     # Olive
]
vehicle_class_colors = {
    2: (0, 255, 0),    # Car: Green
    3: (0, 255, 255),  # Motorcycle: Yellow
    5: (255, 0, 0),    # Bus: Blue
    7: (255, 0, 255)   # Truck: Magenta
}


def select_lines():
    global lines, line_vehicle_counts, class_vehicle_counts

    if lines:  
        print("Using predefined lines.")
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            return
    
        cv2.namedWindow('Select Lines', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Lines', 1280, 720)
        cv2.imshow('Select Lines', frame)
        cv2.setMouseCallback('Select Lines', click_event, {'frame': frame})
    
        print("Click points to create lines. Press 'q' when done selecting.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    for i in range(len(lines)):
         line_vehicle_counts[i] = 0
         class_vehicle_counts[i] = {2: 0, 3: 0, 5: 0, 7: 0}

    print("lines =",lines)

def click_event(event, x, y, flags, params):
    global lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(lines) == 0 or len(lines[-1]) == 2:
            lines.append([(x, y)])
        else:
            lines[-1].append((x, y))
            color = line_colors[(len(lines) - 1) % len(line_colors)]
            cv2.line(params['frame'], lines[-1][0], lines[-1][1], color, 10)
        
        cv2.circle(params['frame'], (x, y), 5, (255, 255, 255), -1)
        cv2.imshow('Select Lines', params['frame'])

def process_video():
    global next_vehicle_id
    frame_count = 0

    cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Detection', 1280, 720)

    # Set up video writer if saving
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Error: Could not create video file at {SAVE_PATH}")
            exit()

        # Total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Processing video... Please wait.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_NTH_FRAME != 0:
            continue

        
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=[2, 3, 5, 7], verbose=False, device=device)
        update_tracking(results[0], frame)
        draw_results(frame)

        if SAVE_VIDEO:
            out.write(frame)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = (current_frame / total_frames) * 100
            print(f"Progress: {progress:.2f}%", end='\r')
        else:
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def update_tracking(result, frame):
    global next_vehicle_id, tracked_vehicles
    current_vehicles = set()

    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_id = int(cls.item())
        confidence = conf.item()

        vehicle_id = match_vehicle(box.tolist())
        if vehicle_id is None:
            vehicle_id = next_vehicle_id
            next_vehicle_id += 1
            tracked_vehicles[vehicle_id] = {
                'box': box.tolist(),
                'class_id': class_id,
                'confidence': confidence,
                'counted_at_line': None,
                'touched_line': False
            }
        else:
            tracked_vehicles[vehicle_id]['box'] = box.tolist()
            tracked_vehicles[vehicle_id]['confidence'] = confidence

        current_vehicles.add(vehicle_id)
        check_line_crossing(vehicle_id, x1, y1, x2, y2)

    tracked_vehicles = {k: v for k, v in tracked_vehicles.items() if k in current_vehicles}

def match_vehicle(new_box):
    global tracked_vehicles
    for vehicle_id, vehicle_info in tracked_vehicles.items():
        if calculate_iou(new_box, vehicle_info['box']) > IOU_THRESHOLD:
            return vehicle_id
    return None

def calculate_iou(box1, box2): 
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def check_line_crossing(vehicle_id, x1, y1, x2, y2):
    global lines, line_vehicle_counts, class_vehicle_counts
    vehicle = tracked_vehicles[vehicle_id]
    
    # Calculate center point of bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    for i, line in enumerate(lines):
        # Check if the center point has crossed the line
        if point_crosses_line((center_x, center_y), line):
            vehicle['touched_line'] = True
            if vehicle['counted_at_line'] is None:
                line_vehicle_counts[i] += 1
                class_vehicle_counts[i][vehicle['class_id']] += 1
                vehicle['counted_at_line'] = i


def point_crosses_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line
    
    # Calculate distances
    arithmitis = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
    paronomastis = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    d = arithmitis/paronomastis
    #print("d = ",d)
    
        
    # Check if point is within the line segment bounds
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return d < BUFFER_ZONE
    
    return False

def draw_results(frame):
    global lines, tracked_vehicles
    for vehicle_id, vehicle_info in tracked_vehicles.items():
        if vehicle_info['touched_line']:
            x1, y1, x2, y2 = map(int, vehicle_info['box'])
            class_id = vehicle_info['class_id']
            confidence = vehicle_info['confidence']

            color = vehicle_class_colors.get(class_id, (0, 255, 0))
            label = f'{model.names[class_id]}: {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for i, line in enumerate(lines):
        color = line_colors[i % len(line_colors)]
        cv2.line(frame, line[0], line[1], color, 10)
        

        count_text = f"car: {class_vehicle_counts[i][2]}, motorcycle: {class_vehicle_counts[i][3]}, bus: {class_vehicle_counts[i][5]}, truck: {class_vehicle_counts[i][7]}"
        
        cv2.putText(frame, f'Line {i+1}: {count_text}', (20, 50 + 30*i), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def print_results():
    global line_vehicle_counts, class_vehicle_counts
    print("Detection results:")
    for i, count in line_vehicle_counts.items():
        class_counts = class_vehicle_counts[i]
        class_counts_str = ', '.join([f"{model.names[cls_id]}: {count}" 
                                      for cls_id, count in class_counts.items()])
        print(f"Line {i+1}: {class_counts_str}")

def save_info_to_txt(elapsed_time):
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    txt_path = os.path.splitext(SAVE_PATH)[0] + '.txt'
    with open(txt_path, 'w') as f:
        f.write(f"Path to saved video: {SAVE_PATH}\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n")
        f.write(f"Processing every {PROCESS_EVERY_NTH_FRAME} frame\n\n")
        f.write(f"Time elapsed: {minutes} minutes and {seconds} seconds\n\n")
        f.write("Detection results:\n")
        for i, count in line_vehicle_counts.items():
            class_counts = class_vehicle_counts[i]
            class_counts_str = ', '.join([f"{model.names[cls_id]}: {count}" 
                                          for cls_id, count in class_counts.items()])
            f.write(f"Line {i+1}: {class_counts_str}\n")

if __name__ == "__main__":
    print(f"Attempting to open video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error opening video file {VIDEO_PATH}")
        exit()

    print(f"Using model: {MODEL_PATH}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print(f"Processing every {PROCESS_EVERY_NTH_FRAME}th frame")
    print(f"{'Saving video to: ' + SAVE_PATH if SAVE_VIDEO else 'Viewing live'}")


    model = YOLO(MODEL_PATH)
    model.to(device)

    select_lines()

    start_time = time.time()
    process_video()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Video processing completed!")
    print_results()
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    if SAVE_VIDEO:
        save_info_to_txt(elapsed_time)
        print(f"Video saved to: {SAVE_PATH}")
        print(f"Detection results saved to: {os.path.splitext(SAVE_PATH)[0] + '.txt'}")



