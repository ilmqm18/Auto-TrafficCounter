import cv2
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def merge_close_detections(detections, distance_threshold):
    merged_detections = []
    used_indices = set()
    for i in range(len(detections)):
        if i in used_indices:
            continue
        x1_i, y1_i, w_i, h_i = detections[i][0]
        conf_i = detections[i][1]
        cls_i = detections[i][2]
        cx_i = x1_i + w_i / 2
        cy_i = y1_i + h_i / 2
        merged_box = [x1_i, y1_i, w_i, h_i]
        total_conf = conf_i
        count = 1
        for j in range(i + 1, len(detections)):
            if j in used_indices:
                continue
            x1_j, y1_j, w_j, h_j = detections[j][0]
            conf_j = detections[j][1]
            cls_j = detections[j][2]
            cx_j = x1_j + w_j / 2
            cy_j = y1_j + h_j / 2
            distance = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
            if distance < distance_threshold:
                x1 = min(merged_box[0], x1_j)
                y1 = min(merged_box[1], y1_j)
                x2 = max(merged_box[0] + merged_box[2], x1_j + w_j)
                y2 = max(merged_box[1] + merged_box[3], y1_j + h_j)
                merged_box = [x1, y1, x2 - x1, y2 - y1]
                total_conf += conf_j
                count += 1
                used_indices.add(j)
        merged_detections.append([merged_box, total_conf / count, cls_i])
        used_indices.add(i)
    return merged_detections

model = YOLO("yolo11x.pt")
cap = cv2.VideoCapture('video_in.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('video_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

tracker = DeepSort(max_age=3, n_init=1, max_cosine_distance=0.3)
line_position = 500
car_count = 0
passed_ids = set()
track_history = {}
min_width = 50
min_height = 50
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Variables for speed calculation
prev_positions = {}  # Store previous position of cars to calculate speed
prev_time = {}  # Store previous time to calculate speed

with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                if classes[i] in [2, 5, 7]:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    if width < min_width or height < min_height:
                        continue
                    confidence = confidences[i]
                    detection_class = int(classes[i])
                    detection = [ [x1, y1, width, height], confidence, detection_class ]
                    detections.append(detection)

        distance_threshold = 50
        detections = merge_close_detections(detections, distance_threshold)

        if len(detections) == 0:
            detections = np.empty((0, 5))

        outputs = tracker.update_tracks(detections, frame=frame)
        overlay = frame.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (0, int(line_position) - 30), (frame.shape[1], int(line_position) + 30), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for track in outputs:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((x_center, y_center))

            # Calculate speed if there is a previous position
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                prev_time_stamp = prev_time[track_id]
                
                # Calculate distance and time difference
                distance = np.sqrt((x_center - prev_x) ** 2 + (y_center - prev_y) ** 2)
                time_diff = 1 / fps  # Time difference between frames
                speed = distance / time_diff  # Speed (in pixels per second)

                # Convert to km/h (assuming the scale is 1 pixel = 1 meter)
                speed_kmh = speed * 3.6
                speed_text = f"Speed: {speed_kmh:.2f} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Store current position and time
            prev_positions[track_id] = (x_center, y_center)
            prev_time[track_id] = pbar.n / fps

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

            if track_id not in passed_ids:
                if line_position - 30 <= y_center <= line_position + 30:
                    car_count += 1
                    passed_ids.add(track_id)

        text = f"Car Count: {car_count}"
        cv2.putText(frame, text, (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        pbar.update(1)
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
