# 1. Business Understanding
## Project Objectives
- Detect and track vehicles from video
- Count the number of cars passing a given line
- Calculate the speed of the car
- We define a line for counting cars and a variable for tracking

## 2. Data understanding
- You need to understand the structure of the video, such as:
- Frame Size
- Frame Rate
- Total Number of Frames

## 3. Data Preparation
- Convert BGR to RGB
- Use YOLO to detect objects in the frame
- Merge bounding boxes of target
- Load YOLO model to detect cars
- Transform data into the format required by the model
- Filter data to get only cars

# 4. Modeling
## Object Detection & Tracking
- Detect vehicles with YOLO
- Use DeepSORT to track each object and calculate the speed of the car
- Count the number of vehicles crossing the line

## 5. Evaluation
- Displays bounding box and vehicle ID
- Displays speed of each vehicle
- Records video results
- Measures the accuracy of detection and tracking, such as
- Is the number of vehicles detected correct?
- Is the speed calculated correctly?

# 6. Deployment
## Implementation
- Output the results via OpenCV
- Save the video as `video_out.mp4`
- Improve the model accuracy
- Save the processed video and make the system work in real time
