#!/usr/bin/env python
# coding: utf-8

import mediapipe as mp
import cv2 
import time
from tqdm import tqdm

# Load MediaPipe Pose and Drawing Utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Pose model configuration
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Function to process each frame and overlay pose landmarks
def process_frame(img):
    start_time = time.time()
    
    h, w = img.shape[0], img.shape[1]
    
    # Convert to RGB for MediaPipe processing
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Iterate through the 33 keypoints
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z

            radius = 5
            color = (0, 255, 0)  # Default color
            
            # Assign colors based on keypoint type
            if i == 0:  # Nose
                color = (0, 0, 255)
            elif i in [11, 12]:  # Shoulders
                color = (223, 155, 6)
            elif i in [23, 24]:  # Hips
                color = (1, 240, 255)
            elif i in [13, 14]:  # Elbows
                color = (140, 47, 240)
            elif i in [25, 26]:  # Knees
                color = (0, 0, 255)
            elif i in [15, 16, 27, 28]:  # Wrists and ankles
                color = (223, 155, 60)
            elif i in [17, 19, 21]:  # Left hand
                color = (94, 218, 121)
            elif i in [18, 20, 22]:  # Right hand
                color = (16, 144, 247)
            elif i in [27, 29, 31]:  # Left foot
                color = (29, 123, 243)
            elif i in [28, 30, 32]:  # Right foot
                color = (193, 182, 255)
            elif i in [9, 10]:  # Mouth
                color = (205, 235, 255)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # Eyes and cheeks
                color = (94, 218, 121)
            
            img = cv2.circle(img, (cx, cy), radius, color, -1)

    else:
        img = cv2.putText(img, 'NO Person', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2)
    
    # Calculate FPS
    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    img = cv2.putText(img, f'FPS {int(FPS)}', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (223, 155, 6), 2)
    
    return img

# Video frame-by-frame processing
def generate_video(input_path):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead
    
    print('Processing video:', input_path)
    
    # Get video capture and frame details
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define video writer with same FPS and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Progress bar setup
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process each frame
                try:
                    frame = process_frame(frame)
                except Exception as e:
                    print(f'Error processing frame: {e}')
                
                # Write the frame to output
                out.write(frame)
                pbar.update(1)
                
        except Exception as e:
            print(f'Processing interrupted: {e}')
    
    # Release resources
    cap.release()
    out.release()
    print('Video saved at:', output_path)

# Main function
if __name__ == '__main__':
    generate_video("path_to_your_video.mp4")  # Replace with the absolute path of your video file
