import cv2
import os

def extract_frames(video_path, output_folder, frame_interval_ms):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Capture the video from the specified path
    cap = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval in frames
    frame_interval = int(fps * (frame_interval_ms / 1000.0))
    frame_interval = 4
    print(frame_interval)
    count = 0
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame if it's the desired interval
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_number += 1
        
        count += 1
    
    cap.release()
    print(f"Extracted {frame_number} frames.")

# Example usage
video_path = 'data/8.mp4'
output_folder = 'output_frames'
frame_interval_ms = 600  # Interval in milliseconds (e.g., 1000ms = 1 second)

extract_frames(video_path, output_folder, frame_interval_ms)
