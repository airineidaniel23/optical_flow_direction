import cv2
import numpy as np

def initialize_points(gray_frame, max_corners=100, quality_level=0.3, min_distance=7):
    """Detect good features to track."""
    points = cv2.goodFeaturesToTrack(gray_frame, mask=None, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    return points

def compute_average_flow_lk(p0, p1):
    """Compute the average flow vector from Lucas-Kanade point pairs."""
    flow_vectors = p1 - p0
    avg_flow = np.mean(flow_vectors, axis=0)
    return avg_flow.flatten()

def process_frames_lk():
    frames = []
    average_flows = []
    frame_index = 1
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        img1_path = f'output_frames/frame_{frame_index:05d}.jpg'
        img2_path = f'output_frames/frame_{frame_index + 1:05d}.jpg'

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            break

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        h, w = gray1.shape
        gray1_top = gray1[:h//2, :]
        gray2_top = gray2[:h//2, :]

        # Initialize points to track
        p0 = initialize_points(gray1_top)

        # Check if points were found
        if p0 is None or len(p0) == 0:
            print(f"No points to track at frame {frame_index}. Skipping this frame.")
            average_flows.append((0, 0))  # Append zero flow for this frame
            frames.append(img1)  # Save the original frame for later visualization
            frame_index += 1
            continue

        # Calculate optical flow using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1_top, gray2_top, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            # Compute average flow
            avg_fx, avg_fy = compute_average_flow_lk(good_old, good_new)
            average_flows.append((avg_fx, avg_fy))
        else:
            print(f"No good points to track at frame {frame_index}. Using previous flow vector.")
            average_flows.append((0, 0))  # Append zero flow if no good points were found

        frames.append(img1)  # Save the original frame for later visualization

        frame_index += 1

    return frames, average_flows

def save_video_with_flow_on_original_video(frames, average_flows, frame_interval, original_video_path, output_video_path):
    # Open the original video
    cap = cv2.VideoCapture(original_video_path)
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))

    index = 0
    avg_fx, avg_fy = 0, 0  # Initialize with no motion

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i % frame_interval == 0 and index < len(average_flows):
            avg_fx, avg_fy = average_flows[index]
            index += 1

        # Draw the average arrow in the center of the top half of the current frame
        center_point = (original_width // 2, original_height // 4)
        arrow_tip = (int(center_point[0] + avg_fx * 10), int(center_point[1] + avg_fy * 10))
        cv2.arrowedLine(frame, center_point, arrow_tip, (0, 0, 255), 2, tipLength=0.5)

        # Write the frame with the arrow overlay
        video_writer.write(frame)

    cap.release()
    video_writer.release()

if __name__ == '__main__':
    frame_interval = 2  # Interval used to sample the frames
    original_video_path = 'data/6.mp4'
    output_video_path = 'output_with_flow_on_original_lk.mp4'

    frames, average_flows = process_frames_lk()
    save_video_with_flow_on_original_video(frames, average_flows, frame_interval, original_video_path, output_video_path)
