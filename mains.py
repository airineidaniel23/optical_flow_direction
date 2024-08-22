import cv2
import numpy as np

def draw_arrows(img, pts1, pts2, color=(0, 255, 0)):
    if len(pts1) == 0 or len(pts2) == 0:
        return img  # If there are no points, return the image unchanged

    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.arrowedLine(img, pt1, pt2, color, 1, tipLength=0.2)
    return img

def compute_average_flow(pts1, pts2):
    if len(pts1) == 0 or len(pts2) == 0:
        return 0, 0  # If there are no points, return zero flow

    # Compute flow vectors
    flow_vectors = pts2 - pts1

    # Compute average flow
    avg_flow = np.mean(flow_vectors, axis=0)

    return avg_flow[0], avg_flow[1]

def process_frames():
    frames = []
    average_flows = []
    frame_index = 1

    while True:
        # Load consecutive frames
        img1_path = f'output_frames/frame_{frame_index:05d}.jpg'
        img2_path = f'output_frames/frame_{frame_index + 1:05d}.jpg'

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            break

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Consider only the top 50% of the image
        h, w = gray1.shape
        gray1_top = gray1[:h//2, :]
        gray2_top = gray2[:h//2, :]

        # Detect feature points in the first frame
        p0 = cv2.goodFeaturesToTrack(gray1_top, mask=None, **feature_params)

        if p0 is None or len(p0) == 0:
            print(f"No good features to track in frame {frame_index}. Skipping to next frame.")
            frame_index += 1
            continue

        # Calculate the optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1_top, gray2_top, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Compute the average flow vector
        avg_fx, avg_fy = compute_average_flow(good_old, good_new)
        average_flows.append((avg_fx, avg_fy))
        frames.append((img1, good_old, good_new))  # Save the frame and points for later visualization

        frame_index += 1  # Move to the next pair of frames

    return frames, average_flows

def display_video_with_flow(frames, average_flows, frame_interval):
    video_path = 'data/6.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Resize factor to fit screen
    max_height = 800  # Maximum height in pixels
    scale_factor = max_height / original_height
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Create a black image of the same size as the resized video
    flow_display = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Center of the screen
    center = (new_width // 2, new_height // 2)
    current_point = center

    while True:  # Loop for the entire process
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from the beginning
        index = 0
        frame_count = 0
        current_point = center
        flow_display.fill(0)  # Reset the flow display

        while cap.isOpened() and index < len(average_flows):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the video frame
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if frame_count % frame_interval == 0:
                # Get the next average flow vector
                avg_fx, avg_fy = average_flows[index]
                next_point = (int(current_point[0] + avg_fx), int(current_point[1] + avg_fy))

                # Check if the arrow reaches the margin
                if next_point[0] < 0 or next_point[0] >= new_width or next_point[1] < 0 or next_point[1] >= new_height:
                    # Reset the screen and start from the center
                    flow_display.fill(0)
                    current_point = center
                    next_point = center

                # Draw the arrow on the black screen
                cv2.arrowedLine(flow_display, current_point, next_point, (0, 0, 255), 2, tipLength=0.5)

                current_point = next_point

                # Draw the current flow and average arrow on the original frame
                original_frame, pts1, pts2 = frames[index]
                flow_frame = original_frame.copy()

                # Draw arrows only on the top 50% of the frame
                flow_frame[:flow_frame.shape[0]//2, :] = draw_arrows(flow_frame[:flow_frame.shape[0]//2, :], pts1, pts2)

                # Draw the average arrow in the center of the top half
                center_point = (flow_frame.shape[1] // 2, flow_frame.shape[0] // 4)
                arrow_tip = (int(center_point[0] + avg_fx * 10), int(center_point[1] + avg_fy * 10))
                cv2.arrowedLine(flow_frame, center_point, arrow_tip, (0, 0, 255), 2, tipLength=0.5)

                # Resize the flow frame to match the new dimensions
                flow_frame = cv2.resize(flow_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                index += 1

            frame_count += 1

            # Combine video frame, flow display, and the current frame with flow visualization
            combined_display = np.hstack((frame, flow_display, flow_frame))

            # Display the result
            cv2.imshow('Video with Optical Flow', combined_display)

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frame_interval = 4  # This is the interval that was used to sample the frames originally
    frames, average_flows = process_frames()
    display_video_with_flow(frames, average_flows, frame_interval)
