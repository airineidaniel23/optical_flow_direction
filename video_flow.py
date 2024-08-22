import cv2
import numpy as np

def draw_arrows(img, flow, step=16, color=(0, 255, 0)):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw arrows
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 1, tipLength=0.2)
    return img

def compute_average_flow(flow):
    fx = flow[..., 0].flatten()
    fy = flow[..., 1].flatten()

    avg_fx = np.mean(fx)
    avg_fy = np.mean(fy)

    return avg_fx, avg_fy

def process_frames():
    frames = []
    average_flows = []
    frame_index = 1

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

        flow = cv2.calcOpticalFlowFarneback(gray1_top, gray2_top, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        avg_fx, avg_fy = compute_average_flow(flow)
        average_flows.append((avg_fx, avg_fy))
        frames.append((img1, flow))

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
    frame_interval = 3  # Interval used to sample the frames
    original_video_path = 'data/7.mp4'
    output_video_path = 'output_with_flow_on_original.mp4'

    frames, average_flows = process_frames()
    save_video_with_flow_on_original_video(frames, average_flows, frame_interval, original_video_path, output_video_path)
