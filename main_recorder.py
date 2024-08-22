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

def save_video_with_flow(frames, average_flows, frame_interval, output_video_path):
    # Define video properties
    original_height, original_width = frames[0][0].shape[:2]
    fps = 30  # Assume FPS, or it can be dynamically calculated based on input video if needed

    max_height = 800  # Maximum height in pixels for scaling
    scale_factor = max_height / original_height
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width * 3, new_height))

    flow_display = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    center = (new_width // 2, new_height // 2)
    current_point = center

    index = 0
    for i in range(len(frames) * frame_interval):
        original_frame, flow = frames[index]
        frame = cv2.resize(original_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        flow_display.fill(0)

        if i % frame_interval == 0 and index < len(average_flows):
            avg_fx, avg_fy = average_flows[index]
            next_point = (int(current_point[0] + avg_fx), int(current_point[1] + avg_fy))

            if next_point[0] < 0 or next_point[0] >= new_width or next_point[1] < 0 or next_point[1] >= new_height:
                flow_display.fill(0)
                current_point = center
                next_point = center

            cv2.arrowedLine(flow_display, current_point, next_point, (0, 0, 255), 2, tipLength=0.5)
            current_point = next_point

            flow_frame = original_frame.copy()
            flow_frame[:flow_frame.shape[0]//2, :] = draw_arrows(flow_frame[:flow_frame.shape[0]//2, :], flow)

            center_point = (flow_frame.shape[1] // 2, flow_frame.shape[0] // 4)
            arrow_tip = (int(center_point[0] + avg_fx * 10), int(center_point[1] + avg_fy * 10))
            cv2.arrowedLine(flow_frame, center_point, arrow_tip, (0, 0, 255), 2, tipLength=0.5)

            flow_frame = cv2.resize(flow_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            index += 1

        combined_display = np.hstack((frame, flow_display, flow_frame))
        video_writer.write(combined_display)

    video_writer.release()

if __name__ == '__main__':
    frame_interval = 3  # Interval used to sample the frames
    output_video_path = 'output_with_flow.mp4'

    frames, average_flows = process_frames()
    save_video_with_flow(frames, average_flows, frame_interval, output_video_path)
