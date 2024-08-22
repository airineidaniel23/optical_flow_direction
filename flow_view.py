import cv2
import numpy as np
import os

def draw_arrows(img, flow, step=16, color=(0, 255, 0)):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw arrows
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 1, tipLength=0.4)
    return img

def compute_average_flow(flow):
    # Flatten the flow vectors
    fx = flow[..., 0].flatten()
    fy = flow[..., 1].flatten()

    # Compute average flow
    avg_fx = np.mean(fx)
    avg_fy = np.mean(fy)

    return avg_fx, avg_fy

def process_and_save_frames():
    frame_index = 1
    output_folder = 'optical_flow/'
    os.makedirs(output_folder, exist_ok=True)

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

        # Compute optical flow between the two consecutive frames
        flow = cv2.calcOpticalFlowFarneback(gray1_top, gray2_top, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Compute the average flow vector
        avg_fx, avg_fy = compute_average_flow(flow)

        # Draw the current flow and average arrow on the original frame
        flow_frame = img1.copy()
        flow_frame[:h//2, :] = draw_arrows(flow_frame[:h//2, :], flow)

        # Draw the average arrow in the center
        center_point = (flow_frame.shape[1] // 2, flow_frame.shape[0] // 4)
        arrow_tip = (int(center_point[0] + avg_fx * 10), int(center_point[1] + avg_fy * 10))
        cv2.arrowedLine(flow_frame, center_point, arrow_tip, (0, 0, 255), 2, tipLength=0.5)

        # Save the result to the output folder
        output_path = os.path.join(output_folder, f'frame_{frame_index:05d}.jpg')
        cv2.imwrite(output_path, flow_frame)

        frame_index += 1

if __name__ == '__main__':
    process_and_save_frames()
