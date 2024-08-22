import cv2
import numpy as np
import os

def draw_arrows(img, points1, points2, color=(0, 255, 0)):
    for (x1, y1), (x2, y2) in zip(points1, points2):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.arrowedLine(img, pt1, pt2, color, 1, tipLength=0.4)
    return img

def compute_average_flow(points1, points2):
    flow_vectors = points2 - points1
    avg_flow_vector = np.mean(flow_vectors, axis=0)
    return avg_flow_vector[0], avg_flow_vector[1]

def process_and_save_frames():
    frame_index = 1
    output_folder = 'optical_flow/'
    os.makedirs(output_folder, exist_ok=True)

    lk_params = dict(winSize=(15, 15), maxLevel=3,
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

        p0 = cv2.goodFeaturesToTrack(gray1_top, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1_top, gray2_top, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0 and len(good_old) > 0:
                avg_fx, avg_fy = compute_average_flow(good_old, good_new)

                flow_frame = img1.copy()
                flow_frame[:h//2, :] = draw_arrows(flow_frame[:h//2, :], good_old, good_new)

                center_point = (flow_frame.shape[1] // 2, flow_frame.shape[0] // 4)
                arrow_tip = (int(center_point[0] + avg_fx * 10), int(center_point[1] + avg_fy * 10))
                cv2.arrowedLine(flow_frame, center_point, arrow_tip, (0, 0, 255), 2, tipLength=0.5)

                output_path = os.path.join(output_folder, f'frame_{frame_index:05d}.jpg')
                cv2.imwrite(output_path, flow_frame)

        frame_index += 1

if __name__ == '__main__':
    process_and_save_frames()
