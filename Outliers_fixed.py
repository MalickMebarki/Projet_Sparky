import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_colour, upper_colour)

        if cv.countNonZero(mask) > spark_threshold:
            yield frame, mask
            
    cap.release()
def get_cluster_centroids(mask, min_cluster_area=50):
  
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_cluster_area:  # Filter based on area
            # Moments to find the centroid of the contour
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    return centroids

def filter_outliers(centroids, threshold=1.5):
    
    centroids = np.array(centroids)
    if centroids.shape[0] < 3:  
        return centroids

    q1_x, q3_x = np.percentile(centroids[:, 0], [40, 60])
    q1_y, q3_y = np.percentile(centroids[:, 1], [30, 70])

    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y

    lower_x, upper_x = q1_x - threshold * iqr_x, q3_x + threshold * iqr_x
    lower_y, upper_y = q1_y - threshold * iqr_y, q3_y + threshold * iqr_y

    filtered_centroids = centroids[
        (centroids[:, 0] >= lower_x) & (centroids[:, 0] <= upper_x) &
        (centroids[:, 1] >= lower_y) & (centroids[:, 1] <= upper_y)
    ]

    return filtered_centroids

def create_centroid_mask(frame, filtered_centroids):
   
    centroid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (cx, cy) in filtered_centroids:
        cv.circle(centroid_mask, (cx, cy), 3, 255, -1)
    return centroid_mask



def detect_lines_regression(centroid_mask):
  
    coords = cv.findNonZero(centroid_mask)
    if coords is not None:
        coords = np.squeeze(coords)
        y = coords[:, 0]
        x = coords[:, 1]
        
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    return None

def draw_lines(frame, slope, intercept):
  
    h, w = frame.shape[:2]
    y_vals = np.array([0, h])
    x_vals = slope * y_vals + intercept
    y_vals = np.clip(y_vals, 0, h)
    x_vals = np.clip(x_vals, 0, w)
    
    frame_with_line = frame.copy()
    cv.line(frame_with_line,
            (int(x_vals[0]), int(y_vals[0])),
            (int(x_vals[1]), int(y_vals[1])),
            (0, 255, 0), 2)
    return frame_with_line

def plot_angle_distribution(angles):
   
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Distribution Angulaire des Lignes")
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_frames_folder = os.path.join(output_folder, "raw_spark_frames")
    pink_cut_frames_folder = os.path.join(output_folder, "pink_cut_spark_frames")

    if not os.path.exists(raw_frames_folder):
        os.makedirs(raw_frames_folder)

    if not os.path.exists(pink_cut_frames_folder):
        os.makedirs(pink_cut_frames_folder)

    frame_count = 0
    all_angles = []

    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        raw_frame_filename = os.path.join(raw_frames_folder, f"raw_frame_{frame_count}.jpg")
        cv.imwrite(raw_frame_filename, frame)

        pink_frame = cv.bitwise_and(frame, frame, mask=mask)
        pink_cut_filename = os.path.join(pink_cut_frames_folder, f"pink_cut_frame_{frame_count}.jpg")
        cv.imwrite(pink_cut_filename, pink_frame)

        centroids = get_cluster_centroids(mask)

        centroids = filter_outliers(centroids, threshold=1.5)  

        centroid_mask = create_centroid_mask(frame, centroids)

        line_params = detect_lines_regression(centroid_mask)

        for cx, cy in centroids:
            cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1) 
        
        if line_params is not None:
            slope, intercept = line_params
            angle = 90 + np.arctan(slope) * 180 / np.pi
            all_angles.append(angle)

            frame_with_lines = draw_lines(pink_frame.copy(), slope, intercept)

            frame_filename = os.path.join(output_folder, f"frame_with_lines_{frame_count}.jpg")
            cv.imwrite(frame_filename, frame_with_lines)

            cv.imshow("Frame avec lignes régressées", frame_with_lines)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        frame_count += 1

    cv.destroyAllWindows()

    plot_angle_distribution(all_angles)


lower_colour = np.array([130, 60, 60]) 
upper_colour = np.array([180, 255, 255]) 
spark_threshold = 100  

video_path = r"C:\Users\nerop\Desktop\Projet info\Sparky.mp4"  


output_folder = r"C:\Users\nerop\Desktop\Projet info\processed_frames"

main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)