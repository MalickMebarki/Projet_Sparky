import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
    """
    Generate frames with a spark based on color thresholding.
    """
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

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
    """
    Get centroids of color clusters from the mask, filtering out small clusters.
    :param mask: Binary mask of the target color.
    :param min_cluster_area: Minimum area for a cluster to be considered.
    :return: List of centroids (x, y) for valid clusters.
    """
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


def create_centroid_mask(frame, centroids):
    """
    Create a binary mask where each centroid is a white pixel.
    """
    centroid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Blank mask
    for (cx, cy) in centroids:
        cv.circle(centroid_mask, (cx, cy), 3, 255, -1)  # Draw circle for each centroid
    return centroid_mask


def detect_hough_line_from_centroids(centroid_mask):
    """
    Detect the best Hough line based on centroids in the mask.
    """
    edges = cv.Canny(centroid_mask, 50, 150, apertureSize=3)
    
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=0, minLineLength=108, maxLineGap=400)
    
    if lines is not None:
        best_line = max(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2))
        return best_line
    return None


def draw_line_on_frame(frame, line):
    """
    Draw the detected Hough line on the frame.
    """
    if line is not None:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def plot_angle_distribution(angles):
    """
    Plot the angular distribution of detected lines.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Angular Distribution of Detected Lines")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    """
    Process the video to extract frames with sparks and draw the best Hough line.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_frames_folder = os.path.join(output_folder, "raw_spark_frames")
    processed_frames_folder = os.path.join(output_folder, "processed_spark_frames")

    os.makedirs(raw_frames_folder, exist_ok=True)
    os.makedirs(processed_frames_folder, exist_ok=True)

    frame_count = 0
    all_angles = []

    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        # Save the raw frame
        raw_frame_path = os.path.join(raw_frames_folder, f"raw_frame_{frame_count}.jpg")
        cv.imwrite(raw_frame_path, frame)

        # Apply mask to extract specific color areas (e.g., sparks)
        spark_frame = cv.bitwise_and(frame, frame, mask=mask)

        # Extract centroids from the mask
        centroids = get_cluster_centroids(mask)

        # Create a binary mask where centroids are marked as white pixels
        centroid_mask = create_centroid_mask(frame, centroids)

        # Detect the best Hough line from the centroid mask
        best_line = detect_hough_line_from_centroids(centroid_mask)

        # Draw the centroids on the frame for visualization
        for cx, cy in centroids:
            cv.circle(spark_frame, (cx, cy), 3, (0, 0, 255), -1)

        # Draw the best line on the frame
        processed_frame = draw_line_on_frame(spark_frame.copy(), best_line)

        # Save the processed frame
        processed_frame_path = os.path.join(processed_frames_folder, f"processed_frame_{frame_count}.jpg")
        cv.imwrite(processed_frame_path, processed_frame)

        # Calculate the angle of the best line (if exists)
        if best_line is not None:
            x1, y1, x2, y2 = best_line[0]
            angle = np.arctan2(y1 - y2, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180
            all_angles.append(angle)

        frame_count += 1
        cv.imshow("Processed Frame", processed_frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    # Plot angle distribution
    plot_angle_distribution(all_angles)

    cv.destroyAllWindows()


# Parameters
lower_colour = np.array([130, 60, 60])  # Adjust HSV lower bound
upper_colour = np.array([180, 255, 255])  # Adjust HSV upper bound
spark_threshold = 100  # Minimum spark threshold

video_path = "sparky.mp4"  # Replace with your video path
output_folder = "Spark_Pictures"

# Run the program
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)





