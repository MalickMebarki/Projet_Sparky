import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
    """
    Generate frames with a spark (filtered by color threshold).
    """
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Unable to open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_colour, upper_colour)

        if cv.countNonZero(mask) > spark_threshold:
            yield frame, mask

    cap.release()

def detect_lines_hough(frame):
    """
    Detect Hough lines from an image.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=150, maxLineGap=450)

    if lines is not None and len(lines) > 0:  
        return lines
    return []

def compute_hough_space(edges):
    """
    Compute the Hough transform accumulator for an edge-detected image,
    with the origin at the top-left corner and the x-axis going right and y-axis going down.
    Theta is the angle between the horizontal axis and rho.
    """
    # Define the rho and theta range
    diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
    thetas = np.linspace(0, np.pi, 180)  # Theta from 0 to pi (angle between the horizontal axis and rho)

    # Create an accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Find edge points (non-zero pixels in the edge image)
    y_idxs, x_idxs = np.nonzero(edges)

    # Iterate through each edge point
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx, theta in enumerate(thetas):
            # Calculate the rho for this edge point and theta
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            # Find the closest rho index
            rho_idx = np.argmin(np.abs(rhos - rho))
            # Increment the accumulator at the appropriate (rho_idx, theta_idx)
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas


def plot_hough_space(accumulator, rhos, thetas, frame_count, output_folder):
    """
    Plot and save the Hough space as a heatmap.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(accumulator, cmap='hot', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect='auto')
    plt.title(f"Hough Space for Frame {frame_count}")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Rho (pixels)")
    plt.colorbar(label="Votes")
    plt.grid(True)

    # Save the Hough space
    hough_space_filename = os.path.join(output_folder, f"hough_space_frame_{frame_count}.jpg")
    plt.savefig(hough_space_filename)
    plt.close()

def filter_lines_by_angle(lines):
    """
    Retain only the best line (first detected line) from the list.
    """
    if lines is None or len(lines) == 0:
        return [], []

    # Select the first line as the "best" line
    best_line = [lines[0]]
    x1, y1, x2, y2 = best_line[0][0]
    angle = np.arctan2(y1 - y2, x2 - x1) * 180 / np.pi
    if angle < 0:
        angle += 180

    return best_line, [angle]

def draw_lines(frame, lines):
    """
    Draw the filtered lines on the frame.
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    """
    Main function to process the video and visualize both the Hough space and lines.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_frames_folder = os.path.join(output_folder, "raw_spark_frames")
    pink_cut_frames_folder = os.path.join(output_folder, "pink_cut_spark_frames")
    hough_space_folder = os.path.join(output_folder, "hough_spaces")

    if not os.path.exists(raw_frames_folder):
        os.makedirs(raw_frames_folder)

    if not os.path.exists(pink_cut_frames_folder):
        os.makedirs(pink_cut_frames_folder)

    if not os.path.exists(hough_space_folder):
        os.makedirs(hough_space_folder)

    frame_count = 0
    all_angles = []

    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        raw_frame_filename = os.path.join(raw_frames_folder, f"raw_frame_{frame_count}.jpg")
        cv.imwrite(raw_frame_filename, frame)

        pink_frame = cv.bitwise_and(frame, frame, mask=mask)
        pink_cut_filename = os.path.join(pink_cut_frames_folder, f"pink_cut_frame_{frame_count}.jpg")
        cv.imwrite(pink_cut_filename, pink_frame)

        gray = cv.cvtColor(pink_frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        # Compute and plot the Hough space
        accumulator, rhos, thetas = compute_hough_space(edges)
        plot_hough_space(accumulator, rhos, thetas, frame_count, hough_space_folder)

        # Detect lines using Hough transform
        lines = detect_lines_hough(pink_frame)
        if lines is not None and len(lines) > 0:  
            filtered_lines, angles = filter_lines_by_angle(lines)

            # Draw the filtered lines on the frame
            frame_with_lines = draw_lines(pink_frame.copy(), filtered_lines)

            frame_filename = os.path.join(output_folder, f"frame_with_filtered_lines_{frame_count}.jpg")
            cv.imwrite(frame_filename, frame_with_lines)
            frame_count += 1

            all_angles.extend(angles)

            # Display the frame with lines (optional for debugging)
            cv.imshow("Frame with Filtered Lines", frame_with_lines)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

    plot_angle_distribution(all_angles)
    cv.destroyAllWindows()

def plot_angle_distribution(angles):
    """
    Plot a histogram of the angles of the detected lines.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Angle Distribution of Lines")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

lower_colour = np.array([130, 60, 60])  
upper_colour = np.array([180, 255, 255])  
spark_threshold = 100  

video_path = "sparky.mp4"  
output_folder = "Merged_Output"

# Run the main function
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)






