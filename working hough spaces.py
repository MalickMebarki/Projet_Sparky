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

def compute_hough_space(edges):
    """
    Compute the Hough transform accumulator for an edge-detected image.
    """
    # Define the rho and theta range
    diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180)

    # Create an accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Find edge points (non-zero pixels in the edge image)
    y_idxs, x_idxs = np.nonzero(edges)

    # Iterate through each edge point
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rhos - rho))
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

def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    """
    Main function to process the video and visualize Hough spaces for frames with sparks.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        # Extract the region with the spark
        spark_frame = cv.bitwise_and(frame, frame, mask=mask)

        # Edge detection
        gray = cv.cvtColor(spark_frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        # Compute the Hough space
        accumulator, rhos, thetas = compute_hough_space(edges)

        # Plot and save the Hough space
        plot_hough_space(accumulator, rhos, thetas, frame_count, output_folder)

        frame_count += 1

        # Display the Hough space (optional for debugging)
        cv.imshow("Edges", edges)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# Parameters
lower_colour = np.array([130, 60, 60])  
upper_colour = np.array([180, 255, 255])  
spark_threshold = 100  

video_path = "sparky.mp4"  
output_folder = "Spark_Hough_Spaces"

# Run the main function
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)







