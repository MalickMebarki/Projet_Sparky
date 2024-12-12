import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

#SECTION 1 (générateur de frames + fonction mean)

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

def mean(liste):
    i = 0
    k = 0
    while i < len(liste):
        k += liste[i]
        i += 1
    return k/i

#SECTION 2 (linear regression with clustering)

def get_cluster_centroids(mask, min_cluster_area=50):
  
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_cluster_area:
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

def draw_lines_reg(frame, slope, intercept):
   
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

def plot_angle_distribution_reg(angles):
    """
    Afficher un graphique de la distribution angulaire des lignes détectées.
    Paramètres:
    - angles: Liste des angles des lignes détectées.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Distribution Angulaire des Lignes avec lin reg")
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

def mean_reg(angles):
    i = 0
    k = 0
    while i < len(angles):
        k += angles[i]
        i += 1
    return print(f"L'angle moyen avec la reg lin est {k/i}")

def sd_lin_reg(angles):
    i = 0
    k = 0
    mu = mean(angles)
    while i < len(angles):
        k += (angles[i] - mu)**2
        i += 1
    h = k/i
    return print(f"L'écart type avec la lin reg est {h**(1/2)}")
    
#SECTION 3 (Hough transform)

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

def compute_hough_space(edges): #ATTENTION C'EST L'ANGLE PAR RAPPORT A LA VERTICALE JE DOIS ENCORE CHANGER
    """
    Compute the Hough transform accumulator for an edge-detected image,
    with the origin at the top-left corner and the x-axis going right and y-axis going down.
    Theta is the angle between the horizontal axis and rho.
    """
    # Define the rho and theta range
    diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    rhos = np.linspace(0, diag_len, diag_len)  # Rho range is strictly positive, up to diag_len
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
            if rho >= 0 and rho < diag_len:  # Ensure rho is within the range [0, diag_len)
                # Find the closest rho index
                rho_idx = np.argmin(np.abs(rhos - rho))
                # Increment the accumulator at the appropriate (rho_idx, theta_idx)
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas


def plot_hough_space(accumulator, rhos, thetas, frame_count, output_folder):
    """
    Plot and save the Hough space as a heatmap.
    """
    import os
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(
        accumulator,
        cmap='hot',
        extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]],
        aspect='auto',
    )
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

def draw_lines_hough(frame, lines):
    """
    Draw the filtered lines on the frame.
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def plot_angle_distribution_hough(angles):
    """
    Plot a histogram of the angles of the detected lines.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Angle Distribution of Lines with hough")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def mean_hough(angles):
    i = 0
    k = 0
    while i < len(angles):
        k += angles[i]
        i += 1
    return print(f"L'angle moyen avec hough est {k/i}")

def sd_hough(angles):
    i = 0
    k = 0
    mu = mean(angles)
    while i < len(angles):
        k += (angles[i] - mu)**2
        i += 1
    h = k/i
    return print(f"L'écart type avec hough est {h**(1/2)}")

#SECTION 4 (Main function)

def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    """
    Main function to process the video and visualize both the Hough space and lines.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the necessary folders
    raw_frames_folder = os.path.join(output_folder, "raw_spark_frames")
    pink_cut_frames_folder = os.path.join(output_folder, "pink_cut_spark_frames")
    hough_space_folder = os.path.join(output_folder, "hough_spaces")
    filtered_lines_folder = os.path.join(output_folder, "frames_with_filtered_lines")
    regression_lines_folder = os.path.join(output_folder, "frames_with_regression_lines")

    for folder in [raw_frames_folder, pink_cut_frames_folder, hough_space_folder,
                   filtered_lines_folder, regression_lines_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    frame_count = 0
    frame_count_hough = 0
    all_angles_hough = []
    all_angles_reg = []

    # Process frames
    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        # Save raw frames
        raw_frame_filename = os.path.join(raw_frames_folder, f"raw_frame_{frame_count}.jpg")
        cv.imwrite(raw_frame_filename, frame)

        # Apply mask and save pink-cut frames
        pink_frame = cv.bitwise_and(frame, frame, mask=mask)
        pink_cut_filename = os.path.join(pink_cut_frames_folder, f"pink_cut_frame_{frame_count}.jpg")
        cv.imwrite(pink_cut_filename, pink_frame)

        # Process Hough Transform
        gray = cv.cvtColor(pink_frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        # Compute and plot the Hough space
        accumulator, rhos, thetas = compute_hough_space(edges)
        plot_hough_space(accumulator, rhos, thetas, frame_count, hough_space_folder)

        # Detect and filter lines using Hough Transform
        lines = detect_lines_hough(pink_frame)
        if lines is not None and len(lines) > 0:
            filtered_lines, angles = filter_lines_by_angle(lines)

            # Draw the filtered lines and save the frame
            frame_with_filtered_lines = draw_lines_hough(pink_frame.copy(), filtered_lines)
            filtered_line_filename = os.path.join(filtered_lines_folder, f"filtered_lines_frame_{frame_count_hough}.jpg")
            cv.imwrite(filtered_line_filename, frame_with_filtered_lines)

            frame_count_hough += 1
            all_angles_hough.extend(angles)

            # Display the frame with filtered lines (optional)
            cv.imshow("Frame with Filtered Lines", frame_with_filtered_lines)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        centroids = get_cluster_centroids(mask)

        centroids = filter_outliers(centroids, threshold=1.5)  

        centroid_mask = create_centroid_mask(frame, centroids)

        line_params = detect_lines_regression(centroid_mask)

        for cx, cy in centroids:
            cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1) 
        
        # Detect lines using regression and save the frames
        if line_params is not None:
            slope, intercept = line_params
            angle = 90 + np.arctan(slope) * 180 / np.pi
            all_angles_reg.append(angle)

            frame_with_regression_lines = draw_lines_reg(pink_frame.copy(), slope, intercept)
            regression_line_filename = os.path.join(regression_lines_folder, f"regression_lines_frame_{frame_count}.jpg")
            cv.imwrite(regression_line_filename, frame_with_regression_lines)

            frame_count += 1

            # Display the frame with regression lines (optional)
            cv.imshow("Frame with Regression Lines", frame_with_regression_lines)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

    mean_reg(all_angles_reg)
    sd_lin_reg(all_angles_reg)
    mean_hough(all_angles_hough)
    sd_hough(all_angles_hough)
    # Plot angle distribution
    plot_angle_distribution_hough(all_angles_hough)
    plot_angle_distribution_reg(all_angles_reg)
    cv.destroyAllWindows()

    
#SECTION 6 (Paramètres)

lower_colour = np.array([130, 60, 60])  
upper_colour = np.array([180, 255, 255])  
spark_threshold = 100  

video_path = "sparky.mp4"  
output_folder = "Merged_Output"

# Run the main function
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)


