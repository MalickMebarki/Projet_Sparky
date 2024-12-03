import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
    """
    Garder les frames avec une étincelle.
    Paramètres:
        -video_path: chemin vers la vidéo
        -lower_colour: couleur du "bas"
        -upper_colour: couleur du "haut"
        -spark_threshold: nombres de pixelles entre les bornes de couleur requis pour garder la frame
    Output:
        -frame: frame avec une étincelle
        -mask: frame avec juste l'étincelle et le reste en noir'
    """
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

def detect_lines_hough(frame):
    """
    Détecte les lignes de Hough probabilistiques sur une image.
    Paramères:
        -frame: image dans laquelle nous voulons détécter des lignes
    Output:
        -lines: liste de toutes les lignes détéctées dans l'image
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=150, maxLineGap=450)

    if lines is not None and len(lines) > 0:  
        return lines
    return []

def filter_lines_by_angle(lines):
    """
    Retain only the best line (first detected line) from the list.

    Paramètres:
        - lines: Liste de lignes détectées.
    Output:
        - filtered_lines: Liste contenant uniquement la première ligne.
        - angles: Liste des angles de la première ligne.
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
    Dessiner les lignes filtrées sur la frame.

    Paramètres:
    - frame: Image sur laquelle dessiner.
    - lines: Lignes à dessiner.

    Output:
    - Image avec les lignes dessinées.
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
    return frame

def main(video_path, lower_colour, upper_colour, spark_threshold, output_folder):
    """
    Fonction globale 2.
    
    Paramètres:
        -video_path: chemin vers la vidéo
        -lower_colour: couleur du "bas"
        -upper_colour: couleur du "haut"
        -spark_threshold: nombres de pixelles entre les bornes de couleur requis pour garder la frame
    Output:
        -histogramme des angles
        -fichier contenant images utiles

    """
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

        lines = detect_lines_hough(pink_frame)

        if lines is not None and len(lines) > 0:  
            filtered_lines, angles = filter_lines_by_angle(lines)

            frame_with_lines = draw_lines(pink_frame.copy(), filtered_lines)

            frame_filename = os.path.join(output_folder, f"frame_with_filtered_lines_{frame_count}.jpg")
            cv.imwrite(frame_filename, frame_with_lines)
            frame_count += 1

            all_angles.extend(angles)

            cv.imshow("Frame avec lignes filtrées", frame_with_lines)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

    plot_angle_distribution(all_angles)
    
    cv.destroyAllWindows()

def plot_angle_distribution(angles):
    """
    Afficher un graphique de la distribution angulaire des lignes détectées .

    Paramètres:
        - angles: Liste des angles des lignes détectées.
    Output:
        -Histgramme des angles
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Distribution Angulaire des Lignes")
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

lower_colour = np.array([130, 60, 60])  
upper_colour = np.array([180, 255, 255])  
spark_threshold = 100  

video_path = "sparky.mp4"  

output_folder = "Spark pictures"

main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)





