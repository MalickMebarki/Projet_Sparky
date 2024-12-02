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

def detect_lines_regression(mask):
    """
    Détecte les lignes dans une image en utilisant une régression linéaire.
    Paramètres:
        -mask: masque binaire où les lignes doivent être détectées
    Output:
        -line_params: paramètres de la ligne détectée (slope, intercept)
    """
    coords = cv.findNonZero(mask)
    if coords is not None:
        coords = np.squeeze(coords)

        # Échantillonner les coordonnées pour ne prendre qu'un point par coordonnée de hauteur
        unique_y_coords = np.unique(coords[:, 1])
        sampled_coords = np.array([coords[coords[:, 1] == y][0] for y in unique_y_coords])

        y = sampled_coords[:, 0]
        x = sampled_coords[:, 1]

        # Appliquer une régression linéaire avec NumPy
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    return None

def draw_lines(frame, slope, intercept):
    """
    Dessiner les lignes régressées sur la frame.
    Paramètres:
    - frame: Image sur laquelle dessiner.
    - slope: Pente de la ligne régressée.
    - intercept: Ordonnée à l'origine de la ligne régressée.
    Output:
    - Image avec les lignes dessinées.
    """
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

def remove_outliers(angles):
    """
    Retirer les outliers de la liste des angles.
    Paramètres:
    - angles: Liste des angles des lignes détectées.
    Output:
    - angles_filtrés: Liste des angles sans outliers.
    """
    angles = np.array(angles)
    Q1 = np.percentile(angles, 25)
    Q3 = np.percentile(angles, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_angles = angles[(angles >= lower_bound) & (angles <= upper_bound)]
    return filtered_angles

def plot_angle_distribution(angles):
    """
    Afficher un graphique de la distribution angulaire des lignes détectées.
    Paramètres:
    - angles: Liste des angles des lignes détectées.
    """
    filtered_angles = remove_outliers(angles)
    plt.figure(figsize=(8, 6))
    plt.hist(filtered_angles, bins=50, range=(-90, 90), color='blue', edgecolor='black')
    plt.title("Distribution Angulaire des Lignes (Sans Outliers)")
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

        # Détection des lignes par régression linéaire
        line_params = detect_lines_regression(mask)

        if line_params is not None:
            slope, intercept = line_params
            angle = np.arctan(slope) * 180 / np.pi
            all_angles.append(angle)

            frame_with_lines = draw_lines(pink_frame.copy(), slope, intercept)

            frame_filename = os.path.join(output_folder, f"frame_with_lines_{frame_count}.jpg")
            cv.imwrite(frame_filename, frame_with_lines)
            frame_count += 1

            cv.imshow("Frame avec lignes régressées", frame_with_lines)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()

    # Afficher le graphique de la distribution angulaire
    plot_angle_distribution(all_angles)

# Paramètres pour la détection des étincelles (couleur rose)
lower_colour = np.array([130, 60, 60])  # Limite inférieure pour les étincelles roses
upper_colour = np.array([180, 255, 255])  # Limite supérieure pour les étincelles roses
spark_threshold = 100  # Nombre minimum de pixels pour détecter une étincelle

# Chemin vers la vidéo
video_path = r"C:\Users\nerop\Videos\Sparky.mp4"  # Remplacer par le chemin réel de la vidéo

# Dossier de sortie pour les frames avec les lignes régressées
output_folder = r"C:\Users\nerop\Desktop\Projet info\processed_frames"

# Sauvegarder les frames avec les lignes régressées et afficher la distribution des angles
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)
