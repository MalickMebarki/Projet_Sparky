# SECTION 1 (Packages)
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

# SECTION 2 (Générateur de frames)

def spark_frame_generator(video_path, lower_colour: list[int], upper_colour: list[int], spark_threshold: int):
    """
    Prend la vidéo, et les paramètres de couleur et rend les frames avec les étincelles
    avec et sans l'arrière plan (raw frame et masked frame).
    """
    # Lecture de la vidéo
    cap = cv.VideoCapture(video_path)

    # Message d'erreur si la vidéo ne s'ouvre pas
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    # Parcourt les frames de la vidéo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Créé le masque de la frame
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_colour, upper_colour)

        # Garde la frame si les conditions des paramètres sont satisfaites
        if cv.countNonZero(mask) > spark_threshold:
            yield frame, mask
            
    cap.release()


# SECTION 3 (Régression linéaire)

def get_cluster_centroids(mask, min_cluster_area=50):
    """
    Trouve les centroïdes des clusters dans le masque en fonction de l'aire minimale définie.
    """
    # Trouve les contours dans le masque
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centroids = []
    
    # Filtre les contours avec une aire suffisante
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= min_cluster_area:
            # Calcule les moments pour trouver les centroïdes
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    return centroids

def filter_outliers(centroids: list[int], threshold=1.5):
    """
    Filtre les centroïdes en éliminant les outliers à l'aide de l'IQR.
    """
    # Vérifie si le nombre de centroïdes est insuffisant pour filtrer
    centroids = np.array(centroids)
    if centroids.shape[0] <= 1:  
        return np.array([])
    if centroids.shape[0] < 3:  
        return centroids

    # Calcule l'IQR pour les coordonnées x et y
    q1_x, q3_x = np.percentile(centroids[:, 0], [40, 60])
    q1_y, q3_y = np.percentile(centroids[:, 1], [30, 70])
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y

    # Détermine les limites pour identifier les outliers
    lower_x, upper_x = q1_x - threshold * iqr_x, q3_x + threshold * iqr_x
    lower_y, upper_y = q1_y - threshold * iqr_y, q3_y + threshold * iqr_y

    # Filtre les centroïdes en gardant ceux dans les limites
    filtered_centroids = centroids[
        (centroids[:, 0] >= lower_x) & (centroids[:, 0] <= upper_x) &
        (centroids[:, 1] >= lower_y) & (centroids[:, 1] <= upper_y)
    ]

    return filtered_centroids

def create_centroid_mask(frame, filtered_centroids: list[int]):
    """
    Crée un masque binaire où les centroïdes filtrés sont ajoutés.
    """
    # Initialise un masque vide de la taille de la frame
    centroid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Ajoute un cercle pour chaque centroïde
    for (cx, cy) in filtered_centroids:
        cv.circle(centroid_mask, (cx, cy), 3, 255, -1)
    return centroid_mask

def detect_lines_regression(centroid_mask):
    """
    Détecte une ligne de régression linéaire basée sur les centroïdes dans le masque.
    """
    # Trouve les coordonnées non nulles dans le masque
    coords = cv.findNonZero(centroid_mask)
    if coords is not None:
        coords = np.squeeze(coords)
        y = coords[:, 0]
        x = coords[:, 1]

        # Calcule la pente et l'interception de la ligne avec np.polyfit
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    return None

def draw_lines_reg(frame, slope: float, intercept: float):
    """
    Dessine la ligne de régression linéaire sur la frame donnée.
    """
    h, w = frame.shape[:2]
    
    # Détermine les points de la ligne à partir de la pente et de l'interception
    y_vals = np.array([0, h])
    x_vals = slope * y_vals + intercept

    
    # Dessine une ligne verte sur une copie de la frame
    frame_with_line = frame.copy()
    cv.line(frame_with_line,
            (int(x_vals[0]), int(y_vals[0])),
            (int(x_vals[1]), int(y_vals[1])),
            (0, 255, 0), 2)
    return frame_with_line

def plot_angle_distribution_reg(angles: list[float]):
    """
    Trace l'histogramme des angles détectés pour analyser leur distribution.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=20, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Distribution Angulaire des Lignes avec la régression linéaire")
    plt.xlabel("Angle (degrés par rapport à l'horizontale dans le sens trigonométrique)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

def mean_reg(angles: list[float]):
    """
    Calcule la moyenne et écrit un phrase reprenant celle-ci.
    """
    i = 0
    k = 0
    while i < len(angles):
        k += angles[i]
        i += 1
    print(f"L'angle moyen avec la régression linéaire est {k/i}")
    return k/i

def sd_lin_reg(angles: list[float]):
    """
    Calcule l'écart type et écrit une phrase reprenant celle-ci.
    """
    i = 0
    k = 0
    mu = np.mean(angles)
    while i < len(angles):
        k += (angles[i] - mu)**2
        i += 1
    h = k/i
    print(f"L'écart type avec la régression linéaire est {h**(1/2)}")
    return h**(1/2)

def qqlin_reg(angles: list[float], moy: float, et: float):
    """
    Trace un QQ plot des angles détectés pour évaluer leur distribution.
    """
    angles_array = np.array(angles)
    sm.qqplot(angles_array, line='45', loc=moy, scale=et)  
    plt.title("QQ Plot de la régression linéaire")
    plt.show()

# SECTION 4 (Transformée de Hough)

def detect_lines_hough(frame):
    """
    Detecte toutes les lignes de Hough avec des paramètres par défaut voulus
    (à changer en cas de l'analyse d'une vidéo différente).
    """
    
    # Prépare la frame pour la détection de ligne de Hough
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    
    # Détection des lignes de Hough avec paramètres voulus (changer ici en cas de vidéo différente)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=150, maxLineGap=450)

    # Vérifie que la ligne n'est pas nulle
    if lines is not None and len(lines) > 0:  
        return lines
    return []

def compute_hough_space(edges): 
    """
    Crée les espaces des paramètres de Hough des frames avec des événement.
    """
    # Calcule la diagonale de la frame
    diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    # Crée une liste de distances possibles (rhos) de 0 à la longueur diagonale de l'image.
    rhos = np.linspace(0, diag_len, diag_len)
    # Crée une liste de 180 angles (thetas) entre -pi/2 et pi/2.
    thetas = np.linspace(-np.pi/2, np.pi/2, 180)

    # Crée un tableau de zéros
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Récupère les éléments non-nuls
    y_idxs, x_idxs = np.nonzero(edges)

    # Parcours de chaque index de colonne dans 'x_idxs' et chaque index de ligne dans 'y_idxs'
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # Parcours de chaque angle 'theta' dans la liste 'thetas' et de son index associé
        for theta_idx, theta in enumerate(thetas):
            
            # Calcul de la distance 'rho' à partir de la définition
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            # Vérifie si rho est entre 0 et la diagonale
            if rho >= 0 and rho < diag_len:
                # Trouver l'indice 'rho_idx' dans le tableau 'rhos' où la valeur de 'rho' est la plus proche
                rho_idx = np.argmin(np.abs(rhos - rho))
                # Incrémentation du compteur
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas


def plot_hough_space(accumulator, rhos: list[float], thetas: list[float], frame_count: int, output_folder):
    """
    Trace l'espace des paramètres de Hough en style "heatmap"'
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(
        accumulator,
        cmap='hot',
        extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]],
        aspect='auto',
    )
    plt.title(f"Espace des paramètres de Hough pour la frame {frame_count}")
    plt.xlabel("Theta (degrés par rapport à l'angle entre rho et l'horizontale dans le sens horaire')")
    plt.ylabel("Rho (pixels)")
    plt.colorbar(label="Votes")
    plt.grid(True)

    hough_space_filename = os.path.join(output_folder, f"hough_space_frame_{frame_count}.jpg")
    plt.savefig(hough_space_filename)
    plt.close()

def filter_lines_by_best(lines: np.ndarray):
    """
    Garde seulement la meilleure ligne (la première détéctée) de la liste et prend son angle
    par rapport à l'horizontale'
    """
    if lines is None or len(lines) == 0:
        return [], []

    # Définition de la meilleure ligne avec cv.HoughLinesP
    best_line = [lines[0]]
    x1, y1, x2, y2 = best_line[0][0]
    # Calcul de l'angle par rapport à l'horizontale dans le sens trigonométrique
    angle = np.arctan2(y1 - y2, x2 - x1) * 180 / np.pi
    # Modification pour avoir des angles entre 0 et 180 degrés
    if angle < 0:
        angle += 180

    return best_line, [angle]

def draw_lines_hough(frame, lines: np.ndarray):
    """
    Dessine les lignes filtrées sur chaque frame.
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def plot_angle_distribution_hough(angles: list[float]):
    """
    Trace un histogramme des angles des meilleures lignes
    """
    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=20, range=(0, 180), color='blue', edgecolor='black')
    plt.title("Distribution angulaire des meilleures lignes de Hough")
    plt.xlabel("Angle (degrés par rapport à l'horizontale dans le sens trigonométrique)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

def mean_hough(angles: list[float]):
    """
    Calcule la moyenne et écrit un phrase reprenant celle-ci.
    """
    i = 0
    k = 0
    while i < len(angles):
        k += angles[i]
        i += 1
    print(f"L'angle moyen des angles avec les lignes de Hough est {k/i}")
    return k/i

def sd_hough(angles: list[float]):
    """
    Calcule l'écart type et écrit une phrase reprenant celle-ci.
    """
    i = 0
    k = 0
    mu = np.mean(angles)
    while i < len(angles):
        k += (angles[i] - mu)**2
        i += 1
    h = k/i
    print(f"L'écart type des angles avec les lignes de Hough est {h**(1/2)}")
    return h**(1/2)

def qqhough(angles: list[float], moy: float, et: float):
    """
    Trace un qqplot 
    """
    angles_array = np.array(angles)
    sm.qqplot(angles_array, line='45', loc = moy , scale = et)  
    plt.title("QQ Plot avec les lignes de Hough")
    plt.show()

# SECTION 5 (Main function)

def main(video_path, lower_colour: list[int], upper_colour: list[int], spark_threshold: int, output_folder):
    """
    Fonction principale qui regroupe toutes les fonctions précédentes et d'autres détails.
    """
    
    # Crée un fichier contenant tous les résultats
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Définit les sous-fichier contenant des images
    raw_frames_folder = os.path.join(output_folder, "raw_spark_frames")
    cut_frames_folder = os.path.join(output_folder, "cut_spark_frames")
    regression_lines_folder = os.path.join(output_folder, "frames_with_regression_lines")
    filtered_lines_folder = os.path.join(output_folder, "frames_with_hough_lines")
    hough_space_folder = os.path.join(output_folder, "hough_spaces")

    # Crée les sous-fichiers
    for folder in [raw_frames_folder, cut_frames_folder, hough_space_folder,
                   filtered_lines_folder, regression_lines_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    frame_count = 0
    frame_count_hough = 0
    all_angles_hough = []
    all_angles_reg = []

    # Traitement des frames
    for frame, mask in spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
        
        # Sauvegarde les frames avec étincelles
        raw_frame_filename = os.path.join(raw_frames_folder, f"raw_frame_{frame_count}.jpg")
        cv.imwrite(raw_frame_filename, frame)

        # Applique le masque
        pink_frame = cv.bitwise_and(frame, frame, mask=mask)
        cut_filename = os.path.join(cut_frames_folder, f"cut_frame_{frame_count}.jpg")
        cv.imwrite(cut_filename, pink_frame)

        # Prépare la frame pour l'espace des paramètres de Hough
        gray = cv.cvtColor(pink_frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        # Calcule l'espace des paramètres de Hough
        accumulator, rhos, thetas = compute_hough_space(edges)
        plot_hough_space(accumulator, rhos, thetas, frame_count, hough_space_folder)

        # Détection et filtrage de frames avec la transformée de Hough
        lines = detect_lines_hough(pink_frame)
        if lines is not None and len(lines) > 0:
            filtered_lines, angles = filter_lines_by_best(lines)

            # Dessine la meilleure ligne de Hough sur la frame
            frame_with_filtered_lines = draw_lines_hough(pink_frame.copy(), filtered_lines)
            filtered_line_filename = os.path.join(filtered_lines_folder, f"filtered_lines_frame_{frame_count_hough}.jpg")
            cv.imwrite(filtered_line_filename, frame_with_filtered_lines)

            frame_count_hough += 1
            all_angles_hough.extend(angles)

            # Montre les frames avec des lignes de Hough
            cv.imshow("Frame with Filtered Lines", frame_with_filtered_lines)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        # Fait les centroïdes sur chaque cluster
        centroids = get_cluster_centroids(mask)

        # Filtre les centroïdes pour éliminer les outliers
        centroids = filter_outliers(centroids, threshold=1.5)  

        # Crée un masque binaire pour les centroïdes
        centroid_mask = create_centroid_mask(frame, centroids)

        # Détection des paramètres de la ligne de régression à partir des centroïdes
        line_params = detect_lines_regression(centroid_mask)

        # Dessine les centroïdes sur la frame
        for cx, cy in centroids:
            cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1) 
        
        # Détecte les lignes avec la régression et les sauvegarde
        if line_params is not None:
            slope, intercept = line_params
            angle = 90 + np.arctan(slope) * 180 / np.pi
            all_angles_reg.append(angle)

            frame_with_regression_lines = draw_lines_reg(pink_frame.copy(), slope, intercept)
            regression_line_filename = os.path.join(regression_lines_folder, f"regression_lines_frame_{frame_count}.jpg")
            cv.imwrite(regression_line_filename, frame_with_regression_lines)

            frame_count += 1

            # Montre les frames avec des lignes de régression
            cv.imshow("Frame with Regression Lines", frame_with_regression_lines)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break

    # Trace les histogrammes des distributions angulaires
    plot_angle_distribution_reg(all_angles_reg)
    plot_angle_distribution_hough(all_angles_hough)

    # Trace les qqplots des distributions angulaires et donne les moyennes et les écarts-types
    qqlin_reg(all_angles_reg, mean_reg(all_angles_reg), sd_lin_reg(all_angles_reg))
    qqhough(all_angles_hough, mean_hough(all_angles_hough), sd_hough(all_angles_hough))
    
    # Test de Shapiro-Wilk (test de normalité) pour trouver une valeur p 
    print(f"La p valeur de Shapiro-Wilk avec la régression linéaire est: {shapiro(all_angles_reg)[1]}")
    print(f"La p valeur de Shapiro-Wilk avec la régression linéaire est: {shapiro(all_angles_hough)[1]}")
    
    
    cv.destroyAllWindows()

    
# SECTION 6 (Paramètres)

# Paramètres de couleur
lower_colour = np.array([130, 60, 60])  
upper_colour = np.array([180, 255, 255])  
spark_threshold = 100  

# Paramètres de la localisation des fichiers dans l'ordinateur
video_path = "sparky.mp4"  
output_folder = "Merged_Output"

# Fonction principale
main(video_path, lower_colour, upper_colour, spark_threshold, output_folder)
