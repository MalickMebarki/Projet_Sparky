import cv2 as cv
import numpy as np
from scipy.stats import linregress

def detect_and_fit_scipy(video_path, lower_colour, upper_colour, spark_threshold):
    """
    Analyse une vidéo pour détecter des particules et ajuste leur trajectoire
    par régression linéaire avec SciPy.

    Paramètres :
    - video_path : Chemin de la vidéo.
    - lower_colour : Limite HSV inférieure pour détecter le rose.
    - upper_colour : Limite HSV supérieure pour détecter le rose.
    - spark_threshold : Nombre minimum de pixels pour détecter une particule.

    Sortie :
    - Images avec les trajectoires ajustées.
    """
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la frame en espace de couleur HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Appliquer le masque pour les couleurs ciblées
        mask = cv.inRange(hsv, lower_colour, upper_colour)

        # Vérifier si le nombre de pixels roses dépasse le seuil
        if cv.countNonZero(mask) > spark_threshold:
            # Extraire les coordonnées des pixels non nuls
            coords = cv.findNonZero(mask)
            print(coords)
            if coords is not None:
                coords = np.squeeze(coords)  # Réduire la dimension si nécessaire

                # Vérifier que des points ont été détectés
                if coords.size > 0:
                    # Séparer les coordonnées x et y
                    y = coords[:, 0]  # Coordonnées x
                    x = coords[:, 1]  # Coordonnées y

                    # Régression linéaire
                    slope, intercept, r_value, p_value, std_err = linregress(y, x)

                    # Générer les coordonnées pour la droite ajustée
                    h, w = frame.shape[:2]  # Dimensions de l'image
                    y_vals = np.array([0, w])  # Bord gauche (x=0) et bord droit (x=w)
                    x_vals = slope * y_vals + intercept  # y = slope * x + intercept

                    # Restreindre y_vals à l'intérieur de l'image (clipper)
                    y_vals = np.clip(y_vals, 0, h)

                    # Tracer la droite sur la frame originale
                    frame_with_line = frame.copy()
                    cv.line(frame_with_line,
                            (int(x_vals[0]), int(y_vals[0])),  # Point de départ (x1, y1)
                            (int(x_vals[1]), int(y_vals[1])),  # Point de fin (x2, y2)
                            (0, 255, 0), 2)  # Couleur verte, épaisseur de 2 pixels

                    # Afficher l'image avec la trajectoire détectée
                    cv.imshow("Trajectoire détectée", frame_with_line)

                    # Sauvegarder l'image avec la trajectoire
                    frame_index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
                    cv.imwrite(f"spark_frame_{frame_index}.jpg", frame_with_line)

                    # Afficher quelques détails de la régression (optionnel)
                    print(f"Frame {frame_index}: Slope={slope:.2f}, Intercept={intercept:.2f}, R^2={r_value**2:.2f}")

                else:
                    # Si aucun point détecté après compression
                    print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Aucun point détecté après compression")
            else:
                print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Aucun point détecté")
        
        # Appuyer sur 'q' pour quitter
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Définir les paramètres
video_path = r"C:\Users\nerop\Videos\Sparky.mp4"
lower_colour = np.array([130, 35, 30])  # Limite HSV inférieure pour le rose
upper_colour = np.array([180, 255, 255])  # Limite HSV supérieure pour le rose
spark_threshold = 8500  # Nombre minimum de pixels pour détecter une particule

# Appeler la fonction pour détecter et ajuster les trajectoires
detect_and_fit_scipy(video_path, lower_colour, upper_colour, spark_threshold)
