import cv2 as cv
import numpy as np

def detect_and_fit_numpy(video_path, lower_colour, upper_colour, spark_threshold):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    # Définir les limites de la zone grise (cadre) à exclure
    gray_frame_top_left = (50, 50)  # Coin supérieur gauche (y, x)
    gray_frame_bottom_right = (400, 400)  # Coin inférieur droit (y, x)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Réduction de bruit avec un flou gaussien
        blurred = cv.GaussianBlur(hsv, (5, 5), 0)

        mask = cv.inRange(blurred, lower_colour, upper_colour)
        cv.imshow("Mask", mask)

        if cv.countNonZero(mask) > spark_threshold:
            coords = cv.findNonZero(mask)
            if coords is not None:
                coords = np.squeeze(coords)
                print(f"Coordonnées détectées: {coords}")

                if coords.size > 0:
                    y = coords[:, 0]
                    x = coords[:, 1]

                    # Vérifier la dimension des coordonnées avant filtrage
                    print(f"Dimensions de x avant filtrage: {x.shape}, Dimensions de y avant filtrage: {y.shape}")

                    # Filtrer les points dans la zone grise (cadre)
                    gray_frame_filter = (x < gray_frame_top_left[0]) | (x > gray_frame_bottom_right[0]) | \
                                        (y < gray_frame_top_left[1]) | (y > gray_frame_bottom_right[1])
                    x = x[gray_frame_filter]
                    y = y[gray_frame_filter]

                    # Vérifier la dimension des coordonnées après filtrage
                    print(f"Dimensions de x après filtrage: {x.shape}, Dimensions de y après filtrage: {y.shape}")
                    

                    # Vérifier la dimension des coordonnées
                    if x.size > 0 and y.size > 0:
                        for i in range(len(x)):
                            cv.circle(frame, (y[i], x[i]), 2, (255, 0, 0), -1)

                        cv.imshow("Points détectés", frame)

                        # Régression linéaire avec NumPy
                        slope, intercept = np.polyfit(y, x, 1)
                        print(f"Slope: {slope}, Intercept: {intercept}")

                        h, w = frame.shape[:2]
                        x_vals = np.array([0, w])
                        y_vals = slope * x_vals + intercept
                        x_vals = np.clip(x_vals, 0, h)
                        y_vals = np.clip(y_vals, 0, w)

                        frame_with_line = frame.copy()
                        cv.line(frame_with_line,
                                (int(y_vals[0]), int(x_vals[0])),
                                (int(y_vals[1]), int(x_vals[1])),
                                (0, 255, 0), 2)

                        cv.imshow("Trajectoire détectée", frame_with_line)
                        frame_index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
                        cv.imwrite(f"spark_frame_{frame_index}.jpg", frame_with_line)
                        print(f"Frame {frame_index}: Slope={slope:.2f}, Intercept={intercept:.2f}")

                    else:
                        print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Aucun point détecté après filtrage")
                else:
                    print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Aucun point détecté après compression")
            else:
                print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Aucun point détecté")

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

video_path = r"C:\Users\nerop\Videos\Sparky.mp4"
lower_colour = np.array([130, 35, 30])
upper_colour = np.array([180, 255, 255])
spark_threshold = 8500

detect_and_fit_numpy(video_path, lower_colour, upper_colour, spark_threshold)
