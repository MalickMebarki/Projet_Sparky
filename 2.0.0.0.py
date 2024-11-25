import cv2 as cv
import numpy as np

def spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold):
    """
    Generator that yields frames containing pink sparks.

    Parameters:
    - video_path: Path to the input video.
    - lower_pink: Lower HSV bound for pink color.
    - upper_pink: Upper HSV bound for pink color.
    - spark_threshold: Minimum number of pink pixels to consider a frame as containing sparks.

    Yields:
    - Frames containing pink sparks (as NumPy arrays).
    """
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create a mask for pinkish colors
        mask = cv.inRange(hsv, lower_colour, upper_colour)

        # Check if the mask has significant "spark" pixels
        if cv.countNonZero(mask) > spark_threshold:
            yield frame

    cap.release()

# Define the relaxed pinkish color range in HSV
lower_colour = np.array([130, 30, 30])  # Relaxed lower bound
upper_colour = np.array([180, 255, 255])  # Relaxed upper bound
spark_threshold = 8500  # Minimum number of pink pixels to qualify as a spark

# Path to the video file
video_path = "sparky.mp4"

# Display the yielded frames with sparks one by one
for i, spark_frame in enumerate(spark_frame_generator(video_path, lower_colour, upper_colour, spark_threshold)):
    # Save the frame as an image file
    cv.imwrite(f"spark_frame_{i}.jpg", spark_frame)

    # Display the frame (optional)
    cv.imshow("Spark Frame", spark_frame)

    # Press 'q' to quit early
    if cv.waitKey(30) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()



