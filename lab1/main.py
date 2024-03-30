import os.path

import numpy as np
import cv2
import glob

CELLS_IN_ROW, CELLS_IN_COLUMN = 15, 11
NUMBER_POINTS_ROW, NUMBER_POINTS_COLUMN = CELLS_IN_ROW - 1, CELLS_IN_COLUMN - 1

# Termination criteria: stop the algorithm iteration if specified number of iterations or accuracy is reached.
# Number of iterations and accuracy to stop specified as second and third parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (x,y,z), like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_points = np.zeros((NUMBER_POINTS_ROW * NUMBER_POINTS_COLUMN, 3), np.float32)
object_points[:, :2] = np.mgrid[0:NUMBER_POINTS_ROW, 0:NUMBER_POINTS_COLUMN].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
points_3D = []  # 3D points in real world space
points_2D = []  # 2D points in image plane.

# Images to calibrate
images = glob.glob('data/calibrate/*.jpg')

# ----------- Points detection
for frame_name in images:
    img = cv2.imread(frame_name)  # Read the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # Find the chess board corners on grayscale image
    chessboard_found, corners = cv2.findChessboardCorners(gray_img, (NUMBER_POINTS_ROW, NUMBER_POINTS_COLUMN))

    # If chessboard found, add object points, image points (after refining them)
    if chessboard_found:
        points_3D.append(object_points)

        corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)  # refining img pts
        points_2D.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (NUMBER_POINTS_ROW, NUMBER_POINTS_COLUMN), corners2, chessboard_found)

        # Show, write to file, whatever
        # cv2.imshow('img', img)
        cv2.imwrite(f'data/results/with_patterns/{os.path.basename(frame_name)}', img)
        cv2.waitKey(500)

# ----------- Camera calibration
(
    _, camera_matrix,
    distortion_coefficient,
    rotation_vectors,
    translation_vectors
) = cv2.calibrateCamera(points_3D, points_2D, gray_img.shape[::-1], None, None)
np.savetxt(f'data/results/camera_matrix.txt', camera_matrix)
np.savetxt(f'data/results/distortion_coefficient.txt', distortion_coefficient)

# ----------- Camera refining
undistort_img_name = images[0]  # take one of the images
undistort_img = cv2.imread(undistort_img_name)
height, width = undistort_img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, distortion_coefficient, (width, height), 1, (width, height)
)  # alpha==1, so pixels are retained with extra black images
x, y, width, height = roi

# ----------- Image undistortion
undistorted_image = cv2.undistort(
    undistort_img, camera_matrix, distortion_coefficient, None, new_camera_matrix
)[y:y + height, x:x + width]  # Crop the image

cv2.imwrite(f'data/results/undistorted/{os.path.basename(undistort_img_name)}', undistorted_image)

# ----------- Calculating re-projection error

mean_error = 0
for i in range(len(points_3D)):
    points_2D_2, _ = cv2.projectPoints(points_3D[i], rotation_vectors[i], translation_vectors[i], camera_matrix,
                                       distortion_coefficient)
    error = cv2.norm(points_2D[i], points_2D_2, cv2.NORM_L2) / len(points_2D_2)
    mean_error += error

print("Re-projection error: ", mean_error / len(points_3D))

cv2.destroyAllWindows()
