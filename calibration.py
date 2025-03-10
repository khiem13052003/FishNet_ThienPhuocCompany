import cv2
import numpy as np

def undistortImage(frame, calibration_file="monoCalibration.xml"):
    
    # Đọc tham số hiệu chuẩn từ file XML
    cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    cameraMatrix = cv_file.getNode('cameraMatrix').mat()
    distCoeffs = cv_file.getNode('distCoeffs').mat()
    cv_file.release()
    h, w = frame.shape[:2]
    # Hiệu chỉnh méo ảnh
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,(w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, new_camera_matrix, (w, h), 5)
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    return undistorted 



