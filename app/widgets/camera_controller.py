from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QEvent
from PyQt6.QtGui import QImage
import cv2
from ultralytics import YOLO
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys
import os
        
class CaptureIpCameraFramesWorker(QThread):
    ImageUpdated = pyqtSignal(QImage)
    CountUpdated = pyqtSignal(int)
    RoiCountUpdated = pyqtSignal(int)

    def __init__(self, url, camera_id) -> None:
        super(CaptureIpCameraFramesWorker, self).__init__()
        self.url = url
        self.camera_id = camera_id
        self.__thread_active = True
        self.fps = 0
        self.__thread_pause = False
        self.modelPath = os.path.join(os.getcwd(),r'assets\model\bestyolo5.pt')
        self.model = YOLO(self.modelPath)
        self.roi_points = np.array([[300, 110], [900, 250], [900, 500], [130, 220]], np.int32)
        # Thêm các ngưỡng cho detection
        self.conf_threshold = 0.05  # Ngưỡng confidence
        self.iou_threshold = 0.4  # Ngưỡng IoU cho NMS

    def point_in_roi(self, point):
        return cv2.pointPolygonTest(self.roi_points, point, False) >= 0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        if cap.isOpened():
            while self.__thread_active:
                if not self.__thread_pause:
                    ret, frame = cap.read()
                    if ret:
                        # Thực hiện detection với các ngưỡng đã thiết lập
                        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                        
                        # Lọc các boxes có confidence cao và đã qua NMS
                        filtered_boxes = []
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                # Lấy confidence score
                                conf = float(box.conf)
                                # Chỉ xử lý các box có confidence cao hơn ngưỡng
                                if conf > self.conf_threshold:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    filtered_boxes.append({
                                        'box': (x1, y1, x2, y2),
                                        'conf': conf,
                                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                    })

                        # Cập nhật số lượng người (sau khi đã lọc)
                        person_count = len(filtered_boxes)
                        self.CountUpdated.emit(person_count)
                        
                        roi_count = 0
                        
                        # Vẽ các boxes đã được lọc
                        for box_info in filtered_boxes:
                            x1, y1, x2, y2 = box_info['box']
                            center_x, center_y = box_info['center']
                            conf = box_info['conf']
                            
                            if self.point_in_roi((center_x, center_y)):
                                roi_count += 1
                            
                            # Vẽ bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Vẽ điểm trung tâm
                            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            # Hiển thị confidence score
                            cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if self.camera_id == 1:
                            cv2.polylines(frame, [self.roi_points], True, (255, 0, 0), 2)
                            self.RoiCountUpdated.emit(roi_count)

                        height, width, channels = frame.shape
                        bytes_per_line = width * channels
                        cv_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        qt_rgb_image = QImage(cv_rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        qt_rgb_image_scaled = qt_rgb_image.scaled(1280, 720, Qt.AspectRatioMode.KeepAspectRatio)
                        self.ImageUpdated.emit(qt_rgb_image_scaled)
                    else:
                        break
        cap.release()
        self.quit()

class CameraEventHandler:
    @staticmethod
    def handle_double_click(window, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if source.objectName() == 'Camera_1':
                if window.list_of_cameras_state["Camera_1"] == "Normal":
                    window.QScrollArea_2.hide()
                    window.list_of_cameras_state["Camera_1"] = "Maximized"
                    window.count_label_1.show()
                    window.count_label_2.hide()
                    window.roi_count_label.show()
                else:
                    window.QScrollArea_2.show()
                    window.list_of_cameras_state["Camera_1"] = "Normal"
                    window.count_label_1.show()
                    window.count_label_2.show()
                    window.roi_count_label.show()
            elif source.objectName() == 'Camera_2':
                if window.list_of_cameras_state["Camera_2"] == "Normal":
                    window.QScrollArea_1.hide()
                    window.list_of_cameras_state["Camera_2"] = "Maximized"
                    window.count_label_2.show()
                    window.count_label_1.hide()
                    window.roi_count_label.hide()
                else:
                    window.QScrollArea_1.show()
                    window.list_of_cameras_state["Camera_2"] = "Normal"
                    window.count_label_2.show()
                    window.count_label_1.show()
                    window.roi_count_label.show()
            # else:
            #     return False
            #return True
        # return False

class CameraWorkerManager:
    def __init__(self, url_1, url_2):
        self.url_1 = url_1
        self.url_2 = url_2
        self.worker_1 = None
        self.worker_2 = None

    def setup_workers(self, ui_callbacks):
        """
        Khởi tạo và cấu hình camera workers
        ui_callbacks: dict chứa các callback functions từ UI
        """
        # Setup worker cho camera 1
        self.worker_1 = CaptureIpCameraFramesWorker(self.url_1, camera_id=1)
        self.worker_1.ImageUpdated.connect(ui_callbacks['show_camera1'])
        self.worker_1.CountUpdated.connect(ui_callbacks['update_count1'])
        self.worker_1.RoiCountUpdated.connect(ui_callbacks['update_roi_count'])

        # Setup worker cho camera 2
        self.worker_2 = CaptureIpCameraFramesWorker(self.url_2, camera_id=2)
        self.worker_2.ImageUpdated.connect(ui_callbacks['show_camera2'])
        self.worker_2.CountUpdated.connect(ui_callbacks['update_count2'])

        # Khởi động threads
        self.worker_1.start()
        self.worker_2.start()

    def stop_workers(self):
        """Dừng các camera workers"""
        if self.worker_1:
            self.worker_1.quit()
        if self.worker_2:
            self.worker_2.quit()


