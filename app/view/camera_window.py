import cv2
from PyQt6.QtWidgets import QMainWindow, QLabel, QMessageBox, QComboBox, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera View")
        self.setFixedSize(800, 600)
        
        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Nút để bắt đầu/dừng camera
        self.toggle_button = QPushButton("Bắt đầu Camera")
        self.toggle_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.toggle_button)
        
        # Label để hiển thị hình ảnh
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Thiết lập camera
        self.cap = None
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)
        self.camera_running = False

    def toggle_camera(self):
        if not self.camera_running:
            # Thử mở camera
            self.cap = cv2.VideoCapture(0)
            while True:
                ret, frame =self.cap.read()
                if not ret:
                    break
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Lỗi", "Không thể mở camera")
                return
                
                self.timer.start(30)  # Cập nhật mỗi 30ms
                self.camera_running = True
                self.toggle_button.setText("Dừng Camera")
            else:
                self.stop_camera()
                self.toggle_button.setText("Bắt đầu Camera")

        # def update_frame(self):
        #     ret, frame = self.cap.read()
        #     if ret:
        #         # Chuyển đổi frame từ BGR sang RGB
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         h, w, ch = frame.shape
        #         bytes_per_line = ch * w
                
        #         # Chuyển đổi frame thành QImage
        #         qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
        #         # Scale image để vừa với label
        #         scaled_image = qt_image.scaled(self.image_label.width(), 
        #                                      self.image_label.height(),
        #                                      Qt.AspectRatioMode.KeepAspectRatio)
                
        #         # Hiển thị frame
        #         self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

    
   
    
    def stop_camera(self):
        if self.camera_running:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
            self.camera_running = False
            self.image_label.clear()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()