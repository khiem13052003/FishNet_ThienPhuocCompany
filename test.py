import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Biến camera
        self.camera_active = False
        self.cap = None  
        self.timer = QTimer()  
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        self.setWindowTitle("Camera App")
        self.setGeometry(100, 100, 800, 600)

        # Nút mở camera
        self.btn_open_camera = QPushButton("Chỉnh ROI", self)
        self.btn_open_camera.clicked.connect(self.toggle_camera)

        # Nút chụp ảnh
        self.btn_capture = QPushButton("Chụp Ảnh", self)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_capture.setEnabled(False)

        # Nút chọn ảnh từ hệ thống
        self.btn_select_image = QPushButton("Chọn Ảnh", self)
        self.btn_select_image.clicked.connect(self.select_image)

        # Label hiển thị ảnh hoặc camera
        self.label_display = QLabel(self)
        self.label_display.setFixedSize(640, 480)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.btn_open_camera)
        layout.addWidget(self.btn_capture)
        layout.addWidget(self.btn_select_image)
        layout.addWidget(self.label_display)
        self.setLayout(layout)

    def toggle_camera(self):
        """Mở hoặc tắt camera"""
        if self.camera_active:
            self.camera_active = False
            self.timer.stop()
            self.cap.release()
            self.label_display.clear()
            self.btn_open_camera.setText("Chỉnh ROI")
            self.btn_capture.setEnabled(False)
        else:
            self.cap = cv2.VideoCapture(0)  
            if not self.cap.isOpened():
                print("Không thể mở camera!")
                return

            self.camera_active = True
            self.timer.start(30)  
            self.btn_open_camera.setText("Tắt Camera")
            self.btn_capture.setEnabled(True)

    def update_frame(self):
        """Cập nhật frame từ camera"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.label_display.setPixmap(QPixmap.fromImage(qimg))

    def capture_frame(self):
        """Chụp ảnh từ camera"""
        ret, frame = self.cap.read()
        if ret:
            file_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                cv2.imwrite(file_path, frame)
                print(f"Ảnh đã lưu tại: {file_path}")

    def select_image(self):
        """Chọn ảnh từ hệ thống"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.label_display.setPixmap(QPixmap(file_path))
            print(f"Đã chọn ảnh: {file_path}")

    def closeEvent(self, event):
        """Giải phóng camera khi đóng"""
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
