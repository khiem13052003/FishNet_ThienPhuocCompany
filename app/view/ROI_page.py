# This would go in your ROI_page.py file
import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QWidget, QGridLayout,QGroupBox)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIntValidator
from PyQt6.QtCore import Qt, QPoint

class ROIDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super(ROIDialog, self).__init__(parent)
        self.setWindowTitle("Chỉnh Vùng ROI")
        self.setMinimumSize(900, 700)
        
        # Tải ảnh đã chụp
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Lỗi: Không thể tải ảnh từ {image_path}")
            return
            
        # Lấy kích thước thật của ảnh
        self.image_height, self.image_width = self.original_image.shape[:2]
        
        # Khởi tạo các điểm ROI - mặc định ở góc của hình chữ nhật ở trung tâm
        center_x, center_y = self.image_width // 2, self.image_height // 2
        offset_x, offset_y = self.image_width // 4, self.image_height // 4
        
        # Các điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
        self.roi_points = [
            QPoint(center_x - offset_x, center_y - offset_y),  # 0: trên-trái
            QPoint(center_x + offset_x, center_y - offset_y),  # 1: trên-phải
            QPoint(center_x + offset_x, center_y + offset_y),  # 2: dưới-phải
            QPoint(center_x - offset_x, center_y + offset_y)   # 3: dưới-trái
        ]
        
        # Biến theo dõi điểm được chọn và kích thước điểm
        self.selected_point = None
        self.point_radius = 8
        
        # Tạo QImage với ROI
        self.q_image = self.update_image_with_roi()
        
        # Tạo layout
        main_layout = QVBoxLayout()
        
        # Widget hiển thị ảnh - chiếm phần lớn không gian
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        
        # Label hiển thị ảnh
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setPixmap(QPixmap.fromImage(self.q_image))
        self.image_label.setScaledContents(True)
        
        # Bật theo dõi chuột
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        
        # Thêm ảnh vào layout với tỷ lệ co giãn lớn
        image_layout.addWidget(self.image_label)
        image_widget.setLayout(image_layout)
        main_layout.addWidget(image_widget, stretch=4)  # Chiếm 80% không gian
        
        # Widget chứa các ô nhập tọa độ - ở dưới ảnh
        coord_widget = QWidget()
        coord_layout = QHBoxLayout()
        
        # Tạo 4 ô nhập liệu cho 4 điểm
        self.point_inputs = []
        labels = ["Điểm 1 (Trên-Trái)", "Điểm 2 (Trên-Phải)", 
                 "Điểm 3 (Dưới-Phải)", "Điểm 4 (Dưới-Trái)"]
        
        for i in range(4):
            point_group = QGroupBox(labels[i])
            point_layout = QVBoxLayout()
            
            # Tạo một ô nhập liệu hiển thị cả X và Y
            coord_input = QLineEdit()
            coord_input.setText(f"{self.roi_points[i].x()}, {self.roi_points[i].y()}")
            
            # Kết nối tín hiệu
            coord_input.textChanged.connect(lambda text, idx=i: self.update_point_from_text(idx, text))
            
            # Lưu trữ tham chiếu
            self.point_inputs.append(coord_input)
            
            # Thêm vào layout
            point_layout.addWidget(coord_input)
            point_group.setLayout(point_layout)
            coord_layout.addWidget(point_group)
        
        coord_widget.setLayout(coord_layout)
        main_layout.addWidget(coord_widget, stretch=1)  # Chiếm 20% không gian
        
        # Buttons
        button_layout = QHBoxLayout()
        
        update_button = QPushButton("Cập Nhật ROI")
        update_button.clicked.connect(self.update_roi)
        
        cancel_button = QPushButton("Hủy")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(update_button)
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Áp dụng stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #424242;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QLineEdit {
                color: white;
                background-color: #555;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #666;
                border-radius: 4px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 8px 24px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
    
    def update_image_with_roi(self):
        # Tạo một bản sao của ảnh để vẽ ROI
        roi_image = self.original_image.copy()
        
        # Vẽ đa giác ROI (hình chữ nhật)
        points = np.array([[p.x(), p.y()] for p in self.roi_points], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(roi_image, [points], True, (0, 255, 0), 2)
        
        # Vẽ các điểm góc
        for i, point in enumerate(self.roi_points):
            color = (255, 0, 0) if i == self.selected_point else (0, 0, 255)
            cv2.circle(roi_image, (point.x(), point.y()), self.point_radius, color, -1)
            # Thêm nhãn cho mỗi điểm
            cv2.putText(roi_image, str(i+1), (point.x() - 5, point.y() - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Chuyển sang RGB cho Qt
        rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        # Chuyển thành QImage
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        return q_image
    
    def mouse_press_event(self, event):
        # Chuyển đổi vị trí chuột tương đối
        pos = event.position()
        
        # Tính tỷ lệ co giãn
        label_size = self.image_label.size()
        scale_x = self.image_width / label_size.width()
        scale_y = self.image_height / label_size.height()
        
        # Vị trí thực tế trên ảnh
        real_x = int(pos.x() * scale_x)
        real_y = int(pos.y() * scale_y)
        
        # Kiểm tra xem nhấp chuột có gần điểm nào không
        for i, point in enumerate(self.roi_points):
            dx = real_x - point.x()
            dy = real_y - point.y()
            if dx*dx + dy*dy < self.point_radius*self.point_radius:
                self.selected_point = i
                break
    
    def mouse_release_event(self, event):
        self.selected_point = None
    
    def mouse_move_event(self, event):
        if self.selected_point is not None:
            # Chuyển đổi vị trí chuột
            pos = event.position()
            
            # Tính tỷ lệ co giãn
            label_size = self.image_label.size()
            scale_x = self.image_width / label_size.width() 
            scale_y = self.image_height / label_size.height()
            
            # Vị trí thực tế trên ảnh
            real_x = int(pos.x() * scale_x)
            real_y = int(pos.y() * scale_y)
            
            # Giới hạn trong kích thước ảnh
            real_x = max(0, min(real_x, self.image_width - 1))
            real_y = max(0, min(real_y, self.image_height - 1))
            
            # Cập nhật các điểm để giữ nguyên hình chữ nhật
            self.update_rectangle_points(self.selected_point, real_x, real_y)
            
            # Cập nhật tất cả các ô nhập liệu
            self.update_all_point_inputs()
            
            # Cập nhật ảnh
            self.q_image = self.update_image_with_roi()
            self.image_label.setPixmap(QPixmap.fromImage(self.q_image))
    
    def update_rectangle_points(self, point_idx, new_x, new_y):
        """
        Cập nhật các điểm để luôn giữ nguyên hình chữ nhật
        point_idx: chỉ số của điểm đang di chuyển
        new_x, new_y: tọa độ mới của điểm đang di chuyển
        """
        # Đảm bảo các điểm giữ nguyên hình chữ nhật
        if point_idx == 0:  # trên-trái
            # Cập nhật điểm 1 (trên-trái)
            self.roi_points[0] = QPoint(new_x, new_y)
            # Cập nhật điểm 1 (trên-phải) - giữ nguyên x, cập nhật y
            self.roi_points[1] = QPoint(self.roi_points[1].x(), new_y)
            # Cập nhật điểm 4 (dưới-trái) - giữ nguyên y, cập nhật x
            self.roi_points[3] = QPoint(new_x, self.roi_points[3].y())
            
        elif point_idx == 1:  # trên-phải
            # Cập nhật điểm 2 (trên-phải)
            self.roi_points[1] = QPoint(new_x, new_y)
            # Cập nhật điểm 1 (trên-trái) - giữ nguyên x, cập nhật y
            self.roi_points[0] = QPoint(self.roi_points[0].x(), new_y)
            # Cập nhật điểm 3 (dưới-phải) - giữ nguyên y, cập nhật x
            self.roi_points[2] = QPoint(new_x, self.roi_points[2].y())
            
        elif point_idx == 2:  # dưới-phải
            # Cập nhật điểm 3 (dưới-phải)
            self.roi_points[2] = QPoint(new_x, new_y)
            # Cập nhật điểm 4 (dưới-trái) - giữ nguyên x, cập nhật y
            self.roi_points[3] = QPoint(self.roi_points[3].x(), new_y)
            # Cập nhật điểm 2 (trên-phải) - giữ nguyên y, cập nhật x
            self.roi_points[1] = QPoint(new_x, self.roi_points[1].y())
            
        elif point_idx == 3:  # dưới-trái
            # Cập nhật điểm 4 (dưới-trái)
            self.roi_points[3] = QPoint(new_x, new_y)
            # Cập nhật điểm 3 (dưới-phải) - giữ nguyên x, cập nhật y
            self.roi_points[2] = QPoint(self.roi_points[2].x(), new_y)
            # Cập nhật điểm 1 (trên-trái) - giữ nguyên y, cập nhật x
            self.roi_points[0] = QPoint(new_x,self.roi_points[0].y())
    
    def update_point_from_text(self, idx, text):
        """
        Cập nhật điểm từ dữ liệu trong ô nhập liệu
        """
        # try:
        # Tách chuỗi thành x và y
        x_str, y_str = text.split(',')
        x = int(x_str.strip())
        y = int(y_str.strip())
        
        # Giới hạn trong kích thước ảnh
        x = max(0, min(x, self.image_width - 1))
        y = max(0, min(y, self.image_height - 1))
        
        # Cập nhật điểm
        self.update_rectangle_points(idx, x, y)
        
        # Cập nhật tất cả các ô nhập liệu để đảm bảo tính nhất quán
        self.update_all_point_inputs()
        
        # Cập nhật ảnh
        self.q_image = self.update_image_with_roi()
        self.image_label.setPixmap(QPixmap.fromImage(self.q_image))
        # except ValueError:
        #     # Bỏ qua nếu định dạng không hợp lệ
        #     pass

    def update_all_point_inputs(self):
        """
        Cập nhật tất cả các ô nhập liệu từ các điểm ROI hiện tại
        """
        for i, point in enumerate(self.roi_points):
            # Ngắt kết nối tín hiệu để tránh đệ quy
            self.point_inputs[i].blockSignals(True)
            
            # Cập nhật văn bản
            self.point_inputs[i].setText(f"{point.x()}, {point.y()}")
            
            # Kết nối lại tín hiệu
            self.point_inputs[i].blockSignals(False)

    def update_roi(self):
        """
        Cập nhật ROI và chấp nhận hộp thoại
        """
        # Tạo danh sách các điểm ROI để trả về
        self.final_roi_points = [(p.x(), p.y()) for p in self.roi_points]
        
        # Chấp nhận hộp thoại (tương đương với việc nhấn OK)
        self.accept()

    def get_roi_points(self):
        """
        Trả về các điểm ROI đã chọn
        """
        if hasattr(self, 'final_roi_points'):
            return self.final_roi_points
        else:
            # Nếu người dùng chưa nhấn cập nhật, trả về các điểm hiện tại
            return [(p.x(), p.y()) for p in self.roi_points]

    def get_roi_image(self):
        """
        Trả về ảnh đã cắt theo ROI
        """
        # Lấy các điểm ROI
        points = np.array(self.get_roi_points(), dtype=np.int32)
        
        # Tạo mask từ các điểm ROI
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Cắt ảnh theo mask
        roi_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
        
        # Tính bounding box của ROI
        x, y, w, h = cv2.boundingRect(points)
        
        # Cắt ảnh theo bounding box
        cropped_roi = roi_image[y:y+h, x:x+w]
        
        return cropped_roi

    def get_perspective_transformed_roi(self):
        """
        Trả về ảnh đã được chỉnh sửa góc nhìn để ROI thành hình chữ nhật
        """
        # Lấy các điểm ROI
        src_points = np.array(self.get_roi_points(), dtype=np.float32)
        
        # Tính kích thước của ảnh sau khi biến đổi
        width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                        ((src_points[1][1] - src_points[0][1]) ** 2))
        width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                            ((src_points[2][1] - src_points[3][1]) ** 2))
        max_width = int(max(width_top, width_bottom))
        
        height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + 
                            ((src_points[3][1] - src_points[0][1]) ** 2))
        height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + 
                            ((src_points[2][1] - src_points[1][1]) ** 2))
        max_height = int(max(height_left, height_right))
        
        # Định nghĩa 4 điểm đích cho phép biến đổi góc nhìn
        dst_points = np.array([
            [0, 0],               # trên-trái
            [max_width - 1, 0],   # trên-phải
            [max_width - 1, max_height - 1],  # dưới-phải
            [0, max_height - 1]   # dưới-trái
        ], dtype=np.float32)
        
        # Tính ma trận biến đổi góc nhìn
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Áp dụng biến đổi góc nhìn
        transformed_image = cv2.warpPerspective(self.original_image, perspective_matrix, 
                                            (max_width, max_height))
        
        return transformed_image

class MainApp:
    @staticmethod
    def captureFrame(self):
        # This method will be called from your ControlPanel class
        # It captures the current frame and saves it to disk
        print("Capturing frame for ROI adjustment")
        
        # Access the current frame
        frame = self.camera_page.camera.pixmap()
        
        if frame and not frame.isNull():
            # Convert QPixmap to QImage and save it
            frame.save("captured_frame.png", "PNG")
            print("Frame captured and saved as captured_frame.png")
        else:
            print("Error: Could not capture frame - no frame available")