# import the require packages.
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap, QIcon, QImage, QIntValidator
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QEvent, QObject, QTime, QDateTime, QTimer
from .ROI_page import MainApp, ROIDialog
import numpy as np
import sys
import os
from PyQt6.QtWidgets import QFileDialog  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from module.deployModel import GUIProcessor


class CaptureWebcamFramesWorker(QThread):
    # Signal emitted when a new image or a new frame is ready.
    ImageUpdated = pyqtSignal(QImage)

    def __init__(self) -> None:
        super(CaptureWebcamFramesWorker, self).__init__()
        # Declare and initialize instance variables.
        self.__thread_active = True
        self.fps = 0
        self.__thread_pause = False

    def run(self) -> None:
        # Capture video from webcam
        cap = cv2.VideoCapture(0)
        # Get default video FPS.
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {self.fps}")
        
        # If video capturing has been initialized already
        if cap.isOpened():
            # While the thread is active.
            while self.__thread_active:
                if not self.__thread_pause:
                    # Grabs, decodes and returns the next video frame.
                    ret, frame = cap.read()
                    # If frame is read correctly.
                    if ret:
                        # Get the frame height, width and channels.
                        height, width, channels = frame.shape
                        # Calculate the number of bytes per line.
                        bytes_per_line = width * channels
                        # Convert image from BGR to RGB
                        cv_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert the image to Qt format.
                        qt_rgb_image = QImage(cv_rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                        # Scale the image
                        qt_rgb_image_scaled = qt_rgb_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                        # Emit this signal to notify that a new image or frame is available.
                        self.ImageUpdated.emit(qt_rgb_image_scaled)
                    else:
                        break
        # When everything done, release the video capture object.
        cap.release()
        # Tells the thread's event loop to exit with return code 0 (success).
        self.quit()

    def stop(self) -> None:
        self.__thread_active = False

    def pause(self) -> None:
        self.__thread_pause = True

    def unpause(self) -> None:
        self.__thread_pause = False


class ControlPanel(QWidget):
    def __init__(self, camera_page):
        super(ControlPanel, self).__init__()
        self.camera_page = camera_page
        # Thêm các biến ROI vào __init__
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        # Right side - Controls
        right_layout = QVBoxLayout()
        # ComboBox
        self.product_label = QLabel("Chọn sản phẩm bạn muốn")
        self.product_label.setStyleSheet("QLabel{color: white; font-size: 15px;}")
        self.product_type = QComboBox()
        self.product_type.addItems(["Lưới bao hoa", "Lưới đánh cá"])

        # Distance of 2 nodes
        self.distance_node_label = QLabel("Khoảng cách giữa 2 nút ngang (cm)")
        self.distance_node_label.setStyleSheet("QLabel{color: white; font-size: 15px;}")
        self.distance_node_edit = QLineEdit()
        self.distance_node_edit.setPlaceholderText("Nhập số cm")
        self.distance_node_edit.setValidator(QIntValidator())

        # Delay Time
        self.gap_time_label = QLabel("Khoảng thời gian giữa 2 lần ")
        self.gap_time_label.setStyleSheet("QLabel{color: white; font-size: 15px;}")
        self.gap_time_edit = QTimeEdit()
        self.gap_time_edit.setTime(QTime(0, 0, 5))
        self.gap_time_edit.setDisplayFormat("HH:mm:ss")

        # Amount of nodes/1 row
        self.total_node_label = QLabel("Tổng số nút trên 1 hàng")
        self.total_node_label.setStyleSheet("QLabel{color: white; font-size: 15px;}")
        self.total_node_edit = QLineEdit()
        self.total_node_edit.setPlaceholderText("Nhập số nút")
        self.total_node_edit.setValidator(QIntValidator())

        # Distance between 2 nodes in a column
        self.distance_node_column_label = QLabel("Loại lưới (khoảng cách giữa 2 nút dọc) (inch)")
        self.distance_node_column_label.setStyleSheet("QLabel{color: white; font-size: 15px;}")
        self.distance_node_column_edit = QLineEdit()
        self.distance_node_column_edit.setPlaceholderText("Nhập khoảng cách (inch)")
        self.distance_node_column_edit.setValidator(QIntValidator())
    
        # ROI custom
        self.roi_custom_button = QPushButton("Chỉnh vùng ROI")
        self.roi_custom_button.setStyleSheet("QPushButton{color: white; font-size: 15px;}")
        self.roi_custom_button.clicked.connect(self.captureFrame)

        self.submit_button= QPushButton("Xác Nhận")
        self.submit_button.setStyleSheet("QPushButton{color: white; font-size: 15px;}")
        
        
        self.delete_button= QPushButton('Hủy thông tin')
        self.delete_button.setStyleSheet("QPushButton{color: white; font-size: 15px;}")
       
        self.result_text= QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("QTextEdit{color: white; font-size: 15px;}")

        self.save_path_edit= QLineEdit()
        self.save_path_edit.setPlaceholderText("Nhập đường dẫn lưu file")
 
        self.browse_button=QPushButton("📂")
        self.browse_button.setStyleSheet("QPushButton{font-size: 15px; background-color: #FFFFFF;}")
        self.browse_button.clicked.connect(self.browse_file)
       
        # StyleSheet
        self.product_type.setStyleSheet("""
            QComboBox{
                font-size: 13px;
                padding: 10px;
                border: 2px solid #FFFFFF;
                border-radius: 8px;
                background-color: #424242; 
                color: white;        
            }          
            QComboBox::drop-down{
                width: 30px;
                border-left: 2px solid #0078D7;            
            }
            QComboBox::down-arrow{
                image: url(app/assets/icons/down.png);
                width: 12px;
                height: 12px;            
            }
            QComboBox QAbstractItemView{
                font-size: 15px;
                background-color: #424242;
                color: white;
                border: 1px solid #FFFFFF;
                selection-background-color: #0078D7;
                selection-color: #000000;                
            }
        """)

        self.distance_node_edit.setStyleSheet("""
            QLineEdit {
                color: white;
                background-color: #424242;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
                font-size: 15px;
            }
        """)
        
        self.gap_time_edit.setStyleSheet("""
            QTimeEdit{
                color: white;
                background-color: #424242;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
                font-size: 15px;
            }
            QTimeEdit::up-button, QTimeEdit::down-button {
                width: 30px;
                height: 17px;
                border: 1px solid #666;
                background-color: #555;
            }
            QTimeEdit::up-button:hover, QTimeEdit::down-button:hover {
                background-color: #666;
            }
            QTimeEdit::up-arrow {
                image: url(app/assets/icons/angle-small-up.png);
                width: 12px;
                height: 15px;
            }
            QTimeEdit::down-arrow {
                image: url(app/assets/icons/angle-small-down.png);
                width: 12px;
                height: 12px;
            }
        """)
        
        self.total_node_edit.setStyleSheet("""
            QLineEdit {
                color: white;
                background-color: #424242;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
                font-size: 15px;
            }
        """)

        self.distance_node_column_edit.setStyleSheet("""
            QLineEdit {
                color: white;
                background-color: #424242;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
                font-size: 15px;
            }
        """)
        self.roi_custom_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 6px 20px;
                border: none;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)

        self.submit_button.setStyleSheet("""
             QPushButton {
                background-color: #00CC00;
                color: white;
                padding: 6px 20px;
                border: none;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #009900;
            }
        """)

        self.delete_button.setStyleSheet("""
             QPushButton {
                background-color:  #CC0000;
                color: white;
                padding: 6px 20px;
                border: none;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color:  #990000;
            }
        """)

        self.result_text.setStyleSheet("""
             QTextEdit{
                color: white;
                background-color: #8a8a8a;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
                font-size: 16px;
                height: 100px;
           }
        """)
        self.save_path_edit.setStyleSheet("""
              QLineEdit{
                color: white;
                background-color: #424242;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        
        path_layout=QHBoxLayout()
        path_layout.addWidget(self.save_path_edit)
        path_layout.addWidget(self.browse_button)
        
        # Add widgets to layout
        right_layout.addWidget(self.product_label)
        right_layout.addWidget(self.product_type)
        right_layout.addWidget(self.distance_node_label)
        right_layout.addWidget(self.distance_node_edit)
        right_layout.addWidget(self.gap_time_label)
        right_layout.addWidget(self.gap_time_edit)
        right_layout.addWidget(self.total_node_label)
        right_layout.addWidget(self.total_node_edit)
        right_layout.addWidget(self.distance_node_column_label)
        right_layout.addWidget(self.distance_node_column_edit)
        right_layout.addWidget(self.roi_custom_button)
        right_layout.addWidget(self.submit_button)
        right_layout.addWidget(self.delete_button)
        right_layout.addWidget(self.result_text)
        right_layout.addLayout(path_layout)
        right_layout.addStretch()

        self.setLayout(right_layout)
    def save_pic(self):
        self.camera_page.handle_save_path_change(self)
    
    def browse_file(self):
        try:
            folder_path_select = self.camera_page.select_folder()  # CameraPage xử lý việc chọn thư mục
            if folder_path_select:
                self.save_path_edit.setText(folder_path_select)  # ControlPanel cập nhật UI của nó
        except Exception as e:
            print(f"Lỗi khi chọn thư mục: {str(e)}")
        
    def captureFrame(self):
        # Pause camera thread
        if self.camera_page.camera_thread.isRunning():
            self.camera_page.camera_thread.pause()
        
        # Lấy frame gốc từ CameraPage
        original_frame = self.camera_page.original_frame

        
        if original_frame and not original_frame.isNull():
            # Lưu frame gốc
            original_frame.save("captured_frame.png", "PNG")
            print("Frame gốc đã được lưu dưới dạng captured_frame.png")
            
            # Mở dialog ROI
            dialog = ROIDialog("captured_frame.png", self)
            
            if dialog.exec() == 1:  # Nếu người dùng nhấn OK
                roi_points = dialog.roi_points
                if len(roi_points) >= 4:
                    # print("Đã cập nhật các điểm ROI:", [(p.x(), p.y()) for p in roi_points])
                    # Lưu tọa độ vào biến instance
                    self.x_start = int(roi_points[0].x())
                    self.x_end = int(roi_points[1].x())
                    self.y_start = int(roi_points[0].y())
                    self.y_end = int(roi_points[3].y())
                    # print("Saved ROI in ControlPanel:", 
                    #       self.x_start, self.x_end, self.y_start, self.y_end)
            
            # Resume camera thread
            self.camera_page.camera_thread.unpause()
        else:
            print("Lỗi: Không thể chụp được frame - không có frame sẵn có")
    

class CameraPage(QMainWindow):
    def __init__(self, main_app) -> None:
        super(CameraPage, self).__init__()
        self.MainApp = main_app
        self.camera = QLabel(self)
        self.camera_state = "Normal"
        self.original_frame = None
        self.setup_datetime_timer()
        self.gui_processor= GUIProcessor()
        self.is_processing = False
        self.control_panel = ControlPanel(self)
        self.control_panel.camera_page = self
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left side - Camera
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Create camera label
        self.camera = QLabel()
        self.camera.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera.setScaledContents(True)
        self.camera.installEventFilter(self)
        self.camera.setObjectName("Camera")
        
        # Create scroll area for camera
        self.camera_scroll = QScrollArea()
        self.camera_scroll.setWidget(self.camera)
        self.camera_scroll.setWidgetResizable(True)
        
        now= QDateTime.currentDateTime()
        self.date_time_label= QLabel(f"Time: {now.toString('dd/MM/yyyy hh:mm:ss')}")
        self.date_time_label.setStyleSheet("QLabel { color: white; font-size: 20px; }")
        
        left_layout.addWidget(self.camera_scroll)
        left_layout.addWidget(self.date_time_label)
        left_widget.setLayout(left_layout)
        
        # Create ControlPanel instance - truyền self trực tiếp
        
        self.control_panel.submit_button.clicked.connect(self.handle_submit)
        self.control_panel.delete_button.clicked.connect(self.handle_delete)
        # Add widgets to main layout
        main_layout.addWidget(left_widget, stretch=2)  # Camera takes 2/3 of width
        main_layout.addWidget(self.control_panel, stretch=1)  # Controls take 1/3 of width
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Setup window properties
        self.setMinimumSize(800, 600)
        self.showMaximized()
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.setWindowTitle("Webcam Viewer")

        # Create and start camera thread
        self.camera_thread = CaptureWebcamFramesWorker()
        self.camera_thread.ImageUpdated.connect(self.show_camera)
        self.camera_thread.start()

        # Khởi tạo gui_processor khi submit form
        # self.gui_processor = None
        
        # Kết nối nút submit với hàm xử lý
        self.control_panel.submit_button.clicked.connect(self.handle_submit)

    def setup_datetime_timer(self):
        # Tạo timer để cập nhật datetime
        self.datetime_timer = QTimer()
        self.datetime_timer.timeout.connect(self.update_datetime)
        self.datetime_timer.start(1000)  # Cập nhật mỗi 1000ms (1 giây)
    
    def update_datetime(self):
        self.now = QDateTime.currentDateTime()
        self.date_time_label.setText(f"Time: {self.now.toString('dd/MM/yyyy hh:mm:ss')}")
    
    def show_camera(self, frame: QImage):
        try:
            self.original_frame = frame.copy()
            
            if self.is_processing:
                # Chuyển QImage sang numpy array
                width = frame.width()
                height = frame.height()
                ptr = frame.bits()
                ptr.setsize(height * width * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
                # Gọi process_frame
                has_error, processed_frame = self.gui_processor.process_frame(arr)
                if has_error and processed_frame is not None:
                    self.control_panel.result_text.setText(
                        f"Số lượng lỗi cho đến hiện tại {self.gui_processor.error_count}"
                    )
                # Chuyển đổi từ BGR sang RGB trước khi tạo QImage
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = processed_frame_rgb.shape
                    bytes_per_line = 3 * width
                    processed_frame_rgb = np.ascontiguousarray(processed_frame_rgb)
                    processed_qimg = QImage(
                        processed_frame_rgb.data,
                        width, 
                        height, 
                        bytes_per_line, 
                        QImage.Format.Format_RGB888
                    )
                    self.camera.setPixmap(QPixmap.fromImage(processed_qimg))
                    return
                # Hiển thị frame gốc nếu không xử lý
            if not self.is_processing:
                self.camera.setPixmap(QPixmap.fromImage(frame))
        except Exception as e:
            print(f"Lỗi trong show_camera: {str(e)}")
            # self.camera.setPixmap(QPixmap.fromImage(frame))        
    
    def select_folder(self):
        """Mở hộp thoại chọn thư mục và trả về đường dẫn"""
        # folder_path = QFileDialog.getExistingDirectory(None, "Chọn thư mục")
        folder_path=QFileDialog(self)
        folder_path.setFileMode(QFileDialog.FileMode.Directory)
        folder_path.setOption(QFileDialog.Option.DontUseNativeDialog)
        if folder_path.exec():
            folder_path_select=folder_path.selectedFiles()[0]
        if folder_path_select:
            self.gui_processor.set_save_path(folder_path_select)  # CameraPage cập nhật processor
        return folder_path_select

    
    def cleanup_datetime(self):
        if self.datetime_timer.isActive():
            self.datetime_timer.stop()
    
    def handle_save_path_change(self):
        """Xu ly khi nguoi dung cho duong dan luu anh"""
        save_path= self.control_panel.set_path_edit.text()
        if save_path and os.path.exists(save_path):
            self.gui_processor.set_save_path(save_path)
        else:
            self.control_panel.result_text.setText("❌ Đường dẫn không hợp lệ")
    
    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if source.objectName() == 'Camera':
                if self.camera_state == "Normal":
                    self.camera_state = "Maximized"
                    # Hide the right panel
                    self.centralWidget().layout().itemAt(1).widget().hide()
                else:
                    self.camera_state = "Normal"
                    # Show the right panel
                    self.centralWidget().layout().itemAt(1).widget().show()
                return True
        return super(CameraPage, self).eventFilter(source, event)

    def closeEvent(self, event) -> None:
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.quit()
            self.cleanup_datetime()
            self.gui_processor.clean_up()
        event.accept()

    def handle_submit(self):
        """
        Xử lý khi người dùng nhấn nút submit
        """
        save_path = self.control_panel.save_path_edit.text()
        if not save_path:
            self.control_panel.result_text.setText("Vui lòng chọn đường dẫn lưu ảnh!")
            return
        if not os.path.exists(save_path):
            self.control_panel.result_text.setText("❌ Đường dẫn không tồn tại")
            return
        self.gui_processor.set_save_path(save_path)
        # Kiểm tra tọa độ ROI
        if None in [self.control_panel.x_start, 
                    self.control_panel.x_end,
                    self.control_panel.y_start, 
                    self.control_panel.y_end]:
            self.control_panel.result_text.setText("Vui lòng chọn vùng ROI trước!")
            return

        # Gửi tọa độ ROI sang GUIProcessor
        self.gui_processor.set_roi_coordinates(
            self.control_panel.x_start,
            self.control_panel.x_end,
            self.control_panel.y_start,
            self.control_panel.y_end
        )
        
        print("ROI coordinates sent to GUIProcessor:", 
              self.gui_processor.x_start,
              self.gui_processor.x_end,
              self.gui_processor.y_start,
              self.gui_processor.y_end)
        
        self.is_processing = True
        self.gui_processor.handle_submit(self.control_panel)

    def handle_delete(self):
        """
        xử lý khi người dùng nhấn nút Hủy
        """
        self.is_processing= False
        self.gui_processor.handle_delete(self.control_panel)