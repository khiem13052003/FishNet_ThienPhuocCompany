from PyQt6.QtWidgets import (QMainWindow, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QMessageBox)
from PyQt6.QtCore import Qt
from .camera_page import CameraPage
from .ROI_page import MainApp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Đăng nhập")
        self.setFixedSize(400, 300)
        
        # Tạo widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Tạo layout dọc
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Tạo tiêu đề
        title_label = QLabel("ĐĂNG NHẬP HỆ THỐNG")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 20px;")
        layout.addWidget(title_label)
        
        # Tạo trường username
        self.username_label = QLabel("Tên đăng nhập:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Nhập tên đăng nhập")
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        
        # Tạo trường password
        self.password_label = QLabel("Mật khẩu:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Nhập mật khẩu")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        
        # Tạo nút đăng nhập
        self.login_button = QPushButton("Đăng nhập")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 16px;
                border-radius: 4px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)
        
        # Căn giữa các widget và thêm khoảng trống
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(50, 20, 50, 20)
        layout.setSpacing(10)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        self.main_app= MainApp
    
        # Đây là nơi bạn sẽ thêm logic xác thực đăng nhập
        if username == '1' and password == '1':
            # Mở cửa sổ camera
            # self.camera_window = CameraPage()
            self.camera_window= CameraPage(self.main_app)
            self.camera_window.show()
            # Ẩn cửa sổ đăng nhập
            self.hide()
        else:
            QMessageBox.warning(self, "Lỗi", "Tên đăng nhập hoặc mật khẩu không đúng!")

