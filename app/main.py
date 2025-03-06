from PyQt6.QtWidgets import QApplication
from views.login_page import LoginPage
import sys

def main():
    # Khởi tạo ứng dụng
    app = QApplication(sys.argv)
    
    # Tạo và hiển thị cửa sổ đăng nhập
    login_window = LoginPage()
    login_window.show() 
    
    # Chạy ứng dụng
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

