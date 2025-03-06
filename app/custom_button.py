from PyQt6.QtWidgets import QPushButton

class CustomButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("background-color: #3498db; color: white; padding: 10px; border-radius: 5px;")
