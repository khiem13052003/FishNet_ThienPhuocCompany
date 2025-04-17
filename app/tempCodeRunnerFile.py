import sys
import os
from PyQt6.QtWidgets import QApplication
from view.main_window import MainWindow

current_dir= os.path.dirname(os.path.abspath(__file__))
parent_dir= os.path.dirname(current_dir)
sys.path.extend([parent_dir,current_dir])
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
if __name__ == '__main__':
    main()
    

