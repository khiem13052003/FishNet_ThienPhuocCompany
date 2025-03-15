import cv2
import os
import datetime
import tkinter as tk
from tkinter import filedialog

# Hàm mở hộp thoại chọn thư mục
def select_folder():
    folder_selected = filedialog.askdirectory()  # Hộp thoại chọn thư mục
    if folder_selected:  # Nếu người dùng chọn thư mục
        save_image(folder_selected)

# Hàm lưu ảnh vào thư mục đã chọn
def save_image(base_folder):
    now = datetime.datetime.now()

    # Tạo thư mục theo ngày bên trong thư mục được chọn
    date_folder = now.strftime("%Y-%m-%d")  # Định dạng: YYYY-MM-DD
    full_folder_path = os.path.join(base_folder, date_folder)

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)

    # Tạo tên file ảnh theo giờ phút giây
    file_name = now.strftime("%H-%M-%S") + ".png"
    file_path = os.path.join(full_folder_path, file_name)

    # Giả sử ảnh cần lưu (thay thế bằng ảnh từ camera nếu cần)
    image = cv2.imread("example.jpg")  # Thay bằng frame từ camera nếu cần

    # Lưu ảnh
    cv2.imwrite(file_path, image)

    print(f"Ảnh đã lưu vào: {file_path}")

# Tạo giao diện với Tkinter
root = tk.Tk()
root.title("Chọn Thư Mục Lưu Ảnh")

# Nút chọn thư mục
btn_select = tk.Button(root, text="Chọn thư mục lưu ảnh", command=select_folder)
btn_select.pack(pady=20)

# Hiển thị cửa sổ
root.mainloop()
