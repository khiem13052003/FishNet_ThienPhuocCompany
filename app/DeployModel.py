import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import numpy as np

class CameraApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dual Camera System")
        
        # Cố định IP cho 2 camera
        self.camera_ip1 = "10.0.0.90"
        self.camera_ip2 = "10.0.0.91"
        
        # Khởi tạo thông tin đăng nhập
        self.username = tk.StringVar(value="hoatuoitt")
        self.password = tk.StringVar(value="Thienphuoc2025")
        
        # Biến để kiểm soát trạng thái video và các đối tượng capture
        self.is_running = False
        self.cap1 = None
        self.cap2 = None
        
        self.create_login_window()
        
    def create_login_window(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(main_frame, text="Camera Login", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Username
        ttk.Label(main_frame, text="Username:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.username).grid(row=1, column=1, sticky="ew", pady=5)
        
        # Password
        ttk.Label(main_frame, text="Password:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.password, show="*").grid(row=2, column=1, sticky="ew", pady=5)
        
        ttk.Button(main_frame, text="Connect", command=self.connect_cameras).grid(row=3, column=0, columnspan=2, pady=20)
        
        main_frame.columnconfigure(1, weight=1)
    
    def create_video_window(self):
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("Dual Camera Feed")
        self.video_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Panel bên trái hiển thị thống kê
        left_panel = ttk.Frame(self.video_window, relief="solid", borderwidth=1, padding=10)
        left_panel.grid(row=0, column=0, sticky='ns', padx=10, pady=10)
        
        ttk.Label(left_panel, text="Statistics", font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # Tạo label đếm số người cho mỗi camera
        self.person_count_label1 = self.create_count_frame(left_panel, "Camera 1")
        self.person_count_label2 = self.create_count_frame(left_panel, "Camera 2")
        
        ttk.Button(left_panel, text="Disconnect", command=self.disconnect_cameras).pack(pady=20)
        
        # Các frame hiển thị video
        self.video_frame1 = ttk.Label(self.video_window)
        self.video_frame1.grid(row=0, column=1, padx=10, pady=10)
        
        self.video_frame2 = ttk.Label(self.video_window)
        self.video_frame2.grid(row=0, column=2, padx=10, pady=10)
    
    def create_count_frame(self, parent, title):
        frame = ttk.Frame(parent)
        frame.pack(pady=10)
        ttk.Label(frame, text=f"{title} Count:", font=('Helvetica', 12)).pack()
        label = ttk.Label(frame, text="0", font=('Helvetica', 24, 'bold'))
        label.pack(pady=5)
        return label

    def connect_cameras(self):
        user = self.username.get().strip()
        pwd = self.password.get().strip()
        
        if not user or not pwd:
            messagebox.showerror("Error", "Please fill all fields")
            return

        # Tạo URL RTSP cho cả 2 camera
        rtsp_url1 = f"rtsp://{user}:{pwd}@{self.camera_ip1}:554/Streaming/Channels/102"
        rtsp_url2 = f"rtsp://{user}:{pwd}@{self.camera_ip2}:554/Streaming/Channels/102"

        # Khởi tạo capture cho 2 camera
        self.cap1 = cv2.VideoCapture(rtsp_url1, cv2.CAP_FFMPEG)
        self.cap2 = cv2.VideoCapture(rtsp_url2, cv2.CAP_FFMPEG)
        
        # Thử kết nối lại nếu chưa mở được camera
        retry = 10  # Thử trong vòng 10 giây
        while retry > 0:
            if self.cap1.isOpened() and self.cap2.isOpened():
                break
            time.sleep(1)
            retry -= 1
            if not self.cap1.isOpened():
                self.cap1 = cv2.VideoCapture(rtsp_url1, cv2.CAP_FFMPEG)
            if not self.cap2.isOpened():
                self.cap2 = cv2.VideoCapture(rtsp_url2, cv2.CAP_FFMPEG)
        
        if not (self.cap1.isOpened() and self.cap2.isOpened()):
            messagebox.showerror("Error", "Cannot connect to both cameras")
            return
        
        # Load model YOLO sau khi kết nối camera thành công
        self.model = YOLO(r'D:/DaiHoc/Intern/PeopleDetected/best.pt')
        
        self.create_video_window()
        self.is_running = True
        
        # Khởi động thread xử lý video cho mỗi camera
        threading.Thread(target=self.update_video, args=(self.cap1, self.video_frame1, self.person_count_label1), daemon=True).start()
        threading.Thread(target=self.update_video, args=(self.cap2, self.video_frame2, self.person_count_label2), daemon=True).start()
        
        self.root.withdraw()
    
    def update_video(self, cap, video_frame, count_label):
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                time.sleep(0.05)
                continue
            
            # Resize frame để hiển thị đồng nhất
            frame = cv2.resize(frame, (480, 360))
            
            # Chạy nhận diện với model YOLO
            results = self.model(frame)
            frame_with_boxes = frame.copy()
            person_count = 0
            
            if results and results[0].boxes is not None and results[0].boxes.cls.numel() > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    unique_boxes = self.get_unique_boxes(boxes.xyxy.cpu().numpy(), threshold=15)
                    person_count = len(unique_boxes)
                    
                    # Vẽ các bounding boxes
                    for box in unique_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Cập nhật số người đếm được
            count_label.configure(text=str(person_count))
            
            # Chuyển frame sang định dạng phù hợp với Tkinter
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            video_frame.configure(image=img_tk)
            video_frame.image = img_tk
            
            # Thêm delay nhỏ để giảm tải CPU
            time.sleep(0.02)
    
    def get_unique_boxes(self, boxes, threshold=15):
        """Loại bỏ bounding boxes trùng lặp dựa trên khoảng cách giữa tâm"""
        unique_boxes = []
        used_indices = set()
        for i in range(len(boxes)):
            if i in used_indices:
                continue
            current_box = boxes[i]
            unique_boxes.append(current_box)
            cx1 = (current_box[0] + current_box[2]) / 2
            cy1 = (current_box[1] + current_box[3]) / 2
            
            for j in range(i + 1, len(boxes)):
                if j in used_indices:
                    continue
                next_box = boxes[j]
                cx2 = (next_box[0] + next_box[2]) / 2
                cy2 = (next_box[1] + next_box[3]) / 2
                if abs(cx1 - cx2) < threshold and abs(cy1 - cy2) < threshold:
                    used_indices.add(j)
        return unique_boxes

    def disconnect_cameras(self):
        self.is_running = False
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        if hasattr(self, 'video_window') and self.video_window.winfo_exists():
            self.video_window.destroy()
        self.root.deiconify()
        
    def on_closing(self):
        self.disconnect_cameras()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CameraApp()
    app.run()