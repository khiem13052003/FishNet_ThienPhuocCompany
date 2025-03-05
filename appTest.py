import cv2
import threading
import tkinter as tk
from tkinter import Canvas, Frame, Scrollbar
from module.deployModel import ImageProcessor

# Khởi tạo instance của ImageProcessor và dictionary chứa các Scale
processor = ImageProcessor()
scales = {}

# Danh sách tham số với thông tin: key, label, from, to, init, kiểu và resolution
params_list = [
    {"key": "ROI_x_start",       "label": "ROI_x_start",        "from": 0,    "to": 1920, "init": processor.extract_roi_size[0],  "type": "int",   "resolution": 1},
    {"key": "ROI_x_end",         "label": "ROI_x_end",          "from": 0,    "to": 1920, "init": processor.extract_roi_size[1],  "type": "int",   "resolution": 1},
    {"key": "ROI_y_start",       "label": "ROI_y_start",        "from": 0,    "to": 1080, "init": processor.extract_roi_size[2],  "type": "int",   "resolution": 1},
    {"key": "ROI_y_end",         "label": "ROI_y_end",          "from": 0,    "to": 1080, "init": processor.extract_roi_size[3],  "type": "int",   "resolution": 1},
    {"key": "Distance2Node",     "label": "Distance2Node",      "from": 0.0,  "to": 10.0, "init": processor.distance_2node,       "type": "float", "resolution": 0.01},
    {"key": "GKernel",          "label": "GKernel",           "from": 1,    "to": 31,   "init": processor.extract_maskNet_GaussianKernelSize[0], "type": "int", "resolution": 1},
    {"key": "Alpha",             "label": "Alpha",            "from": 0,    "to": 20,   "init": processor.extract_maskNet_addWeighted[0],        "type": "float", "resolution": 0.01},
    {"key": "Beta",              "label": "Beta",             "from": -20,  "to": 0,    "init": processor.extract_maskNet_addWeighted[1],        "type": "float", "resolution": 0.01},
    {"key": "ClipLimit",         "label": "ClipLimit",        "from": 0,    "to": 20,   "init": processor.extract_maskNet_CLAHE[0],              "type": "int",   "resolution": 1},
    {"key": "TileGridX",         "label": "TileGridX",        "from": 1,    "to": 100,  "init": processor.extract_maskNet_CLAHE[1][0],           "type": "int",   "resolution": 1},
    {"key": "TileGridY",         "label": "TileGridY",        "from": 1,    "to": 100,  "init": processor.extract_maskNet_CLAHE[1][1],           "type": "int",   "resolution": 1},
    {"key": "Thresh",            "label": "Thresh",           "from": 0,    "to": 255,  "init": processor.extract_maskNet_threshold,             "type": "float", "resolution": 0.01},
    {"key": "Erode_kernel_x",    "label": "Erode_kernel_x",   "from": 1,    "to": 20,   "init": processor.detect_node_erode[0][0],            "type": "int",   "resolution": 1},
    {"key": "Erode_kernel_y",    "label": "Erode_kernel_y",   "from": 1,    "to": 20,   "init": processor.detect_node_erode[0][1],            "type": "int",   "resolution": 1},
    {"key": "Erode_iter",        "label": "Erode_iter",       "from": 0,    "to": 10,   "init": processor.detect_node_erode[1],               "type": "int",   "resolution": 1},
    {"key": "Opened_kernel_x",   "label": "Opened_kernel_x",  "from": 1,    "to": 20,   "init": processor.detect_node_opened[0],              "type": "int",   "resolution": 1},
    {"key": "Opened_kernel_y",   "label": "Opened_kernel_y",  "from": 1,    "to": 20,   "init": processor.detect_node_opened[1],              "type": "int",   "resolution": 1},
    {"key": "Dilated_kernel_x",  "label": "Dilated_kernel_x", "from": 1,    "to": 20,   "init": processor.detect_node_dilated[0][0],          "type": "int",   "resolution": 1},
    {"key": "Dilated_kernel_y",  "label": "Dilated_kernel_y", "from": 1,    "to": 20,   "init": processor.detect_node_dilated[0][1],          "type": "int",   "resolution": 1},
    {"key": "Dilated_iter",      "label": "Dilated_iter",     "from": 0,    "to": 10,   "init": processor.detect_node_dilated[1],             "type": "int",   "resolution": 1},
    {"key": "Min_Area",          "label": "Min_Area",         "from": 0,    "to": 500,  "init": processor.gen_centers_min_area,                "type": "float", "resolution": 0.01},
    {"key": "Group_Y_Threshold", "label": "Group_Y_Threshold","from": 0,    "to": 20,   "init": processor.group_points_by_y_threshold,         "type": "float", "resolution": 0.01},
    {"key": "FilterRows_Threshold", "label": "FilterRows_Threshold", "from": 0.0, "to": 1.0, "init": processor.filter_rows_threshold,         "type": "float", "resolution": 0.01},
    {"key": "Expected_X_Dist",   "label": "Expected_X_Dist",  "from": 0,    "to": 100,  "init": processor.check_error_expected_x_distance,     "type": "float", "resolution": 0.01},
    {"key": "Allowed_X_Error",   "label": "Allowed_X_Error",  "from": 0,    "to": 10,   "init": processor.check_error_allowed_x_error,         "type": "float", "resolution": 0.01},
    {"key": "Expected_Angle",    "label": "Expected_Angle",   "from": 0,    "to": 90,   "init": processor.check_error_expected_angle,          "type": "float", "resolution": 0.01},
    {"key": "Allowed_Angle_Error","label": "Allowed_Angle_Error", "from": 0,  "to": 15,   "init": processor.check_error_allowed_angle_error,     "type": "float", "resolution": 0.01},
    {"key": "Expected_Y_Dist",   "label": "Expected_Y_Dist",  "from": 0,    "to": 200,  "init": processor.check_error_expected_y_distance,     "type": "float", "resolution": 0.01},
    {"key": "Allowed_Y_Error",   "label": "Allowed_Y_Error",  "from": 0,    "to": 20,   "init": processor.check_error_allowed_y_error,         "type": "float", "resolution": 0.01},
]

# Tạo dictionary để tra cứu kiểu dữ liệu của mỗi tham số
param_types = {param["key"]: param["type"] for param in params_list}

def get_val(key):
    """
    Lấy giá trị từ thanh trượt theo key.
    Nếu kiểu là float thì làm tròn 2 chữ số, còn lại chuyển về int.
    """
    value = scales[key].get()
    if param_types[key] == "float":
        return round(value, 2)
    else:
        return int(value)

def video_loop():
    """
    Hàm xử lý video:
      - Mở camera và đặt độ phân giải 1920x1080.
      - Liên tục đọc frame, cập nhật tham số từ giao diện và xử lý frame.
      - Hiển thị kết quả xử lý và lắng nghe phím 'q' để thoát.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    # Đặt độ phân giải cho camera
    ImageProcessor.set_resolution(cap, list_reso=(1920, 1080))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cập nhật các tham số từ giao diện
        processor.extract_roi_size = (
            get_val("ROI_x_start"),
            get_val("ROI_x_end"),
            get_val("ROI_y_start"),
            get_val("ROI_y_end")
        )
        processor.distance_2node = get_val("Distance2Node")

        # Gaussian kernel: đảm bảo số lẻ
        gk = get_val("GKernel")
        if gk % 2 == 0:
            gk += 1
        processor.extract_maskNet_GaussianKernelSize = (gk, gk)

        processor.extract_maskNet_addWeighted = (get_val("Alpha"), get_val("Beta"))
        processor.extract_maskNet_CLAHE = (
            get_val("ClipLimit"),
            (get_val("TileGridX"), get_val("TileGridY"))
        )
        processor.extract_maskNet_threshold = get_val("Thresh")
        processor.detect_node_erode = (
            (get_val("Erode_kernel_x"), get_val("Erode_kernel_y")),
            get_val("Erode_iter")
        )
        processor.detect_node_opened = (get_val("Opened_kernel_x"), get_val("Opened_kernel_y"))
        processor.detect_node_dilated = (
            (get_val("Dilated_kernel_x"), get_val("Dilated_kernel_y")),
            get_val("Dilated_iter")
        )
        processor.gen_centers_min_area = get_val("Min_Area")
        processor.group_points_by_y_threshold = get_val("Group_Y_Threshold")
        processor.filter_rows_threshold = get_val("FilterRows_Threshold")
        processor.check_error_expected_x_distance = get_val("Expected_X_Dist")
        processor.check_error_allowed_x_error = get_val("Allowed_X_Error")
        processor.check_error_expected_angle = get_val("Expected_Angle")
        processor.check_error_allowed_angle_error = get_val("Allowed_Angle_Error")
        processor.check_error_expected_y_distance = get_val("Expected_Y_Dist")
        processor.check_error_allowed_y_error = get_val("Allowed_Y_Error")
        
        # Xử lý frame và hiển thị kết quả
        error, result = processor.process(frame, isLoadImg=False, isShow=False)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

def create_control_panel(root):
    """
    Tạo giao diện điều khiển với Tkinter:
      - Sử dụng Canvas và Scrollbar để hiển thị các thanh trượt có thể cuộn.
      - Sắp xếp các thanh trượt thành các hàng và cột.
    """
    canvas = Canvas(root)
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    # Cập nhật vùng cuộn của canvas khi frame thay đổi kích thước
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Xử lý sự kiện con lăn chuột
    def _on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    # Tạo các Scale và sắp xếp 2 thanh trượt trên 1 hàng
    for index, param in enumerate(params_list):
        row = index // 2
        col = index % 2
        key = param["key"]
        if param["type"] == "float":
            scales[key] = tk.Scale(
                scrollable_frame,
                from_=float(param["from"]),
                to=float(param["to"]),
                resolution=param["resolution"],
                label=param["label"],
                orient=tk.HORIZONTAL,
                length=250
            )
        else:
            scales[key] = tk.Scale(
                scrollable_frame,
                from_=param["from"],
                to=param["to"],
                resolution=param["resolution"],
                label=param["label"],
                orient=tk.HORIZONTAL,
                length=250
            )
        scales[key].set(param["init"])
        scales[key].grid(row=row, column=col, padx=5, pady=5, sticky='ew')

    # Cấu hình giãn đều các cột
    for col in range(2):
        scrollable_frame.grid_columnconfigure(col, weight=1)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scales

def main():
    """
    Hàm main:
      - Tạo cửa sổ chính với Tkinter.
      - Tạo giao diện điều khiển và khởi chạy xử lý video trong một luồng riêng.
    """
    global root, scales
    root = tk.Tk()
    root.geometry("550x700")
    root.title("Control Panel")

    scales = create_control_panel(root)

    video_thread = threading.Thread(target=video_loop, daemon=True)
    video_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()
