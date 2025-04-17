import cv2
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QEvent, QTimer, QDateTime, QTime
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

class ImageProcessor:
    def __init__(self,
                 extract_roi_size: tuple[int, int, int, int] = (),
                 distance_net_cm_y: float = 7.344,
                 distance_net_cm_x: float = 1,
                 distance_net_pixel_cm: float = 21,
                 undistort_image_cameraMatrix: np.ndarray = np.array([[1231.3286774632959, 0.0, 744.03151798533293],
                                                                      [0.0, 1223.7983175377726, 358.9846249652723],
                                                                      [0.0, 0.0, 1.0]], dtype=np.float64),
                 undistort_image_distCoeffs: np.ndarray = np.array([0.11552635184620702, -0.09582978701393477,
                                                                    -0.00031662782604741855, 0.031827549046139048,
                                                                    -0.078108159887496087], dtype=np.float64),
                 extract_maskNet_GaussianKernelSize: tuple = (11,11),
                 extract_maskNet_addWeighted: tuple=(7,-6),
                 extract_maskNet_CLAHE: tuple=(7,(60,40)),
                 extract_maskNet_threshold: float = 190,
                 detect_node_erode: tuple=((5,3), 1), 
                 detect_node_opened: tuple=(2,2),
                 detect_node_dilated: tuple=((3,3), 4),
                 gen_centers_min_area: float = 50.0,
                 group_points_by_y_threshold: float = 4.0,
                 filter_rows_threshold: float=0.5,
                 check_error_allowed_x_error: float = 5.0,
                 check_error_expected_angle: float = 0.0, 
                 check_error_allowed_angle_error: float = 10.0,
                 check_error_allowed_y_error: float = 50.0):
        """
        Khởi tạo các tham số cấu hình cho quá trình xử lý ảnh.

        Parameters:
            extract_roi_size (tuple): Kích thước vùng ROI theo định dạng (x_start, x_end, y_start, y_end).
            distance_net_cm_y (float): Kích thước thực 1 mắt lưới theo chiều dọc (cm).
            distance_net_cm_x: (float): Kích thước thực giữa 2 nút liên tiếp (cm).
            distance_net_pixel_cm: (float): Kích thước 1 cm (pixel).
            distance_cm_cam2net: (float): khoảng cách từ camera đến lưới (cm).
            undistort_image_cameraMatrix (np.ndarray): Ma trận nội tại của máy ảnh.
            undistort_image_distCoeffs (np.ndarray): Hệ số biến dạng của máy ảnh.
            extract_maskNet_GaussianKernelSize (tuple): Kích thước kernel cho Gaussian blur.
            extract_maskNet_addWeighted (tuple): Hệ số cho phép cộng trọng số để tăng cường độ sắc nét (alpha, beta).
            extract_maskNet_CLAHE (tuple): Thông số cho CLAHE gồm (clipLimit, (tileGridSize_x, tileGridSize_y)).
            extract_maskNet_threshold (float): Ngưỡng để chuyển ảnh sang nhị phân.
            detect_node_erode (tuple): Thông số cho phép xói mòn, gồm (kernel size, iterations).
            detect_node_opened (tuple): Kích thước kernel cho phép mở (morphological opening).
            detect_node_dilated (tuple): Thông số cho phép giãn nở, gồm (kernel size, iterations).
            gen_centers_min_area (float): Diện tích tối thiểu để xem một đối tượng là nút hợp lệ.
            group_points_by_y_threshold (float): Ngưỡng cho phép sai lệch theo trục y để nhóm các điểm.
            filter_rows_threshold (float): Tỉ lệ phần trăm số nút tối thiểu (so với trung bình) để duy trì hàng.
            check_error_expected_x_distance (float): Khoảng cách ngang dự kiến giữa các nút.
            check_error_allowed_x_error (float): Sai số cho phép đối với khoảng cách ngang.
            check_error_expected_angle (float): Góc dự kiến giữa các nút khi nối với nhau.
            check_error_allowed_angle_error (float): Sai số cho phép đối với góc.
            check_error_expected_y_distance (float): Khoảng cách dọc dự kiến giữa các hàng nút.
            check_error_allowed_y_error (float): Sai số cho phép đối với khoảng cách dọc.
        """
        self.extract_roi_size = extract_roi_size
        self.distance_net_cm_y = distance_net_cm_y
        self.distance_net_cm_x = distance_net_cm_x
        self.distance_net_pixel_cm = distance_net_pixel_cm
        self.undistort_image_cameraMatrix = undistort_image_cameraMatrix
        self.undistort_image_distCoeffs = undistort_image_distCoeffs
        self.extract_maskNet_GaussianKernelSize = extract_maskNet_GaussianKernelSize
        self.extract_maskNet_addWeighted = extract_maskNet_addWeighted
        self.extract_maskNet_CLAHE = extract_maskNet_CLAHE
        self.extract_maskNet_threshold = extract_maskNet_threshold
        self.detect_node_erode = detect_node_erode
        self.detect_node_opened = detect_node_opened
        self.detect_node_dilated = detect_node_dilated
        self.group_points_by_y_threshold = group_points_by_y_threshold
        self.gen_centers_min_area = gen_centers_min_area
        self.filter_rows_threshold = filter_rows_threshold
        self.check_error_allowed_x_error = check_error_allowed_x_error
        self.check_error_expected_angle = check_error_expected_angle
        self.check_error_allowed_angle_error = check_error_allowed_angle_error
        self.check_error_allowed_y_error = check_error_allowed_y_error

    @staticmethod
    def set_resolution(cap: cv2.VideoCapture, list_reso: tuple=(1920, 1080)) -> None:
        """Đặt độ phân giải theo tuple (width, height)"""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, list_reso[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, list_reso[1])

    @staticmethod
    def cm_to_pixel(cmPerPixel: float, distance_cm: float) -> float:
        """
        Chuyển đổi kích thước thực của đối tượng (cm) sang kích thước hiển thị (pixels)
        dựa trên nguyên lý tam giác tương tự.
        
        Công thức chuyển đổi:
            P = (F * W) / D
        trong đó:
        - F là độ dài tiêu cự (đã được hiệu chuẩn từ đối tượng chuẩn)
        - W là kích thước thực của đối tượng cần đo (cm)
        - D là khoảng cách từ camera đến đối tượng (cm)
        
        Parameters:
            object_width_cm (float): Kích thước thực của đối tượng (cm).
            distance_cm (float): Khoảng cách từ camera đến đối tượng (cm).
            focal_length (float): Độ dài tiêu cự đã hiệu chuẩn (theo công thức calibrate_focal_length).
        
        Returns:
            float: Kích thước đối tượng trong ảnh tính theo pixels.
        """
        return round(distance_cm*cmPerPixel,2)

    def extract_roi(self, image: np.ndarray, isShow: bool, size: tuple) -> np.ndarray:
        """
        Cắt và trích xuất vùng ROI từ ảnh dựa trên kích thước đã cho.
        """
        # print("Received ROI size in extract_roi:", size)
        if len(size) != 4:
            print("Invalid ROI size tuple")
            return image

        x_start, x_end, y_start, y_end = size
        # print("Extracting ROI with coordinates:", 
        #       x_start, x_end, y_start, y_end)
        
        rs = image[y_start:y_end, x_start:x_end]
        if isShow:
            cv2.moveWindow("ROI", 100, 100)
        return rs


    def extract_maskNet(self, roi: np.ndarray, isShow: bool,
        GaussianKernelSize: tuple=(11,11),
        addWeighted: tuple=(7,-6),
        CLAHE: tuple=(7,(60,40)),
        threshold: float=190
    ) -> np.ndarray:
        """
        Tăng cường độ tương phản và chuyển ảnh ROI sang nhị phân.

        Quy trình:
          - Chuyển ảnh ROI sang thang độ xám.
          - Áp dụng GaussianBlur để giảm nhiễu.
          - Tăng độ sắc nét qua phép cộng trọng số.
          - Sử dụng CLAHE để tăng độ tương phản.
          - Chuyển ảnh đã xử lý sang ảnh nhị phân theo ngưỡng.

        Parameters:
            roi (np.ndarray): Ảnh ROI đầu vào.
            isShow (bool): Cờ xác định có hiển thị các bước trung gian hay không.
            GaussianKernelSize (tuple): Kích thước kernel cho GaussianBlur.
            addWeighted (tuple): Hệ số (alpha, beta) cho phép tăng sắc nét.
            CLAHE (tuple): Thông số cho CLAHE gồm (clipLimit, (tileGridSize_x, tileGridSize_y)).
            threshold (float): Ngưỡng chuyển đổi ảnh xám sang nhị phân.

        Returns:
            np.ndarray: Ảnh nhị phân sau khi xử lý.
        """
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray_roi", gray_roi)

        #Tăng độ tương phản
        filtered = cv2.GaussianBlur(gray_roi, GaussianKernelSize, 0)
        # median_filtered = cv2.medianBlur(claheImage, 11)
        sharpened = cv2.addWeighted(gray_roi, addWeighted[0], filtered, addWeighted[1], 0)
        # f2 = cv2.GaussianBlur(sharpened, (3,3), 0)
        clahe = cv2.createCLAHE(clipLimit=CLAHE[0], tileGridSize=CLAHE[1])
        claheImage = clahe.apply(sharpened)
        # cv2.imshow("sharpened", claheImage)
        # cv2.moveWindow("sharpened", 0, 0)

        #chuyển sang ảnh nhị phân
        _, thresh = cv2.threshold(claheImage, threshold, 255, cv2.THRESH_BINARY)

        # if isShow:
        #     # cv2.imshow("thresh", thresh)
        #     # cv2.moveWindow("thresh", 0, 0)
        return thresh
    
    def undistort_image(self, image: np.ndarray, cameraMatrix: np.ndarray, distCoeffs: np.ndarray, isShow: bool):
        """
        Hiệu chỉnh ảnh dựa trên thông số máy ảnh đã được hiệu chỉnh.

        Quy trình:
        - Tính toán new camera matrix tối ưu và xác định vùng quan tâm (ROI) dựa trên kích thước ảnh.
        - Hiệu chỉnh ảnh sử dụng ma trận máy ảnh ban đầu và các hệ số biến dạng.
        - Cắt ảnh theo ROI để loại bỏ các vùng ảnh không hợp lệ sau khi hiệu chỉnh.

        Parameters:
            image (np.ndarray): Ảnh đầu vào cần được hiệu chỉnh.
            cameraMatrix (np.ndarray): Ma trận nội tại của máy ảnh.
            distCoeffs (np.ndarray): Hệ số biến dạng của máy ảnh.
            isShow (bool): Cờ cho biết có hiển thị các bước trung gian hay không.

        Returns:
            np.ndarray: Ảnh sau khi được hiệu chỉnh và cắt theo ROI.
        """
        h, w = image.shape[:2]
        
        # Tính toán new camera matrix tối ưu và ROI
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
        
        # Hiệu chỉnh ảnh
        undistorted_img = cv2.undistort(image, cameraMatrix, distCoeffs, None, newCameraMatrix)
        
        # Cắt ảnh theo ROI
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        # if isShow:
        #     cv2.imshow("undistort_image", undistorted_img)
        #     cv2.moveWindow(0,0)

        return undistorted_img

    def detect_node(self, image: np.ndarray, 
                       isShow: bool, 
                       erode: tuple=((5,3), 1), 
                       opened: tuple=(2,2),
                       dilated: tuple=((3,3), 4)) -> np.ndarray:
        """
        Phát hiện các góc (nút) lưới trong ảnh nhị phân thông qua các thao tác xói mòn, mở và giãn nở.

        Quy trình:
          - Xói mòn ảnh để loại bỏ nhiễu.
          - Sử dụng phép mở (opening) để loại bỏ các chi tiết nhỏ không mong muốn.
          - Giãn nở ảnh để nhấn mạnh các góc phát hiện được.

        Parameters:
            image (np.ndarray): Ảnh nhị phân đầu vào.
            isShow (bool): Cờ xác định có hiển thị các bước trung gian hay không.
            erode (tuple): Thông số cho phép xói mòn gồm (kernel size, iterations).
            opened (tuple): Kích thước kernel cho phép mở.
            dilated (tuple): Thông số cho phép giãn nở gồm (kernel size, iterations).

        Returns:
            np.ndarray: Ảnh sau khi giãn nở, với các góc được nhấn mạnh.
        """
        #xói mòn ảnh
        erodeImage = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,erode[0]), iterations = erode[1])
        # cv2.imshow("erode", erodeImage)
        # cv2.moveWindow("erode", 0, 0)

        # Loại bỏ các chi tiết không đủ chiều ngang (opening)
        openedImage = cv2.morphologyEx(erodeImage, cv2.MORPH_OPEN, np.ones(opened, np.uint8))
        # cv2.imshow("openedImage", openedImage)
        # cv2.moveWindow("openedImage", 0, 0)
        # Thực hiện phép dilation, có thể thay đổi iterations để tăng mức độ nở
        dilatedImage = cv2.dilate(openedImage, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,dilated[0]), iterations=dilated[1] )

        # if isShow:
        #     cv2.imshow("dilatedImage", dilatedImage)
        #     cv2.moveWindow("dilatedImage", 0, 0)
        return dilatedImage

    def draw_mask_on_image(self, image: np.ndarray, mask: np.ndarray, isShow: bool, 
        color=(0, 0, 255), alpha=1):
        """
        Vẽ overlay của mask lên ảnh gốc với màu sắc và độ trong suốt đã cho.

        Parameters:
            image (np.ndarray): Ảnh gốc.
            mask (np.ndarray): Mask nhị phân (các giá trị > 0 được xem là vùng mask).
            isShow (bool): Cờ xác định có hiển thị ảnh kết quả hay không.
            color (tuple): Màu dùng để vẽ mask (theo định dạng BGR).
            alpha (float): Hệ số trong suốt của overlay.

        Returns:
            np.ndarray: Ảnh sau khi overlay mask.
        """
        # Chuyển mask về dạng nhị phân (0 hoặc 1)
        mask = (mask > 0).astype(np.uint8)
        
        mask_colored = np.zeros_like(image, dtype=np.uint8)
        mask_colored[mask == 1] = color
        masked_image = cv2.addWeighted(image, 1, mask_colored, alpha, 0, dtype=cv2.CV_8U)
        
        # if isShow:
        #     cv2.imshow("detect corners rgb", masked_image)
        #     cv2.moveWindow("detect corners rgb", 0, 0)

        return masked_image

    def gen_centers(self, binary_image: np.ndarray, raw_image: np.ndarray, isShow: bool, min_area: float=50) -> tuple:
        """
        Nhận diện các thành phần liên thông trong ảnh nhị phân và tính toán tâm của mỗi thành phần.

        Parameters:
            binary_image (np.ndarray): Ảnh nhị phân chứa các thành phần liên thông.
            raw_image (np.ndarray): Ảnh gốc dùng để vẽ tâm các thành phần.
            isShow (bool): Cờ xác định có hiển thị ảnh kết quả hay không.
            min_area (float): Diện tích tối thiểu để xem một đối tượng là nút hợp lệ.

        Returns:
            tuple: (Danh sách các tâm dạng (x, y), Ảnh gốc đã được vẽ tâm).
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        centers = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                centers.append((cx, cy))
                cv2.circle(raw_image, (cx, cy), 5, (0, 0, 255), -1)
        # if isShow:
        #     cv2.imshow("Centers", raw_image)
        #     cv2.moveWindow("Centers", 0, 0)
        return centers, raw_image
    
    def group_points_by_y(
        self,
        points: tuple[tuple[int, int]],
        raw_image: np.ndarray,
        isShow: bool,
        threshold: float=4
    ) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
        """
        Nhóm các điểm dựa trên tọa độ y và vẽ các đường nối giữa các điểm cùng nhóm.

        Quy trình:
          1. Sắp xếp các điểm theo thứ tự tăng dần của tọa độ y.
          2. Quét danh sách và nhóm các điểm có hiệu số y nhỏ hơn hoặc bằng threshold.
          3. Sắp xếp các điểm trong mỗi nhóm theo thứ tự tăng dần của tọa độ x.
          4. Vẽ các đường nối và chấm tròn các điểm trong từng nhóm trên ảnh.

        Parameters:
            points (tuple): Danh sách các điểm dạng (x, y).
            raw_image (np.ndarray): Ảnh dùng để vẽ các nhóm điểm.
            isShow (bool): Cờ xác định có hiển thị ảnh kết quả hay không.
            threshold (float): Ngưỡng cho phép sai lệch trên trục y để nhóm các điểm.

        Returns:
            tuple: (Danh sách các nhóm điểm, Ảnh đã vẽ kết quả).
        """
        if not points:
            return [], raw_image

        # 1) Sắp xếp theo y tăng dần
        points.sort(key=lambda p: p[1])

        # Lưu lại danh sách điểm đã sắp xếp theo y để in số thứ tự sau
        sorted_points = points.copy()

        groups = []
        current_group = [points[0]]

        # 2) Gom nhóm theo threshold
        for i in range(1, len(points)):
            prev_point = points[i - 1]
            current_point = points[i]
            
            # Chênh lệch y của hai điểm liên tiếp
            diff_y = current_point[1] - prev_point[1]

            if diff_y <= threshold:
                # Vẫn nằm trong cùng một nhóm
                current_group.append(current_point)
            else:
                # Tạo nhóm mới
                groups.append(current_group)
                current_group = [current_point]

        # Đừng quên nhóm cuối cùng
        groups.append(current_group)

        # 3) Sắp xếp từng nhóm theo x tăng dần
        for group in groups:
            # color = [random.randint(0, 255) for _ in range(3)]
            color = (0,255,0)
            # Vẽ các đoạn thẳng nối điểm liền kề
            if len(group) > 1:
                for i in range(len(group) - 1):
                    pt1 = group[i]
                    pt2 = group[i + 1]
                    cv2.line(raw_image, pt1, pt2, (234, 63, 247), 2)
            # Vẽ chấm tròn cho từng điểm
            for (x, y) in group:
                cv2.circle(raw_image, (x, y), 5, color, -1)

        # # 5) In số thứ tự của các điểm (theo thứ tự tăng dần của y) lên ảnh.
        # #    Sử dụng sorted_points đã được sắp xếp theo y.
        # for idx, (x, y) in enumerate(sorted_points, start=1):
        #     # Dịch chuyển vị trí in số để không che điểm
        #     cv2.putText(raw_image, str(idx), (x + 7, y - 7),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        # if isShow:
        #     cv2.imshow("Grouped Points", raw_image)
        #     cv2.moveWindow("Grouped Points", 0, 0)
        return groups, raw_image

    def filter_rows(self, matrix: list, threshold: float = 0.5) -> list:
        """
        Lọc các hàng (row) dựa trên số lượng nút để lấy ra hàng ở giữa và các hàng lân cận.

        Parameters:
            matrix (list): Danh sách các hàng, mỗi hàng là danh sách các điểm.
            threshold (float): Tỉ lệ phần trăm số nút tối thiểu (so với trung bình) để duy trì hàng.

        Returns:
            list: Danh sách các hàng đã được lọc.
        """
        if not matrix:
            return []

        avg_nodes = sum(len(row) for row in matrix) / len(matrix) #trung bình số nút trong 1 hàng
        filtered_rows = [row for row in matrix if len(row) >= threshold * avg_nodes] #lọc ra những hàng có số nút ít hơn threshold% (50%)

        if not filtered_rows:
            return []

        #lọc ra hàng đầu, cuối. nút ở đầu, cuối.
        n = len(filtered_rows)
        mid_index = n // 2 - 1 if n % 2 == 0 else n // 2
        start_index = max(0, mid_index - 1)
        end_index = min(n, mid_index + 2)
        return filtered_rows[start_index:end_index]

    def check_errors(self, matrix: list, raw_image: np.ndarray,
                     expected_x_distance: float = 20.0, allowed_x_error: float = 5.0,
                 expected_angle: float = 0.0, allowed_angle_error: float = 10.0,
                 expected_y_distance: float = 80.0, allowed_y_error: float = 8.0) -> tuple[np.ndarray,bool]:
        """
        Kiểm tra lỗi trong việc sắp xếp các điểm dựa trên khoảng cách ngang, góc và khoảng cách dọc giữa các hàng.

        Quy trình:
          - Với mỗi cặp điểm liên tiếp trong một hàng, kiểm tra khoảng cách theo trục x và góc tạo thành.
          - Tính trung bình vị trí của các hàng và kiểm tra khoảng cách dọc giữa các hàng.
          - Vẽ các đường với màu sắc khác nhau để thể hiện kết quả kiểm tra (lỗi hay không).

        Parameters:
            matrix (list): Danh sách các hàng, mỗi hàng chứa các điểm.
            raw_image (np.ndarray): Ảnh dùng để vẽ các chỉ số lỗi.
            expected_x_distance (float): Khoảng cách ngang dự kiến giữa các điểm.
            allowed_x_error (float): Sai số cho phép cho khoảng cách ngang.
            expected_angle (float): Góc dự kiến giữa các điểm khi nối với nhau.
            allowed_angle_error (float): Sai số cho phép của góc.
            expected_y_distance (float): Khoảng cách dọc dự kiến giữa các hàng.
            allowed_y_error (float): Sai số cho phép của khoảng cách dọc.

        Returns:
            tuple: (Ảnh đã vẽ chỉ báo lỗi, Cờ Boolean cho biết có lỗi hay không).
        """
        error = False
        allowed_x_min = expected_x_distance - allowed_x_error
        allowed_x_max = expected_x_distance + allowed_x_error
        allowed_angle_min = expected_angle - allowed_angle_error
        allowed_angle_max = expected_angle + allowed_angle_error

        color_good = (0, 255, 0)
        color_error_x = (0, 0, 255)
        color_error_degree = (0, 255, 255)
        color_error_y = (0, 165, 255)

        for row in matrix:
            for i in range(len(row) - 1):
                p1 = row[i]
                p2 = row[i+1]
                dx = abs(p2[0] - p1[0])
                dy = p2[1] - p1[1]
                angle = math.degrees(math.atan2(dy, dx))
                if dx < allowed_x_min or dx > allowed_x_max:
                    cv2.line(raw_image, p1, p2, color_error_x, 2)
                    error = True
                elif angle < allowed_angle_min or angle > allowed_angle_max:
                    cv2.line(raw_image, p1, p2, color_error_degree, 2)
                    error = True
                else:
                    cv2.line(raw_image, p1, p2, color_good, 2)

        avg_points = []
        for row in matrix:
            if row:
                avg_x = int(sum(p[0] for p in row) / len(row))
                avg_y = int(sum(p[1] for p in row) / len(row))
                avg_points.append((avg_x, avg_y))

        for i in range(len(avg_points) - 1):
            pt1 = avg_points[i]
            pt2 = avg_points[i+1]
            dy = abs(pt2[1] - pt1[1])
            if dy < (expected_y_distance - allowed_y_error) or dy > (expected_y_distance + allowed_y_error):
                cv2.line(raw_image, pt1, pt2, color_error_y, 2)
                cv2.circle(raw_image,pt1,2,color_error_y,2)
                cv2.circle(raw_image,pt2,2,color_error_y,2)
                error = True

        return raw_image, error

    def process(self, src, isLoadImg=False, isShow=False, skip_roi=False):
        """
        Xử lý ảnh
        
        Parameters:
            skip_roi: Flag cho biết có bỏ qua bước cắt ROI không,Nếu True, sẽ bỏ qua bước cắt ROI.
            src: Đường dẫn file ảnh hoặc mảng ảnh.
            isLoadImg (bool): Nếu True, sẽ load ảnh từ file.
            isShow (bool): Nếu True, sẽ hiển thị các bước trung gian và kết quả cuối cùng.
            
        
        Returns:
            tuple: (Cờ báo lỗi (True nếu có lỗi), Ảnh kết quả cuối cùng đã vẽ chỉ báo lỗi).
        """
        try:
            # Kiểm tra ảnh đầu vào
            if src is None or src.size == 0:
                    print("Ảnh đầu vào không hợp lệ")
                    return False, src

                # Bước 1: Đọc và chuyển đổi màu nếu cần
            if isLoadImg:
                    image = cv2.imread(src)
            else:
            
                    # Đảm bảo ảnh ở định dạng BGR
                    if len(src.shape) == 3 and src.shape[2] == 3:
                        image = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
                    else:
                        image = src
                    cv2.imshow('image',image)
                # Kiểm tra sau khi đọc ảnh
            if image is None or image.size == 0:
                    print("Không thể đọc ảnh")
                    return False, src
                
            sort_image = self.undistort_image(image, self.undistort_image_cameraMatrix, self.undistort_image_distCoeffs, isShow=False)
                # Bước 2: Xử lý ROI
            if not skip_roi:
                    roi = self.extract_roi(sort_image, isShow=False, size=self.extract_roi_size)
            else:
                    # Đảm bảo định dạng màu đúng
                    roi = cv2.cvtColor(src, cv2.COLOR_RGB2BGR) if len(src.shape) == 3 else src

                # Debug: In ra thông tin về ảnh
                # print("Shape của ROI:", roi.shape if roi is not None else "None")
                # print("Kiểu dữ liệu của ROI:", roi.dtype if roi is not None else "None")
            cv2.imshow('roi',roi)
                # Tiếp tục xử lý...
            edges = self.extract_maskNet(roi, isShow=True,
                                            GaussianKernelSize=self.extract_maskNet_GaussianKernelSize,
                                            addWeighted=self.extract_maskNet_addWeighted,
                                            CLAHE=self.extract_maskNet_CLAHE,
                                            threshold=self.extract_maskNet_threshold)

                # Bước 4: Phát hiện góc và tạo mặt nạ
            corner_mask = self.detect_node(edges, isShow=True)
            self.draw_mask_on_image(roi, corner_mask, isShow=True)
            # Bước 5: Xác định các nút lưới
            centers, center_image = self.gen_centers(corner_mask, roi, isShow=True, 
                                                    min_area=self.gen_centers_min_area)
            # Bước 6: Nhóm các điểm theo trục y
            gr, gr_image = self.group_points_by_y(centers, roi, isShow=True, 
                                                threshold=self.group_points_by_y_threshold)
            # print(gr)
            # Bước 7: Lọc các hàng
            filter_gr = self.filter_rows(gr, 
                                        threshold=self.filter_rows_threshold)
            # Bước 8: Kiểm tra lỗi
            final_rs, e = self.check_errors(filter_gr, gr_image,
                                            expected_x_distance=self.distance_net_cm_x,
                                            allowed_x_error=self.check_error_allowed_x_error,
                                            expected_angle=self.check_error_expected_angle,
                                            allowed_angle_error=self.check_error_allowed_angle_error,
                                            expected_y_distance=self.distance_net_cm_y,
                                            allowed_y_error=self.check_error_allowed_y_error)
            if isShow:
                cv2.imshow("Final Result", final_rs)
                # cv2.moveWindow("Final Result", 0, 0)
                print(f"x check: {self.distance_net_cm_x} pixel\ty check: {self.distance_net_cm_y} pixel\t angle check: {self.check_error_expected_angle} degree")
                print("Lỗi") if e else print("Không lỗi")
                print("---------------------------")
            # cv2.imshow('final_rs',final_rs)
            # print('distance_net_x', self.distance_net_cm_x)
            # print('distance_net_y',self.distance_net_cm_y)
            return e,final_rs
            
        except Exception as e:
            print(f"Lỗi trong process: {str(e)}")
        return False, src
            

class GUIProcessor:
    def __init__(self):
        self.distance_net_x = 0
        self.count_timer = None
        # self.total_node = 0
        self.distance_net_y = 0
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0
        self.erro_data = []
        # Khởi tạo ImageProcessor một lần để tái sử dụng
        self.image_processor = ImageProcessor()
        self.error_count= 0
        self.gap_seconds =5
        self.last_process_time= None
        self.save_path= ""
        self.now= QDateTime.currentDateTime()

    def handle_submit(self, control_panel):
        distance_net_x = control_panel.distance_node_edit.text()
        gap_time = control_panel.gap_time_edit.time()
        # total_node = control_panel.total_node_edit.text()
        distance_net_y = control_panel.distance_node_column_edit.text()
        
        if not distance_net_x:
            control_panel.result_text.setText("Vui lòng nhập khoảng cách giữa 2 nút ngang !")
            return
        # if not total_node:
        #     control_panel.result_text.setText("Vui lòng nhập tổng số nút trên 1 hàng !")
        #     return
        if not distance_net_y:
            control_panel.result_text.setText("Vui lòng nhập khoảng cách giữa 2 nút nằm dọc !")
            return
        
        self.gap_seconds = gap_time.hour() * 3600 + gap_time.minute() * 60 + gap_time.second()
        
        self.image_processor.distance_net_cm_x= float(distance_net_x)
        self.image_processor.distance_net_cm_y=float(distance_net_y)
        
        self.error_count= 0
        self.erro_data =[]
        self.last_process_time= None
        
        # Xóa kết quả cũ
        control_panel.result_text.clear()
        control_panel.submit_button.setEnabled(False)
        
        if self.count_timer and self.count_timer.isActive():
            self.count_timer.stop()
            
        


    def check_timer(self):
        """Kiểm tra xem có nên xử lý frame hiện tại không"""
        current_time= QDateTime.currentDateTime()
        if self.last_process_time is None:
            self.last_process_time= current_time
            return True
        
        time_diff= self.last_process_time.secsTo(current_time)
        if time_diff >= self.gap_seconds:
            self.last_process_time= current_time
            return True
        return False

    def handle_delete(self, control_panel):
        if self.count_timer and self.count_timer.isActive():
            self.count_timer.stop()

        control_panel.distance_node_edit.clear()
        control_panel.gap_time_edit.setTime(QTime(0, 0, 5))
        # control_panel.total_node.clear()
        control_panel.distance_node_column_edit.clear()
        control_panel.result_text.clear()
        control_panel.submit_button.setEnabled(True)
        self.error_data = []
        self.error_count=0

        self.image_processor.distance_net_cm_x= None
        self.image_processor.distance_net_cm_y=None

    def clean_up(self):
        if self.count_timer and self.count_timer.isActive():
            self.count_timer.stop()

    def set_save_path(self,path):
        """set folder path để lưu ảnh"""
        self.save_path= path
        folder_name= self.now.toString("yyyy-MM-dd")
        self.path_folder= os.path.join(self.save_path,folder_name)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
    
    def save_error_pic(self, frame, error_time: QDateTime):
        """Lưu Frame khi detect được lỗi"""
        try:
            if not self.save_path:
                    return False
            file_name = error_time.toString("HH-mm-ss") + ".png"
            file_path= os.path.join(self.path_folder,file_name)

            cv2.imwrite(file_path,frame)
            print("da luu anh")
            return True
        except Exception as e:
            print("Loi khi luu anh")
            return False
    
    def process_frame(self, frame):
        try: 
            check_error= self.check_timer()
            
            # Kiểm tra frame đầu vào
            if frame is None or frame.size == 0:
                print("Frame đầu vào rỗng")
                return False, None

            # Kiểm tra tọa độ ROI
            if None in [self.x_start, self.x_end, self.y_start, self.y_end]:
                print("Tọa độ ROI chưa được thiết lập")
                return False, None

            # Tạo tuple size cho hàm extract_roi
            roi_size = (int(self.x_start), int(self.x_end), 
                    int(self.y_start), int(self.y_end))
            
            # print("ROI size tuple:", roi_size)

            # Trích xuất vùng ROI
            roi_frame = self.image_processor.extract_roi(
                image=frame,
                isShow=False,
                size=roi_size
            )

            if roi_frame is None or roi_frame.size == 0:
                print("ROI frame rỗng")
                return False, None

            # Xử lý frame với vùng ROI đã cắt
            has_error, processed_frame = self.image_processor.process(
                src=roi_frame,  # Truyền frame đã cắt ROI
                isLoadImg=False,
                isShow=False,
                skip_roi=True  # Thêm flag để không cắt ROI lần nữa
            )
            cv2.imshow("roi frame",roi_frame)
            if check_error and has_error:
                error_time = QDateTime.currentDateTime()
                self.error_count += 1
                self.erro_data.append({
                    'time': error_time,
                    'frame': processed_frame.copy()
                })
                print('current time: ', error_time)
                self.save_error_pic(processed_frame,error_time)
            return has_error, processed_frame
        except Exception as e:
            print(f"Lỗi khi xử lý frame: {str(e)}")
            return False, None


    
    # def check_count(self, control_panel):
    #     """
    #     Kiểm tra và cập nhật số lượng lỗi theo thời gian
    #     """
    #     # Đếm số lỗi trong khoảng thời gian
    #     current_time = QDateTime.currentDateTime()
    #     # Thêm logic đếm lỗi và cập nhật giao diện ở đây
        
    #     pass

    def set_roi_coordinates(self, x_start, x_end, y_start, y_end):
        """Set tọa độ ROI cho việc xử lý"""
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

# if __name__ == "__main__":
#     folder_path = r"D:\DaiHoc\Intern\ThienPhuocCompany\data_fishNet\luoiMoi2"  # Thay đổi đường dẫn phù hợp
#     processor = ImageProcessor()
#     processor.processImgFolder(folder_path)

#Ký tên: Khoa
