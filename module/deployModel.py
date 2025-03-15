import cv2
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self,
                 extract_roi_size: tuple[int, int, int, int] = (200, 1700, 200, 620),
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
                 detect_node_dilated: tuple=((3,3), 2),
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

    def extract_roi(self, image: np.ndarray, isShow: bool, size: tuple=(200, 1700, 200, 620)) -> np.ndarray:
        """
        Cắt và trích xuất vùng ROI từ ảnh dựa trên kích thước đã cho.

        Parameters:
            image (np.ndarray): Ảnh đầu vào.
            isShow (bool): Cờ xác định có hiển thị ROI hay không.
            size (tuple): Vùng cắt theo định dạng (x_start, x_end, y_start, y_end).

        Returns:
            np.ndarray: Ảnh cắt ra theo vùng ROI.
        """
        x_start, x_end, y_start, y_end = size
        rs = image[y_start:y_end, x_start:x_end]
        if isShow:
            cv2.imshow("ROI", rs)
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
        cv2.imshow("sharpened", claheImage)
        cv2.moveWindow("sharpened", 0, 0)

        #chuyển sang ảnh nhị phân
        _, thresh = cv2.threshold(claheImage, threshold, 255, cv2.THRESH_BINARY)

        if isShow:
            cv2.imshow("thresh", thresh)
            cv2.moveWindow("thresh", 0, 0)
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
        if isShow:
            cv2.imshow("undistort_image", undistorted_img)
            cv2.moveWindow(0,0)

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
        cv2.imshow("erode", erodeImage)
        cv2.moveWindow("erode", 0, 0)

        # Loại bỏ các chi tiết không đủ chiều ngang (opening)
        openedImage = cv2.morphologyEx(erodeImage, cv2.MORPH_OPEN, np.ones(opened, np.uint8))
        cv2.imshow("openedImage", openedImage)
        cv2.moveWindow("openedImage", 0, 0)
        # Thực hiện phép dilation, có thể thay đổi iterations để tăng mức độ nở
        dilatedImage = cv2.dilate(openedImage, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,dilated[0]), iterations=dilated[1] )

        if isShow:
            cv2.imshow("dilatedImage", dilatedImage)
            cv2.moveWindow("dilatedImage", 0, 0)
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
        
        if isShow:
            cv2.imshow("detect corners rgb", masked_image)
            cv2.moveWindow("detect corners rgb", 0, 0)

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
        if isShow:
            cv2.imshow("Centers", raw_image)
            cv2.moveWindow("Centers", 0, 0)
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
            group.sort(key=lambda p: p[0])

        # 4) Vẽ các điểm và đường nối trong từng nhóm
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

        if isShow:
            cv2.imshow("Grouped Points", raw_image)
            cv2.moveWindow("Grouped Points", 0, 0)
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

    def process(self, src, isLoadImg: bool = False, isShow: bool = False) -> tuple:
        """
        Thực hiện pipeline xử lý ảnh hoàn chỉnh gồm:
          1. Đọc ảnh từ file hoặc sử dụng ảnh đầu vào.
          1.1. Calibration ảnh
          2. Cắt vùng ROI.
          3. Tăng cường độ tương phản và chuyển sang ảnh nhị phân.
          4. Phát hiện các góc của lưới.
          5. Xác định tâm của các nút.
          6. Nhóm các tâm theo trục y.
          7. Lọc các hàng dựa trên số lượng nút.
          8. Kiểm tra lỗi về khoảng cách và góc giữa các điểm.

        Parameters:
            src: Đường dẫn file ảnh hoặc mảng ảnh.
            isLoadImg (bool): Nếu True, sẽ load ảnh từ file.
            isShow (bool): Nếu True, sẽ hiển thị các bước trung gian và kết quả cuối cùng.

        Returns:
            tuple: (Cờ báo lỗi (True nếu có lỗi), Ảnh kết quả cuối cùng đã vẽ chỉ báo lỗi).
        """
        self.distance_net_pixel_x = ImageProcessor.cm_to_pixel(self.distance_net_pixel_cm,self.distance_net_cm_x)
        self.distance_net_pixel_y = ImageProcessor.cm_to_pixel(self.distance_net_pixel_cm,self.distance_net_cm_y)
        # Bước 1: Đọc ảnh
        image = cv2.imread(src) if isLoadImg else src
        image = self.undistort_image(image, self.undistort_image_cameraMatrix, self.undistort_image_distCoeffs,isShow=False)
        # Bước 2: Cắt ROI
        roi = self.extract_roi(image, isShow=False, size=self.extract_roi_size)
        # Bước 3: Phát hiện cạnh
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
                                         expected_x_distance=self.distance_net_pixel_x,
                                         allowed_x_error=self.check_error_allowed_x_error,
                                         expected_angle=self.check_error_expected_angle,
                                         allowed_angle_error=self.check_error_allowed_angle_error,
                                         expected_y_distance=self.distance_net_pixel_y,
                                         allowed_y_error=self.check_error_allowed_y_error)
        if isShow:
            cv2.imshow("Final Result", final_rs)
            cv2.moveWindow("Final Result", 0, 0)
            print(f"x check: {self.distance_net_pixel_x} pixel\ty check: {self.distance_net_pixel_y} pixel\t angle check: {self.check_error_expected_angle} degree")
            print("Lỗi") if e else print("Không lỗi")
            print("---------------------------")

        return e, final_rs

    def processImgFolder(self, folderPath: str) -> None:
        """
        Duyệt qua các file ảnh trong thư mục, xử lý từng ảnh và hiển thị kết quả.

        Quy trình:
          - Kiểm tra sự tồn tại của thư mục.
          - Lấy danh sách các file ảnh có đuôi mở rộng cho phép.
          - Xử lý từng ảnh qua pipeline xử lý.
          - Chờ phím nhấn để chuyển sang ảnh tiếp theo hoặc thoát.

        Parameters:
            folderPath (str): Đường dẫn tới thư mục chứa ảnh.

        Returns:
            None.
        """
        if not os.path.isdir(folderPath):
            print(f"Lỗi: Thư mục '{folderPath}' không tồn tại!")
            return

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(folderPath) if f.lower().endswith(image_extensions)]
        if not image_files:
            print("Không tìm thấy ảnh trong thư mục!")
            return

        for img_name in image_files:
            img_path = os.path.join(folderPath, img_name)
            try:
                error, final_rs = self.process(img_path, isLoadImg=True, isShow=True)
            except ValueError as e:
                print(e)
                continue

            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                print("Thoát chương trình.")
                break
            cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = r"D:\DaiHoc\Intern\ThienPhuocCompany\data_fishNet"  # Thay đổi đường dẫn phù hợp
    processor = ImageProcessor()
    processor.processImgFolder(folder_path)