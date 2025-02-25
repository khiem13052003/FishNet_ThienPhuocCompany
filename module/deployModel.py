import cv2
import numpy as np
import os
from collections import defaultdict
import random
import math

class ImageProcessor:
    def __init__(self,
                 roi_coords: tuple = (0, 1920, 700, 1000),
                 distance_2node: float = 1.4,
                 detect_corners_block_size: int = 5,  # càng giảm càng nhỏ
                 detect_corners_ksize: int = 3,         # càng giảm vùng càng ngắn
                 detect_corners_k: float = 0.1,         # càng giảm vùng càng lớn
                 detect_corners_threshold_ratio: float = 0.07,
                 detect_corners_dilated_ksize: tuple = (3,3),
                 detect_corners_closed_ksize: tuple = (3,3),
                 gen_centers_min_area: float = 70.0,
                 group_points_by_y_threshold: float = 4.0,
                 check_error_expected_x_distance: float = 32.0, check_error_allowed_x_error: float = 3.0,
                 check_error_expected_angle: float = 0.0, check_error_allowed_angle_error: float = 8.0,
                 check_error_expected_y_distance: float = 30.0, check_error_allowed_y_error: float = 8.0):
        """
        Khởi tạo các tham số cho xử lý ảnh.
        """
        self.roi_coords = roi_coords
        self.distance_2node = distance_2node
        self.detect_corners_block_size = detect_corners_block_size
        self.detect_corners_ksize = detect_corners_ksize
        self.detect_corners_k = detect_corners_k
        self.detect_corners_threshold_ratio = detect_corners_threshold_ratio
        self.detect_corners_dilated_ksize = detect_corners_dilated_ksize
        self.detect_corners_closed_ksize = detect_corners_closed_ksize
        self.group_points_by_y_threshold = group_points_by_y_threshold
        self.gen_centers_min_area = gen_centers_min_area
        self.check_error_expected_x_distance = check_error_expected_x_distance
        self.check_error_allowed_x_error = check_error_allowed_x_error
        self.check_error_expected_angle = check_error_expected_angle
        self.check_error_allowed_angle_error = check_error_allowed_angle_error
        self.check_error_expected_y_distance = check_error_expected_y_distance
        self.check_error_allowed_y_error = check_error_allowed_y_error

    @staticmethod
    def set_resolution(cap: cv2.VideoCapture, list_reso: tuple) -> None:
        # Đặt độ phân giải theo tuple (width, height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, list_reso[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, list_reso[1])

    def cm_to_pixels(self, cm: float, distance_cm: float, distance_pixel: float) -> float:
        """
        Quy đổi cm sang pixel dựa trên số pixel trên mỗi cm.
        """
        pixels_per_cm = distance_pixel / distance_cm 
        pixels = cm * pixels_per_cm
        return pixels

    def extract_roi(self, image: np.ndarray, isShow: bool) -> np.ndarray:
        """Cắt vùng ROI theo roi_coords."""
        x_start, x_end, y_start, y_end = self.roi_coords
        rs = image[y_start:y_end, x_start:x_end]
        if isShow:
            cv2.imshow("ROI", rs)
        return rs

    def extract_edges(self, roi: np.ndarray, isShow: bool) -> np.ndarray:
        """
        Tiền xử lý ROI:
          - Chuyển ảnh sang grayscale.
          - Áp dụng Gaussian Blur.
          - Phát hiện cạnh bằng Canny.
        """
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 50, 255)
        if isShow:
            cv2.imshow("Edges", edges)
        return edges

    def detect_corners(self, edges: np.ndarray, isShow: bool) -> np.ndarray:
        """
        Phát hiện góc bằng thuật toán Harris và tạo mặt nạ.
        """
        corners = cv2.cornerHarris(np.float32(edges),
                                   self.detect_corners_block_size,
                                   self.detect_corners_ksize,
                                   self.detect_corners_k)
        threshold_value = self.detect_corners_threshold_ratio * corners.max()
        _, corner_mask = cv2.threshold(corners, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Dilation: sử dụng kernel riêng cho bước này
        kernel_dilate = np.ones(self.detect_corners_dilated_ksize, np.uint8)
        corner_mask_dilated = cv2.dilate(corner_mask, kernel_dilate)
        
        # Morphological closing: sử dụng kernel từ detect_corners_closed_ksize
        kernel_closed = np.ones(self.detect_corners_closed_ksize, np.uint8)
        closed_img = cv2.morphologyEx(corner_mask_dilated, cv2.MORPH_CLOSE, kernel_closed, iterations=3)
        
        rs = np.uint8(closed_img)
        if isShow:
            cv2.imshow("Corner Mask", rs)
        return rs

    def draw_mask_on_image(self, image: np.ndarray, mask: np.ndarray, isShow: bool, color=(0, 0, 255), alpha=1):
        """
        Vẽ mask trực tiếp lên ảnh với màu và độ trong suốt cho trước.
        """
        # Chuyển mask về dạng nhị phân (0 hoặc 1)
        mask = (mask > 0).astype(np.uint8)
        
        mask_colored = np.zeros_like(image, dtype=np.uint8)
        mask_colored[mask == 1] = color
        masked_image = cv2.addWeighted(image, 1, mask_colored, alpha, 0, dtype=cv2.CV_8U)
        
        if isShow:
            cv2.imshow("detect corners", masked_image)

        return masked_image

    def gen_centers(self, binary_image: np.ndarray, raw_image: np.ndarray, min_area: float, isShow: bool) -> tuple:
        """
        Nhận diện các nhóm điểm ảnh liền kề và tính tâm của mỗi nhóm.
        Trả về: (danh sách tâm, ảnh đã vẽ các điểm tâm)
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
        return centers, raw_image

    def group_points_by_y(self, points: list, raw_image: np.ndarray, threshold: float, isShow: bool) -> tuple:
        """
        Nhóm các điểm theo trục y và vẽ các đường nối các điểm trong cùng nhóm.
        Trả về: (danh sách các nhóm điểm, ảnh đã vẽ)
        """
        if not points:
            return [], raw_image

        points.sort(key=lambda p: p[1])
        groups = defaultdict(list)
        group_index = 0
        current_group_y = points[0][1]
        current_group_x = points[0][0]

        for point in points:
            x, y = point
            tempY = y - current_group_y
            tempX = x - current_group_x

            if tempY > threshold:
                group_index += 1

            current_group_y = y
            current_group_x = x
            groups[group_index].append((x, y))

            if tempX < 0 and len(groups[group_index]) >= 2:
                groups[group_index][-2], groups[group_index][-1] = groups[group_index][-1], groups[group_index][-2]

        for group_points in groups.values():
            group_points.sort(key=lambda p: p[0])
            color = [random.randint(0, 255) for _ in range(3)]
            if len(group_points) > 1:
                for i in range(len(group_points) - 1):
                    pt1 = group_points[i]
                    pt2 = group_points[i+1]
                    cv2.line(raw_image, pt1, pt2, (234, 63, 247), 2)
            for (x, y) in group_points:
                cv2.circle(raw_image, (x, y), 5, color, -1)

        if isShow:
            cv2.imshow("Grouped Points", raw_image)

        return list(groups.values()), raw_image

    def filter_rows(self, matrix: list, threshold: float = 0.5) -> list:
        """
        Lọc các hàng theo số lượng nút và lấy ra hàng ở giữa cùng với các hàng lân cận.
        """
        if not matrix:
            return []

        avg_nodes = sum(len(row) for row in matrix) / len(matrix)
        filtered_rows = [row for row in matrix if len(row) >= threshold * avg_nodes]

        if not filtered_rows:
            return []

        n = len(filtered_rows)
        mid_index = n // 2 - 1 if n % 2 == 0 else n // 2
        start_index = max(0, mid_index - 1)
        end_index = min(n, mid_index + 2)
        return filtered_rows[start_index:end_index]

    def check_errors(self, matrix: list, raw_image: np.ndarray,
                     expected_x_distance: float, allowed_x_error: float,
                     expected_angle: float, allowed_angle_error: float,
                     expected_y_distance: float, allowed_y_error: float) -> tuple:
        """
        Kiểm tra các lỗi về khoảng cách và góc giữa các điểm, vẽ các đường lỗi lên ảnh.
        Trả về: (ảnh đã vẽ, flag báo lỗi)
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
                error = True

        return raw_image, error

    def process(self, src, isLoadImg: bool = False, isShow: bool = False) -> tuple:
        """
        Pipeline xử lý ảnh:
          1. Đọc ảnh.
          2. Cắt ROI.
          3. Phát hiện cạnh.
          4. Phát hiện góc.
          5. Xác định tâm các nút.
          6. Nhóm các điểm.
          7. Lọc các hàng và kiểm tra lỗi.
        Trả về: (ảnh gốc, edges, corner_mask, kết quả cuối cùng)
        """
        # Bước 1: Đọc ảnh
        image = cv2.imread(src) if isLoadImg else src
        # Bước 2: Cắt ROI
        roi = self.extract_roi(image, isShow=False)
        # Bước 3: Phát hiện cạnh
        edges = self.extract_edges(roi, isShow=False)
        # Bước 4: Phát hiện góc và tạo mặt nạ
        corner_mask = self.detect_corners(edges, isShow=False)
        self.draw_mask_on_image(roi, corner_mask, isShow=False)
        # Bước 5: Xác định các nút lưới
        centers, center_image = self.gen_centers(corner_mask, roi, self.gen_centers_min_area, isShow=False)
        # Bước 6: Nhóm các điểm theo trục y
        gr, gr_image = self.group_points_by_y(centers, roi, self.group_points_by_y_threshold, isShow=False)
        # Bước 7: Lọc các hàng
        filter_gr = self.filter_rows(gr)
        # Bước 8: Kiểm tra lỗi
        final_rs, e = self.check_errors(filter_gr, gr_image,
                                         self.check_error_expected_x_distance,
                                         self.check_error_allowed_x_error,
                                         self.check_error_expected_angle,
                                         self.check_error_allowed_angle_error,
                                         self.check_error_expected_y_distance,
                                         self.check_error_allowed_y_error)
        if isShow:
            cv2.imshow("Final Result", final_rs)
            print("Lỗi") if e else print("Không lỗi")

        return e, final_rs

    def processImgFolder(self, folderPath: str) -> None:
        """
        Duyệt qua các file ảnh trong folder, xử lý và hiển thị kết quả.
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

    def realTime(self) -> None:
        """
        Xử lý thời gian thực từ hai camera với độ phân giải 1920x1080.
        """
        cap1 = cv2.VideoCapture(0)  # Mở camera 0
        # cap2 = cv2.VideoCapture(1)  # Mở camera 1

        if not cap1.isOpened():
            print("Lỗi: Không mở được một trong hai camera!")
            cap1.release()
            # cap2.release()
            return

        ImageProcessor.set_resolution(cap1, (1920, 1080))
        # ImageProcessor.set_resolution(cap2, (1920, 1080))

        try:
            while True:
                ret1, frame1 = cap1.read()
                # ret2, frame2 = cap2.read()

                if not ret1 or frame1 is None or frame1.size == 0:
                    print("Lỗi: Không lấy được frame hợp lệ từ camera 1!")
                else:
                    try:
                        error, final_rs1 = self.process(frame1, isLoadImg=False, isShow=False)
                        cv2.imshow("1", final_rs1)
                    except Exception as e:
                        print("Lỗi khi xử lý frame từ camera 1:", e)

                # if not ret2 or frame2 is None or frame2.size == 0:
                #     print("Lỗi: Không lấy được frame hợp lệ từ camera 2!")
                # else:
                #     try:
                #         error, final_rs2 = self.process(frame2, isLoadImg=False, isShow=False)
                #         cv2.imshow("2", final_rs2)
                #     except Exception as e:
                #         print("Lỗi khi xử lý frame từ camera 2:", e)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Đã dừng xử lý do nhận lệnh KeyboardInterrupt.")
        finally:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
