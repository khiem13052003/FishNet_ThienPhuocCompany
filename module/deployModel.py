import cv2
import numpy as np
import os
from collections import defaultdict
import random
import math
points =[]

class ImageProcessor:
    def __init__(self,
                 distance_2node: int =1.4,
                 roi_coords: tuple = (0, 1920, 700, 1000),
                 detect_corners_block_size: int = 5, #càng giảm càng nhỏ
                 detect_corners_ksize: int = 3, #cang giảm vùng càng ngắn.
                 detect_corners_k: float = 0.1, #càng giảm vùng càng lớn
                 detect_corners_threshold_ratio: float = 0.07,
                 detect_corners_dilated_ksize: tuple = (3,3),
                 detect_corners_closed_ksize: tuple = (3,3),
                 gen_centers_min_area: int = 70,
                 group_points_by_y_threshold: int = 4,
                 check_error_expected_x_distance: int = 32, check_error_allowed_x_error: int = 3,
                 check_error_expected_angle: int = 0, check_error_allowed_angle_error: int = 8,
                 check_error_expected_y_distance: int = 30, check_error_allowed_y_error: int = 8):
        """
        roi_coords: (x_start, x_end, y_start, y_end)
        distance_2node
        Các tham số khác dùng cho Harris Corner Detection và DBSCAN clustering.
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
    # Hàm xử lý sự kiện chuột
    def mouse_event(event, x, y, flags, param):
        global img, points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))  # Lưu tọa độ điểm được chọn

            if len(points) == 1:
                # Vẽ điểm đầu tiên
                cv2.circle(img, points[0], 5, (0, 255, 0), -1)
            
            elif len(points) == 2:
                # Vẽ điểm thứ hai
                cv2.circle(img, points[1], 5, (0, 255, 0), -1)

                # Vẽ đường thẳng nối hai điểm
                cv2.line(img, points[0], points[1], (255, 0, 0), 2)

                # Tính khoảng cách
                distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)

                # Hiển thị khoảng cách lên ảnh
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                cv2.putText(img, f"{distance:.2f} px", (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Reset để chọn điểm mới
                points = []

            cv2.imshow("Image", img)


    def cm_to_pixels(self, cm, distance_cm, distance_pixel):
        """
        Quy đổi cm sang pixel dựa trên thông số camera và khoảng cách.
        
        Tham số:
        - cm: Kích thước thực tế (cm) cần quy đổi.
        - focal_length: Tiêu cự của camera (mm).
        - sensor_width: Chiều rộng cảm biến (mm).
        - image_width: Chiều rộng ảnh (pixels).
        - distance: Khoảng cách từ camera đến vật thể (cm).

        Trả về:
        - Giá trị tương ứng bằng pixels.
        """
        # Tính số pixels trên mỗi cm
        pixels_per_cm =  distance_pixel / distance_cm 
        
        # Quy đổi cm sang pixel
        pixels = cm * pixels_per_cm
        return pixels


    def load_image(self, image_path: str, isShow: bool) -> np.ndarray:
        """Tải ảnh từ đường dẫn."""
        image = cv2.imread(image_path)
        if isShow:
            cv2.imshow("Original Image", image)
        return image

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
          - (Có thể thêm CLAHE, threshold, morphology nếu cần)
        """
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Áp dụng Gaussian Blur để giảm nhiễu (cải thiện hiệu quả Canny)
        blurred = cv2.GaussianBlur(gray_roi, (3,3), 1.4)

        # Áp dụng Canny edge detection
        # Tham số thứ 2 và thứ 3 là ngưỡng thấp và ngưỡng cao, có thể điều chỉnh để phù hợp với ảnh của bạn.
        edges = cv2.Canny(blurred, 50, 255)
        if isShow:
            cv2.imshow("Edges", edges)
        return edges

    def detect_corners(self, edges: np.ndarray, isShow: bool) -> np.ndarray:
        """
        Sử dụng thuật toán Harris để phát hiện góc và tạo mặt nạ.
        """
        corners = cv2.cornerHarris(np.float32(edges),
                                   self.detect_corners_block_size,
                                   self.detect_corners_ksize,
                                   self.detect_corners_k)
        threshold_value = self.detect_corners_threshold_ratio * corners.max()
        _, corner_mask = cv2.threshold(corners, threshold_value, 255, cv2.THRESH_BINARY)
        # Mở rộng nhóm điểm góc bằng phép giãn nở
        kernel = np.ones(self.detect_corners_dilated_ksize, np.uint8)
        corner_mask_dilated = cv2.dilate(corner_mask, kernel)
        # Tạo kernel cho phép toán đóng
        kernel = np.ones(self.detect_corners_dilated_ksize, np.uint8)
        # Áp dụng morphological closing để lấp đầy các vùng trắng
        closed_img = cv2.morphologyEx(corner_mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
        rs = np.uint8(closed_img)
        if isShow:
            cv2.imshow("Corner Mask", rs)
        return np.uint8(rs)
    def draw_mask_on_image(self, image: np.ndarray, mask: np.ndarray, isShow: bool, color=(0, 0, 255), alpha=1):
        """
        Vẽ mask trực tiếp lên ảnh với màu mong muốn.

        Args:
            image (np.ndarray): Ảnh gốc (dạng BGR).
            mask (np.ndarray): Ảnh mask dạng numpy array (giá trị 0 hoặc 1, hoặc 0-255).
            color (tuple, optional): Màu vẽ mask (mặc định là đỏ - BGR: (0, 0, 255)).
            alpha (float, optional): Độ trong suốt của mask (0: trong suốt, 1: che phủ hoàn toàn).

        Returns:
            np.ndarray: Ảnh đã được vẽ mask.
        """
        # Đảm bảo mask ở dạng nhị phân 0 hoặc 1
        if mask.max() > 1:
            mask = mask / 255  # Chuyển về 0 hoặc 1 nếu mask ở dạng 0-255

        # Tạo ảnh màu của mask
        mask_colored = np.zeros_like(image, dtype=np.uint8)
        mask_colored[mask == 1] = color  # Gán màu mong muốn

        # Chỉ áp dụng lên vùng có mask
        masked_image = cv2.addWeighted(image, 1, mask_colored, alpha, 0, dtype=cv2.CV_8U)
        if isShow:
            cv2.imshow("detect corners", masked_image)

        return masked_image
    

    def gen_centers(self, binary_image: np.ndarray, raw_image: np.ndarray, min_area: int, isShow: bool) -> np.ndarray:

            """
            Nhận diện các nhóm điểm ảnh liền kề, tìm tâm của mỗi nhóm và tô màu lên ảnh.
            
            Parameters:
            - binary_image: Ảnh nhị phân đầu vào (0 và 255).
            - raw_image: ảnh ban đầu
            - min_area: Ngưỡng diện tích tối thiểu để giữ lại nhóm.

            Returns:
            - result_img: Ảnh đã tô màu các nhóm điểm ảnh.x
            - centers: Danh sách tọa độ tâm của các nhóm [(x1, y1), (x2, y2), ...].
            """
            # Tìm các thành phần kết nối
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

            centers = []  # Lưu danh sách tâm của các nhóm

            for i in range(1, num_labels):  # Bỏ qua nhãn 0 (background)
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area >= min_area:  # Chỉ xử lý nhóm có diện tích đủ lớn
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])  # Tọa độ tâm
                    centers.append((cx, cy))
                    
                    # Vẽ tâm bằng hình tròn xanh
                    cv2.circle(raw_image, (cx, cy), 5, (0, 0, 255), -1)
            if isShow:
                cv2.imshow("Centers", raw_image)
            return centers,raw_image

    def group_points_by_y(self, points: list, raw_image: np.ndarray, threshold: int, isShow: bool) -> list:
        # Nếu không có điểm nào thì trả về kết quả rỗng
        if not points:
            return [], raw_image

        # Sắp xếp các điểm theo giá trị y (tăng dần)
        points.sort(key=lambda p: p[1])
        
        groups = defaultdict(list)
        group_index = 0
        # Sử dụng điểm đầu tiên làm mốc cho nhóm đầu tiên
        current_group_y = points[0][1]
        current_group_x = points[0][0]
        
        for point in points:
            x, y = point
            # Tính hiệu số so với mốc của nhóm hiện tại
            tempY = y - current_group_y
            tempX = x - current_group_x
            
            # Nếu sự chênh lệch y vượt quá threshold, tạo nhóm mới
            if tempY > threshold:
                group_index += 1
            
            # Cập nhật mốc cho nhóm hiện tại (dù có tạo nhóm mới hay không)
            current_group_y = y
            current_group_x = x
            
            # Thêm điểm vào nhóm hiện tại
            groups[group_index].append((x, y))
            
            # Nếu điểm mới có hoành độ nhỏ hơn so với điểm trước (trong cùng nhóm)
            # và nhóm có đủ 2 phần tử thì hoán đổi 2 phần tử cuối để đảm bảo thứ tự
            if tempX < 0 and len(groups[group_index]) >= 2:
                groups[group_index][-2], groups[group_index][-1] = groups[group_index][-1], groups[group_index][-2]

        # Vẽ các điểm và nối các điểm trong cùng 1 nhóm
        for group_points in groups.values():
            # Sắp xếp các điểm trong nhóm theo trục x (từ trái sang phải)
            group_points.sort(key=lambda p: p[0])
            # Sử dụng màu cố định hoặc thay đổi nếu cần
            color = (0, 255, 0)
            # Nếu nhóm có từ 2 điểm trở lên thì vẽ đường nối
            if len(group_points) > 1:
                for i in range(len(group_points) - 1):
                    pt1 = group_points[i]
                    pt2 = group_points[i+1]
                    cv2.line(raw_image, pt1, pt2, (234, 63, 247), 2)
            # Vẽ từng điểm
            for (x, y) in group_points:
                cv2.circle(raw_image, (x, y), 5, color, -1)
        
        if isShow:
            cv2.imshow("Grouped Points", raw_image)
        
        return list(groups.values()), raw_image
    def filter_rows(self, matrix: list, threshold: int=0.5) -> list:
        """
        Chọn ra các hàng dựa trên:
        - Lọc bỏ các hàng có số nút ít hơn threshold * (số nút trung bình).
        - Xác định hàng ở giữa (nếu số hàng chẵn thì lấy hàng ở vị trí (n/2 - 1)).
        - Trả về hàng ở giữa cùng với 2 hàng trước và 2 hàng sau (nếu có).
        
        :param matrix: Danh sách các hàng, mỗi hàng là danh sách các điểm (x,y).
        :param threshold: Ngưỡng tỷ lệ so với trung bình (mặc định 0.5).
        :return: Danh sách các hàng đã được chọn.
        """
        # Nếu matrix rỗng, trả về danh sách rỗng.
        if not matrix:
            return []
        
        # Tính số nút trung bình trên mỗi hàng
        avg_nodes = sum(len(row) for row in matrix) / len(matrix)
        
        # Lọc bỏ các hàng có số nút ít hơn threshold * avg_nodes
        filtered_rows = [row for row in matrix if len(row) >= threshold * avg_nodes]
        
        # Nếu không còn hàng nào sau khi lọc, trả về danh sách rỗng.
        if not filtered_rows:
            return []
        
        n = len(filtered_rows)
        
        # Xác định chỉ số hàng giữa:
        # Nếu số hàng là số chẵn thì lấy hàng ở vị trí (n/2 - 1),
        # nếu số hàng là số lẻ thì lấy hàng ở vị trí n//2.
        if n % 2 == 0:
            mid_index = n // 2 - 1
        else:
            mid_index = n // 2
        
        # Xác định khoảng các hàng cần lấy: 2 hàng trước và 2 hàng sau hàng giữa.
        start_index = max(0, mid_index - 1)
        end_index = min(n, mid_index + 2)  # +3 vì slice [start, end) không lấy end
        
        # Trả về danh sách các hàng được chọn
        return filtered_rows[start_index:end_index]

    def check_errors(self, matrix: list, raw_image: np.ndarray,
                                expected_x_distance: int, allowed_x_error: int,
                                expected_angle: int, allowed_angle_error: int,
                                expected_y_distance: int, allowed_y_error: int) -> list:
        """
        Kiểm tra lỗi và vẽ các đường lỗi lên ảnh bằng OpenCV.
        
        Tham số:
        - raw_image: Ảnh gốc (numpy array) cần vẽ lên.
        - matrix: Danh sách các hàng, mỗi hàng là danh sách các điểm (x, y).
                    Giả sử các điểm trong mỗi hàng đã được sắp theo thứ tự x tăng dần.
        - expected_x_distance: Khoảng cách mong muốn giữa 2 nút liền kề theo x (mặc định 37 pixel).
        - allowed_x_error: Sai số cho phép theo x (mặc định ±3 pixel).
        - expected_angle: Góc mong muốn so với đường ngang (mặc định 0°).
        - allowed_angle_error: Sai số cho phép của góc (mặc định ±4°).
        - expected_y_distance: Khoảng cách mong muốn giữa trung bình y của 2 hàng liền kề (mặc định 15 pixel).
        - allowed_y_error: Sai số cho phép theo y (mặc định ±3 pixel).
        
        Kết quả: Ảnh raw_image với các đường lỗi được vẽ lên.
        """
        error = False
        # Xác định khoảng cho phép theo x và góc
        allowed_x_min = expected_x_distance - allowed_x_error
        allowed_x_max = expected_x_distance + allowed_x_error
        allowed_angle_min = expected_angle - allowed_angle_error
        allowed_angle_max = expected_angle + allowed_angle_error

        # Định nghĩa màu cho các lỗi (theo BGR)
        color_good = (0,255,0)
        color_error_x = (0, 0, 255)    # Đỏ cho lỗi kích thước x
        color_error_degree = (0, 255, 255)  # Vàng cho lỗi góc lệch
        color_error_y = (0, 165, 255)  # Cam cho lỗi kích thước y

        # --- Kiểm tra từng hàng: khoảng cách x và góc lệch ---
        for row in matrix:
            for i in range(len(row) - 1):
                p1 = row[i]
                p2 = row[i+1]
                # Tính khoảng cách theo x và hiệu số y
                dx = abs(p2[0] - p1[0])
                dy = p2[1] - p1[1]
                angle = math.degrees(math.atan2(dy, dx))
                
                # Kiểm tra lỗi kích thước x (Lỗi 1)
                if dx < allowed_x_min or dx > allowed_x_max:
                    cv2.line(raw_image, p1, p2, color_error_x, 2)
                    error = True
                # Kiểm tra lỗi góc lệch (Lỗi 2)
                elif angle < allowed_angle_min or angle > allowed_angle_max:
                    cv2.line(raw_image, p1, p2, color_error_degree, 2)
                    error = True
                else:
                    cv2.line(raw_image, p1, p2, color_good, 2)

        
        # --- Tính điểm trung bình (đại diện) cho mỗi hàng ---
        avg_points = []
        for row in matrix:
            if row:
                avg_x = int(sum(p[0] for p in row) / len(row))
                avg_y = int(sum(p[1] for p in row) / len(row))
                avg_points.append((avg_x, avg_y))
        
        # --- Kiểm tra lỗi kích thước y giữa các hàng (Lỗi 3) ---
        for i in range(len(avg_points) - 1):
            pt1 = avg_points[i]
            pt2 = avg_points[i+1]
            dy = abs(pt2[1] - pt1[1])
            if dy < (expected_y_distance - allowed_y_error) or dy > (expected_y_distance + allowed_y_error):
                cv2.line(raw_image, pt1, pt2, color_error_y, 2)
                error = True
        
        return raw_image, error

    def process(self, src, isLoadImg: bool = False, isShow: bool=False) -> tuple:
        """
        Pipeline xử lý ảnh với hiển thị từng bước:
        1. Tải ảnh (nếu src là đường dẫn).
        2. Hiển thị ảnh gốc.
        3. Cắt ROI và hiển thị.
        4. Tiền xử lý ROI và phát hiện cạnh, hiển thị.
        5. Phát hiện góc bằng Harris, hiển thị mặt nạ góc.
        6. Xác định các nút lưới (center) và hiển thị.
        7. Nhóm các điểm theo trục y, nối các điểm trong nhóm và hiển thị.
        8. Lọc các hàng và kiểm tra lỗi, hiển thị kết quả cuối cùng.
        
        Sau mỗi bước, cửa sổ sẽ chờ nhấn phím (cv2.waitKey(0)) để bạn có thể quan sát hình ảnh.
        
        Trả về: (original image, edges, corner_mask, final result)
        """
        #Bước 0: Tính toán khoảng cách giữa 2 nút:
        distance = self.cm_to_pixels(self.distance_2node, 1.4, 30)
        # Bước 1: Đọc ảnh
        image = src if isLoadImg else self.load_image(src, isShow=True)

        # Bước 2: Cắt ROI
        roi = self.extract_roi(image, isShow=False)

        # Bước 3: Tiền xử lý và phát hiện cạnh
        edges = self.extract_edges(roi, isShow=False)

        # Bước 4: Phát hiện góc với Harris và tạo mặt nạ
        corner_mask = self.detect_corners(edges, isShow=False)
        self.draw_mask_on_image(roi, corner_mask, isShow=True)

        # Bước 5: Xác định các nút lưới và vẽ tâm lên ảnh
        centers, center_image = self.gen_centers(corner_mask, roi, self.gen_centers_min_area, isShow=False)

        # Bước 6: Nhóm các điểm theo trục y và nối các điểm trong nhóm
        gr, gr_image = self.group_points_by_y(centers, roi, self.group_points_by_y_threshold, isShow=False)

        # Bước 7: Lọc các hàng theo yêu cầu
        filter_gr = self.filter_rows(gr)

        # Bước 8: Kiểm tra lỗi và vẽ kết quả cuối cùng
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

        return image, edges, corner_mask, final_rs

    def processImgFolder(self, folderPath: str) -> None:
        """
        Duyệt qua tất cả các file ảnh trong folder, xử lý từng ảnh và hiển thị kết quả.
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
                image, processed_roi, corner_mask, final_rs = self.process(img_path, isLoadImg=False, isShow=True)
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
        Xử lý thời gian thực từ camera với độ phân giải 1920x1080.
        """
        cap = cv2.VideoCapture(0)  # Mở camera mặc định

        if not cap.isOpened():
            print("Lỗi: Không mở được camera!")
            return

        # Đặt độ phân giải 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("Lỗi: Không lấy được frame hợp lệ từ camera!")
                    continue  # Bỏ qua frame lỗi và tiếp tục lấy frame mới

                try:
                    image, processed_roi, corner_mask, final_rs = self.process(
                        frame, isLoadImg=True, isShow=True
                    )
                except Exception as e:
                    print("Lỗi khi xử lý frame:", e)
                    continue  # Bỏ qua frame lỗi và tiếp tục lấy frame mới

                cv2.imshow("image", image)

                # Nhấn 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Đã dừng xử lý do nhận lệnh KeyboardInterrupt.")
        finally:
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    folder_path = r"C:\Users\dinhk\OneDrive\Pictures\Camera Roll\58cm_moi"  # Thay đổi đường dẫn phù hợp
    processor = ImageProcessor()
    # processor.processImgFolder(folder_path)
    processor.realTime()  

    # # Đọc ảnh
    # img = cv2.imread(r"C:\Users\dinhk\OneDrive\Pictures\Camera Roll\58cm_moi\WIN_20250219_14_04_58_Pro.jpg")

    # cv2.imshow("Image", img)
    # cv2.setMouseCallback("Image", ImageProcessor.mouse_event)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
