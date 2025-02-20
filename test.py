import cv2
import numpy as np
import os
from collections import defaultdict
import random
import math

class ImageProcessor:
    def __init__(self,
                 roi_coords: tuple = (0, 1280, 530, 720),
                 detect_corners_block_size: int = 7,
                 detect_corners_ksize: int = 3,
                 detect_corners_k: float = 0.09,
                 detect_corners_threshold_ratio: float = 0.01,
                 group_keypoints_dbscan_eps: float = 13,
                 group_keypoints_dbscan_min_samples: int = 60):
        """
        roi_coords: (x_start, x_end, y_start, y_end)
        Các tham số khác dùng cho Harris Corner Detection và DBSCAN clustering.
        """
        self.roi_coords = roi_coords
        self.detect_corners_block_size = detect_corners_block_size
        self.detect_corners_ksize = detect_corners_ksize
        self.detect_corners_k = detect_corners_k
        self.detect_corners_threshold_ratio = detect_corners_threshold_ratio
        self.group_keypoints_dbscan_eps = group_keypoints_dbscan_eps
        self.group_keypoints_dbscan_min_samples = group_keypoints_dbscan_min_samples

    def load_image(self, image_path: str) -> np.ndarray:
        """Tải ảnh từ đường dẫn."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể tải ảnh từ: {image_path}")
        return image

    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Cắt vùng ROI theo roi_coords."""
        x_start, x_end, y_start, y_end = self.roi_coords
        return image[y_start:y_end, x_start:x_end]

    def extract_edges(self, roi: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ROI:
        - Chuyển ảnh sang grayscale.
        - Phát hiện canh Canny
        """
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Áp dụng Gaussian Blur để giảm nhiễu (cải thiện hiệu quả Canny)
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 1.4)

        # Áp dụng Canny edge detection
        # Tham số thứ 2 và thứ 3 là ngưỡng thấp và ngưỡng cao, có thể điều chỉnh để phù hợp với ảnh của bạn.
        edges = cv2.Canny(blurred, 50, 255)
        # cv2.imshow("edges", edges)
        
        return edges

    def detect_corners(self, processed_img: np.ndarray) -> tuple:
        """
        Sử dụng thuật toán Harris để phát hiện góc và tạo mặt nạ.
        """
        corners = cv2.cornerHarris(np.float32(processed_img),
                                   self.detect_corners_block_size,
                                   self.detect_corners_ksize,
                                   self.detect_corners_k)
        threshold_value = self.detect_corners_threshold_ratio * corners.max()
        _, corner_mask = cv2.threshold(corners, threshold_value, 255, cv2.THRESH_BINARY)
        return corners, np.uint8(corner_mask)
    
    def gen_centers(self, binary_image: np.ndarray, raw_image: np.ndarray, min_area: int = 78) -> np.ndarray:
        """
        Nhận diện các nhóm điểm ảnh liền kề, tìm tâm của mỗi nhóm và tô màu lên ảnh.
        
        Parameters:
        - image: Ảnh nhị phân đầu vào (0 và 255).
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

        # cv2.imshow("result_img", raw_image)

        return raw_image, centers


    def group_points_by_y(self, points, raw_image, threshold=19):
        # Nếu không có điểm nào thì trả về kết quả rỗng
        if not points:
            return [], raw_image

        # Sắp xếp các điểm theo giá trị y
        points.sort(key=lambda p: p[1])
        
        groups = defaultdict(list)
        group_index = 0
        # Dùng giá trị y của điểm đầu tiên làm mốc cho nhóm đầu tiên
        current_group_y = points[0][1]
        
        for point in points:
            x, y = point
            # Nếu sự chênh lệch với mốc của nhóm hiện tại lớn hơn threshold,
            # tạo nhóm mới và cập nhật mốc
            if abs(y - current_group_y) > threshold:
                group_index += 1
                current_group_y = y
            groups[group_index].append((x, y))
        
        # Vẽ các điểm và nối các điểm trong cùng 1 nhóm
        for group_points in groups.values():
            # Sắp xếp các điểm trong nhóm theo trục x để nối theo thứ tự từ trái sang phải
            group_points.sort(key=lambda p: p[0])
            # Lấy màu ngẫu nhiên cho nhóm
            color = [random.randint(0,255) for x in range(3)]
            # Nối các điểm với nhau nếu nhóm có từ 2 điểm trở lên
            if len(group_points) > 1:
                for i in range(len(group_points) - 1):
                    pt1 = group_points[i]
                    pt2 = group_points[i+1]
                    cv2.line(raw_image, pt1, pt2, (234, 63, 247), 2)
            # Vẽ các điểm
            for (x, y) in group_points:
                cv2.circle(raw_image, (x, y), 5, color, -1)

        return list(groups.values()), raw_image
    def filter_rows(self, rows: list, nums_row: int = 2, nums_col: int = 1) -> list:
        """
        Loại bỏ nums_row hàng đầu và nums_row hàng cuối của danh sách rows,
        và với mỗi hàng, loại bỏ nums_col điểm đầu và nums_col điểm cuối.

        Args:
            rows (list): Danh sách các hàng, mỗi hàng là danh sách các điểm.
            nums_row (int, optional): Số hàng cần loại bỏ ở đầu và cuối. Mặc định là 2.
            nums_col (int, optional): Số điểm cần loại bỏ ở đầu và cuối mỗi hàng. Mặc định là 2.

        Returns:
            list: Danh sách các hàng sau khi lọc.
        """
        # Nếu tổng số hàng không đủ để loại bỏ cả hai đầu (trước và sau) thì trả về danh sách rỗng
        if not rows or len(rows) <= 2 * nums_row:
            return []
        
        # Loại bỏ nums_row hàng đầu và nums_row hàng cuối
        filtered_rows = rows[nums_row:-nums_row]
        
        # Với mỗi hàng, chỉ giữ lại những phần tử giữa nums_col đầu và nums_col cuối
        # Điều kiện len(row) > 2 * nums_col đảm bảo hàng có đủ số điểm để lọc
        result = [row[nums_col:-nums_col] for row in filtered_rows if len(row) > 2 * nums_col]
        
        return result

    def check_errors(self, rows: list, raw_image, allowed_x_threshold: int = 15, allowed_y_threshold: int = 8) -> any:
        """
        Kiểm tra lỗi trong các hàng điểm.
        - Nếu khoảng cách theo x vượt quá allowed_x_threshold: vẽ đường màu vàng.
        - Nếu góc lệch giữa đoạn thẳng nối 2 điểm và đường nằm ngang vượt quá allowed_y_threshold (độ): vẽ đường màu đỏ.
        - Nếu không lỗi, đường được vẽ màu xanh lá.
        
        Args:
            rows (list): Danh sách các hàng, mỗi hàng là danh sách các điểm (x, y).
            raw_image: Ảnh gốc dùng để vẽ đường.
            allowed_x_threshold (int, optional): Ngưỡng lỗi theo x (mặc định 15 pixel).
            allowed_y_threshold (int, optional): Ngưỡng lỗi theo góc lệch (độ, mặc định 8°).
        
        Returns:
            Ảnh raw_image đã được vẽ các đường báo lỗi.
        """
        for row in rows:
            for i in range(len(row) - 1):
                pt1 = row[i]
                pt2 = row[i + 1]
                
                diff_x = abs(pt2[0] - pt1[0])
                diff_y = abs(pt2[1] - pt1[1])
                
                # Mặc định không lỗi: màu xanh lá
                color = (0, 255, 0)
                
                # Kiểm tra lỗi theo trục x
                if diff_x > allowed_x_threshold:
                    color = (0, 255, 255)  # Vàng
                else:
                    # Tính góc lệch giữa đoạn thẳng nối pt1, pt2 và đường nằm ngang
                    # Nếu diff_x = 0, giả sử góc = 90° (hoặc bạn có thể xử lý khác)
                    if diff_x == 0:
                        angle_deg = 90
                    else:
                        angle_deg = math.degrees(math.atan(diff_y / diff_x))
                    
                    # Nếu góc lệch vượt quá ngưỡng, báo lỗi theo y: vẽ màu đỏ
                    if angle_deg > allowed_y_threshold:
                        color = (0, 0, 255)
                
                cv2.line(raw_image, pt1, pt2, color, 2)
        
        return raw_image

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
        # Bước 1: Đọc ảnh
        image = src if isLoadImg else self.load_image(src)
        cv2.imshow("Original Image", image)
        cv2.waitKey(0)

        # Bước 2: Cắt ROI
        roi = self.extract_roi(image)
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

        # Bước 3: Tiền xử lý và phát hiện cạnh
        edges = self.extract_edges(roi)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)

        # Bước 4: Phát hiện góc với Harris và tạo mặt nạ
        corners, corner_mask = self.detect_corners(edges)
        cv2.imshow("Corner Mask", corner_mask)
        cv2.waitKey(0)

        # Bước 5: Xác định các nút lưới và vẽ tâm lên ảnh
        center_image, centers = self.gen_centers(corner_mask, roi)
        cv2.imshow("Centers", center_image)
        cv2.waitKey(0)

        # Bước 6: Nhóm các điểm theo trục y và nối các điểm trong nhóm
        gr, gr_image = self.group_points_by_y(centers, roi)
        cv2.imshow("Grouped Points", gr_image)
        cv2.waitKey(0)

        # Bước 7: Lọc các hàng theo yêu cầu
        filter_gr = self.filter_rows(gr)

        # Bước 8: Kiểm tra lỗi và vẽ kết quả cuối cùng
        final_rs = self.check_errors(filter_gr, gr_image, allowed_x_threshold=32, allowed_y_threshold=12)
        cv2.imshow("Final Result", final_rs)
        cv2.waitKey(0)

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
                image = self.load_image(img_path)
            except ValueError as e:
                print(e)
                continue

            image, processed_roi, corner_mask, final_rs = self.process(image, isLoadImg=True, isShow=True)
            # cv2.imshow("final_rs", final_rs)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                print("Thoát chương trình.")
                break
            cv2.destroyAllWindows()

    def realTime(self) -> None:
        """
        Xử lý thời gian thực từ camera.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Lỗi: Không mở được camera!")
            return

        self.roi_coords = (0, 1280, 475, 720)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi: Không lấy được frame từ camera!")
                break
            frame = cv2.resize(frame, (1280, 720))
            image, processed_roi, corner_mask, final_rs = self.process(frame, isLoadImg=True)
            cv2.imshow("final_rs", final_rs)
            cv2.imshow("raw_image",image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = r"C:\Users\dinhk\OneDrive\Pictures\Camera Roll\60_thieu_cm"  # Thay đổi đường dẫn phù hợp
    processor = ImageProcessor()
    processor.processImgFolder(folder_path)
    # processor.realTime()
