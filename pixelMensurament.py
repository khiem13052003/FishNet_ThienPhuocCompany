import cv2
import numpy as np

# Danh sách lưu các điểm được click
points = []

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lưu tọa độ điểm vừa click
        points.append((x, y))
        # Vẽ một vòng tròn nhỏ tại điểm click
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        if len(points) == 2:
            # Vẽ đường nối giữa 2 điểm
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)
            # Tính khoảng cách Euclid giữa 2 điểm
            distance = round(np.linalg.norm(np.array(points[0]) - np.array(points[1])),2)
            # Tính tọa độ trung điểm để hiển thị khoảng cách
            mid_point = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            # Hiển thị khoảng cách trên ảnh
            cv2.putText(img, f"{distance} px", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Reset danh sách để đo lần sau
            points.clear()
        
        cv2.imshow("Image", img)

if __name__ == "__main__":
    # Đọc ảnh từ file; thay "sample.jpg" bằng đường dẫn tới ảnh của bạn
    img = cv2.imread(r"C:\Users\dinhk\OneDrive\Pictures\Camera Roll\WIN_20250310_11_44_44_Pro.jpg")
    if img is None:
        print("Không thể đọc ảnh!")
        exit()
    
    cv2.imshow("Image", img)
    # Đăng ký hàm callback cho cửa sổ "Image"
    cv2.setMouseCallback("Image", click_event)
    
    # Chương trình sẽ chạy cho đến khi nhấn phím bất kỳ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
