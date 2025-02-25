import cv2 
import math
points = []
img = cv2.imread(r"D:\DaiHoc\Intern\ThienPhuocCompany\data_fishNet\40cm\WIN_20250215_08_40_15_Pro.jpg")
def mouse_event(event, x, y, flags, param):
    global img, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Lưu tọa độ điểm được chọn

        if len(points) == 1:
            cv2.circle(img, points[0], 5, (0, 255, 0), -1)
        elif len(points) == 2:
            cv2.circle(img, points[1], 5, (0, 255, 0), -1)
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            cv2.putText(img, f"{distance:.2f} px", (mid_x, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            points.clear()  # Reset danh sách điểm

        cv2.imshow("Image", img)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindows()