# This is a new Python file. You can start coding here.
import matplotlib.pyplot as plt

# Danh sách các điểm, mỗi điểm dạng (x, y)
points = [(2, 5), (1, 4), (1, 2), (2, 1), (1, 6), (2, 3)]

# Sắp xếp theo thứ tự tăng dần của x, nếu x bằng nhau thì sắp xếp theo y tăng dần.
points_sorted = sorted(points, key=lambda point: (point[1], point[0]))
print("Các điểm sau khi sắp xếp:")
print(points_sorted)

# Tách các giá trị x và y để vẽ đường nối
x_vals = [p[0] for p in points_sorted]
y_vals = [p[1] for p in points_sorted]

# Tạo hình vẽ
plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')

# Vẽ nhãn cho từng điểm để dễ theo dõi (tùy chọn)
for i, (x, y) in enumerate(points_sorted):
    plt.text(x, y, f"  {i}", fontsize=12, color='red')

plt.title("Đường nối các điểm sau khi sắp xếp")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
