import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu đầu vào
X_original = np.array([
    [190, 173, 182, 181, 162, 164, 180, 155, 171, 170, 181, 182, 189, 184, 209, 210]
]).T
y = np.array([59, 61, 59, 55, 53, 54, 52, 51, 63, 64, 76, 69, 66, 72, 69, 80])

# Chuẩn hóa dữ liệu
X = (X_original - np.mean(X_original)) / np.std(X_original)

# Thêm cột 1 vào X để tính hệ số chặn
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Khởi tạo các tham số
theta = np.zeros(2)
m = len(y)
alpha = 0.01  # Tốc độ học
num_iterations = 1000

# Hàm tính giá trị dự đoán
def predict(X, theta):
    return np.dot(X, theta)

# Hàm tính cost
def compute_cost(X, y, theta):
    predictions = predict(X, theta)
    return np.sum((predictions - y) ** 2) / (2 * m)

# Gradient descent
for _ in range(num_iterations):
    predictions = predict(X, theta)
    theta = theta - (alpha / m) * np.dot(X.T, (predictions - y))

# In kết quả
print("Hệ số hồi quy:")
print(f"Hệ số chặn (theta_0): {theta[0]:.4f}")
print(f"Hệ số góc (theta_1): {theta[1]:.4f}")

# Tính giá trị dự đoán
y_pred = predict(X, theta)

# Tính R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nR-squared: {r_squared:.4f}")

# Tính MSE
mse = np.mean((y - y_pred)**2)
print(f"Mean Squared Error: {mse:.4f}")

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.scatter(X_original, y, color='blue', label='Dữ liệu gốc')
plt.plot(X_original, y_pred, color='red', label='Dự đoán')
plt.xlabel('Thời gian học')
plt.ylabel('Diểm số')
plt.title('Hồi quy tuyến tính')
plt.legend()
plt.grid(True)
plt.show()