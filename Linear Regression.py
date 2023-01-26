import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn import metrics as sq
from sklearn import linear_model
clf = linear_model.LinearRegression()

#Đọc tệp
data = pd.read_csv("F:/Ngôn ngữ lập trình python/Báo cáo chuyên đề/DuLieuYKhoa.csv", sep=";")
data.head
#In dữ liệu
print(data)
#Lấy dataframe bề dày thành mạch làm biên mục tiêu
Y = data['BEDAYTM'].to_numpy()
#Lấy datafarame không chứa bề dày thành mạch làm biến giải thích
X = data.drop("BEDAYTM", axis = 1)
#Phân loại dữ liệu train và test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 20)
#Tạo model suy đoán
clf.fit(X_train, Y_train)
#In hệ số hồi quy của các biến giải thích xếp theo thứ tự tăng dần
print("\nHỆ SỐ HỒI QUY")
print(pd.DataFrame({"Tên": X_train.columns, "Hệ số": np.abs(clf.coef_)}).sort_values(by='Hệ số'))
#In sai số
print("\nSAI SỐ")
print(clf.intercept_)
#print(clf.score())
#Tiến hành dự đoán với bộ dữ liệu test
Y_pred = clf.predict(X_test)
print("\nGIÁ TRỊ Y DỰ ĐOÁN")
print(Y_pred)
#In giá trị y test thực tế
print("\nGIÁ TRỊ Y THỰC TẾ")
print(Y_test)
#Kiểm tra mức độ lỗi của model (Mean Squared Error)
mse = sq.mean_squared_error(Y_test, Y_pred)
print("KIỂM TRA MỨC ĐỘ LỖI CỦA MÔ HÌNH")
print(mse)
#Bảng biểu so sánh giá trị y dự đoán và y thực tế
plt.scatter(Y_test, Y_pred)
plt.xlabel("Giá trị thực tế: $Y_i$")
plt.ylabel("Giá trị dự đoán: $\hat{Y}_i$")
plt.title("Bảng biểu so sánh $Y_i$ vs $\hat{Y}_i$")
plt.show()
