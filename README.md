
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("Downloads/student_exam_scores.csv")


X = data[['hours_studied', 'sleep_hours', 'attendance_percent']]
Y = data['exam_score']

model =  LinearRegression()
model.fit(X,Y)

predicted_coffee = model.predict(X)

mae = mean_absolute_error(Y,predicted_coffee)
mse = mean_squared_error(Y,predicted_coffee)
rmse = np.sqrt(mse)

print("Mean Absolute Error(MAE): ",mae)
print("Mean Sqared Error(MSE): ",mse)
print("Root Mean Sqared Error(RMSE): ",rmse)
