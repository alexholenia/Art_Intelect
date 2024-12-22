import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)

# Побудова графіка даних
plt.scatter(X, y, color='green', label='Данні')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Випадкові дані')
plt.legend()
plt.show()

# Лінійна регресія
linear_regressor = LinearRegression()
linear_regressor.fit(X.reshape(-1, 1), y)  # Перетворюємо X на 2D масив
y_pred_linear = linear_regressor.predict(X.reshape(-1, 1))

# Побудова графіка лінійної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(X, y_pred_linear, color='blue', linewidth=3, label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.legend()
plt.show()

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))  # Перетворюємо X на 2D масив
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)

# Побудова графіка поліноміальної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(np.sort(X), poly_regressor.predict(poly_features.transform(np.sort(X).reshape(-1, 1))), color='red', linewidth=3, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія (ступінь 3)')
plt.legend()
plt.show()

# Оцінка якості моделей
# Лінійна регресія
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
mae_linear = mean_absolute_error(y, y_pred_linear)
print("Лінійна регресія:")
print("Mean Squared Error (MSE):", round(mse_linear, 2))
print("R2 score:", round(r2_linear, 2))
print("Mean Absolute Error (MAE):", round(mae_linear, 2))

# Отримання коефіцієнтів і перехоплення
print("Коефіцієнти лінійної регресії:", linear_regressor.coef_)
print("Перехоплення лінійної регресії:", linear_regressor.intercept_)

# Поліноміальна регресія
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)
mae_poly = mean_absolute_error(y, y_pred_poly)
print("\nПоліноміальна регресія:")
print("Mean Squared Error (MSE):", round(mse_poly, 2))
print("R2 score:", round(r2_poly, 2))
print("Mean Absolute Error (MAE):", round(mae_poly, 2))

# Отримання коефіцієнтів і перехоплення
print("Коефіцієнти поліноміальної регресії:", poly_regressor.coef_)
print("Перехоплення поліноміальної регресії:", poly_regressor.intercept_)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X.reshape(-1, 1), y)

from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
    ("poly_features",
     PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
    ])
plot_learning_curves(polynomial_regression, X.reshape(-1, 1), y)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
# Функція для побудови кривих навчання
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    # Побудова графіків
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")
    plt.legend()
# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m)
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m)
# Створення поліноміальної регресії 2-го ступеня
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
# Побудова кривих навчання для поліноміальної регресії 2-го ступеня
plot_learning_curves(polynomial_regression, X.reshape(-1, 1), y)
# Відображення графіку
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.title("Learning Curves (Polynomial Regression - Degree 10)")
plt.show()
