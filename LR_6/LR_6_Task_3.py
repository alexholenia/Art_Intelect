import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Завантаження даних
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

# Перегляд доступних стовпців
print("Доступні стовпці в даних:")
print(data.columns)

# Видалення пропусків
data = data.dropna()

# Вибір релевантних стовпців
features = ["train_type", "train_class", "fare"]

# Підготовка даних
X = data[features].apply(lambda x: pd.factorize(x)[0])  # Перетворення категоріальних даних на числові
y = data["price"] > data["price"].median()  # Цільова змінна: висока/низька ціна

# Навчання моделі
model = GaussianNB()
model.fit(X, y)

# Прогноз
predictions = model.predict(X)
print("\nРезультати прогнозування (перші 10 записів):")
print(predictions[:10])

# Статистика цін
print("\nСтатистика цін на квитки:")
print(data["price"].describe())
