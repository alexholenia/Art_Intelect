#1
from sklearn.datasets import load_iris

# Завантаження даних
iris_dataset = load_iris()

# Виведення ключів об'єкта iris_dataset
print("Ключі iris_dataset:\n{}".format(iris_dataset.keys()))

# Виведення опису набору даних
print(iris_dataset['DESCR'][:193] + "\n...")

# Виведення назв сортів ірисів, які ми хочемо передбачити
print("Назви відповідей:\n{}".format(iris_dataset['target_names']))

# Виведення назв ознак
print("Назва ознак:\n{}".format(iris_dataset['feature_names']))

# Виведення типу масиву data
print("Тип масиву data: {}".format(type(iris_dataset['data'])))

# Виведення форми масиву data
print("Форма масиву data:\n{}".format(iris_dataset['data'].shape))

# Виведення значень ознак для перших п'яти прикладів
print("Значення ознак для перших п'яти прикладів:\n{}".format(iris_dataset['data'][:5]))

# Виведення типу масиву target
print("Тип масиву target: {}".format(type(iris_dataset['target'])))

# Виведення відповідей (сорти ірисів у числовому форматі)
print("Відповіді:\n{}".format(iris_dataset['target']))

#2
# Завантаження необхідних бібліотек
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Завантаження датасету з URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Перевірка розміру датасету
print("Розмір датасету (кількість екземплярів і атрибутів):", dataset.shape)
# Виведення перших 20 рядків даних
print("\nПерші 20 рядків датасету:")
print(dataset.head(20))
# Статистичне зведення кожного атрибута
print("\nСтатистичне зведення датасету:")
print(dataset.describe())
# Розподіл за атрибутом 'class'
print("\nРозподіл класів:")
print(dataset.groupby('class').size())
# КРОК 2. Візуалізація даних
# Діаграма розмаху (boxplot) для кожного атрибута
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle("Діаграма розмаху атрибутів")
plt.show()
# Гістограма для кожного атрибута
dataset.hist()
plt.suptitle("Гістограма розподілу атрибутів")
plt.show()
# Матриця діаграм розсіювання
scatter_matrix(dataset)
plt.suptitle("Матриця діаграм розсіювання")
plt.show()

#3
from sklearn.model_selection import train_test_split

# Перетворення датасету на NumPy масив
array = dataset.values

# Вибір перших 4-х стовпців як X (вхідні дані)
X = array[:, 0:4]

# Вибір 5-го стовпця як y (цільова змінна)
y = array[:, 4]

# Розподіл даних на навчальний (X_train, y_train) та тестовий (X_test, y_test) набори
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Перевірка розмірів наборів
print(f"Розмір навчального набору: {X_train.shape}")
print(f"Розмір тестового набору: {X_validation.shape}")

#4
# Імпортуємо необхідні бібліотеки
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from matplotlib import pyplot

# Завантажуємо алгоритми моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Оцінка моделей
results = []
names = []
for name, model in models:
    # Використовуємо 10-кратну крос-валідацію
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    # Додаємо результати
    results.append(cv_results)
    names.append(name)

    # Виводимо середнє значення точності та стандартне відхилення для кожної моделі
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')
# Побудова діаграми розмаху для порівняння точності моделей
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#6
# Створюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#7
# Оцінюємо прогноз
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Завантажуємо набір даних Iris
iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target

# Розділяємо дані на тренувальну і тестову вибірку
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# Створюємо і навчаємо модель KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Новий зразок (ірис з довжиною чашолистки 5 см, шириною чашолистки 2.9 см, довжиною пелюстки 1 см і шириною пелюстки 0.2 см)
X_new = np.array([[5, 2.9, 1, 0.2]])

# Робимо прогноз для нового зразка
prediction = knn.predict(X_new)

# Виводимо результат
print("Форма масиву X_new: {}".format(X_new.shape))
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(iris_dataset['target_names'][prediction]))
