# Імпорт необхідних бібліотек
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# Вхідний файл із даними
input_file = 'income_data.txt'
# Читання та підготовка даних (такий самий процес, як у попередньому завданні)
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 2500
# Відкриття файлу та читання рядків
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1
# Перетворення на масив numpy та кодування текстових даних
X = np.array(X)
y = np.array(y)
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)
X = X_encoded.astype(float)
# Розбиття даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Функція для навчання, прогнозування та оцінки класифікатора з різними ядрами
def evaluate_svm_kernel(kernel_name):
    classifier = SVC(kernel=kernel_name, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Обчислення показників якості
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Kernel: {kernel_name}")
    print("F1 score: {:.2f}%".format(f1 * 100))
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("")
# Оцінка для поліноміального, гаусового (RBF) та сигмоїдального ядер
evaluate_svm_kernel('poly')
evaluate_svm_kernel('rbf')
evaluate_svm_kernel('sigmoid')
