import pandas as pd
import matplotlib.pyplot as plt
import math

# Clase para representar el árbol de decisiones
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # Función para calcular la entropía de un conjunto de datos
    def _entropy(self, y):
        entropy = 0
        classes = y.unique()
        total_samples = len(y)

        for c in classes:
            p_c = len(y[y == c]) / total_samples
            entropy -= p_c * math.log2(p_c)

        return entropy

    # Función para calcular la ganancia de información de un atributo en un conjunto de datos
    def _information_gain(self, X, y, feature):
        total_entropy = self._entropy(y)
        feature_values = X[feature].unique()

        weighted_entropy = 0
        for val in feature_values:
            subset_y = y[X[feature] == val]
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * self._entropy(subset_y)

        return total_entropy - weighted_entropy

    # Función para encontrar el mejor atributo para dividir el conjunto de datos
    def _find_best_split(self, X, y):
        best_feature = None
        best_info_gain = -1

        for feature in X.columns:
            info_gain = self._information_gain(X, y, feature)
            if info_gain > best_info_gain:
                best_feature = feature
                best_info_gain = info_gain

        return best_feature, best_info_gain

    # Función para construir el árbol de decisiones recursivamente
    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(y.unique()) == 1:
            return y.mode().iloc[0]

        best_feature, info_gain = self._find_best_split(X, y)
        tree = { 'feature': best_feature, 'info_gain': info_gain, 'samples': len(y),
                 'value': [len(y[y == c]) for c in y.unique()] }

        if best_feature is None:
            return tree

        tree['children'] = {}
        for val in X[best_feature].unique():
            subset_X = X[X[best_feature] == val].drop(best_feature, axis=1)
            subset_y = y[X[best_feature] == val]
            tree['children'][val] = self._build_tree(subset_X, subset_y, depth + 1)

        return tree

    # Función para hacer predicciones
    def _predict(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        value = row[feature]

        if value not in tree['children']:
            return tree['children']['__default']
        
        return self._predict(row, tree['children'][value])

    # Función para calcular la tasa de acierto
    def _accuracy(self, y_true, y_pred):
        return sum(y_true == y_pred) / len(y_true)

    # Función para entrenar el modelo
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    # Función para hacer predicciones
    def predict(self, X):
        return [self._predict(row, self.tree) for _, row in X.iterrows()]

    # Función para obtener la ganancia de información en cada paso
    def get_information_gains(self):
        information_gains = []
        self._get_information_gains(self.tree, information_gains)
        return information_gains

    def _get_information_gains(self, tree, information_gains):
        if 'feature' in tree:
            information_gains.append(tree['info_gain'])
            for _, child_tree in tree['children'].items():
                self._get_information_gains(child_tree, information_gains)


# Lee los datos de entrenamiento y prueba desde los archivos .csv
train_data = pd.read_csv('virus_train.csv')
test_data = pd.read_csv('virus_test.csv')

# Concatena los datos de entrenamiento y prueba para asegurar la misma codificación one-hot
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Prepara los datos de entrenamiento y prueba
X_train = pd.get_dummies(combined_data.drop('disease', axis=1))[:len(train_data)]
X_test = pd.get_dummies(combined_data.drop('disease', axis=1))[len(train_data):]

y_train = train_data['disease']
y_test = test_data['disease']

# Crea y entrena el clasificador del árbol de decisiones
clf = DecisionTree(max_depth=None)
clf.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcula la tasa de acierto del árbol
accuracy = clf._accuracy(y_test, y_pred)

# Obtener la ganancia de información en cada paso
information_gains = clf.get_information_gains()

# Guarda los resultados en un archivo .txt
with open('results.txt', 'w') as file:
    file.write("Datos de entrenamiento:\n")
    file.write(str(train_data) + "\n\n")
    file.write("Datos de prueba:\n")
    file.write(str(test_data) + "\n\n")
    file.write("Tasa de acierto del arbol: {:.2f}%\n".format(accuracy * 100))
    file.write("\nGanancia de informacion en cada paso:\n")
    for idx, gain in enumerate(information_gains):
        file.write("Paso {}: {:.4f}\n".format(idx + 1, gain))
    file.write("\nClasificacion de datos de prueba:\n")
    for i, row in test_data.iterrows():
        file.write("Datos:\n{} - Clase real: {} - Clase pronosticada: {}\n".format(
            row.drop('disease'), row['disease'], y_pred[i]))

print("Proceso completado. Los resultados se han guardado en 'results.txt'")
