import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Función auxiliar para obtener la ganancia de información en cada nodo
def get_information_gain(tree, node_id):
    gain = tree.impurity[node_id] - (tree.impurity[tree.children_left[node_id]] * tree.weighted_n_node_samples[tree.children_left[node_id]] +
                                     tree.impurity[tree.children_right[node_id]] * tree.weighted_n_node_samples[tree.children_right[node_id]]) / tree.weighted_n_node_samples[node_id]
    return gain

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
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcula la tasa de acierto del árbol
accuracy = accuracy_score(y_test, y_pred)

# Visualiza el árbol de decisiones y muestra la ganancia de información en cada paso
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=X_train.columns, class_names=clf.classes_, filled=True, impurity=True, fontsize=10)
plt.show()

# Obtener la ganancia de información en cada paso
def get_information_gains(tree):
    information_gains = []
    _get_information_gains(tree.tree_, 0, information_gains)
    return information_gains

def _get_information_gains(tree, node_id, information_gains):
    gain_info = get_information_gain(tree, node_id)
    if gain_info is not None:
        information_gains.append(gain_info)
    if tree.children_left[node_id] != -1:
        _get_information_gains(tree, tree.children_left[node_id], information_gains)
    if tree.children_right[node_id] != -1:
        _get_information_gains(tree, tree.children_right[node_id], information_gains)

information_gains = get_information_gains(clf)

# Guarda los resultados en un archivo .txt
with open('results.txt', 'w') as file:
    file.write("Datos de entrenamiento:\n")
    file.write(str(train_data) + "\n\n")
    file.write("Datos de prueba:\n")
    file.write(str(test_data) + "\n\n")
    file.write("Tasa de acierto del arbol: {:.2f}%\n".format(accuracy * 100))
    file.write("Ganancia de informacion en cada paso:\n")
    for idx, gain in enumerate(information_gains):
        file.write("Paso {}: {:.4f}\n".format(idx + 1, gain))
    file.write("\nClasificacion de datos de prueba:\n")
    for i, row in test_data.iterrows():
        file.write("Datos: {} - Clase real: {} - Clase pronosticada: {}\n".format(
            row.drop('disease'), row['disease'], y_pred[i]))

print("Proceso completado. Los resultados se han guardado en 'results.txt' y se ha mostrado el arbol de decisiones.")
