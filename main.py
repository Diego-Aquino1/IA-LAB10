import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

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

# Visualiza el árbol de decisiones y guarda la imagen en un archivo
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X_train.columns,  
                                class_names=clf.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
#graph.format = 'png'
#graph.render('decision_tree')

# Guarda los resultados en un archivo .txt
with open('results.txt', 'w') as file:
    file.write("Datos de entrenamiento:\n")
    file.write(str(train_data) + "\n\n")
    file.write("Datos de prueba:\n")
    file.write(str(test_data) + "\n\n")
    file.write("Tasa de acierto del árbol: {:.2f}%\n".format(accuracy * 100))
    file.write("Clasificación de datos de prueba:\n")
    for i, row in test_data.iterrows():
        file.write("Datos: {} - Clase real: {} - Clase pronosticada: {}\n".format(
            row.drop('disease'), row['disease'], y_pred[i]))

print("Proceso completado. Los resultados se han guardado en 'results.txt' y el árbol se ha generado como 'decision_tree.png'.")
