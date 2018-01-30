import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values # prendi le labels delle prime due classi di fiori
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length')
# plt.ylabel('petal lenght')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number	of	misclassifications')
# plt.show()




Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
