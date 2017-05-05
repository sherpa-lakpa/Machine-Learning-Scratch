import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []

for i in range(25):

	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?', -99999, inplace=True) #-99999 to make data outlier
	df.drop(['id'], 1, inplace=True)

	#X for features
	X = np.array(df.drop(['class'],1))
	#X = preprocessing.scale(X)

	#Y for label or class
	y = np.array(df['class'])

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

	clf = neighbors.KNeighborsClassifier(n_jobs=-1)

	clf.fit(X_train,y_train)

	accuracy = clf.score(X_test,y_test)

	# print(accuracy)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))


# example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,4,1,5,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures),-1)

# prediction = clf.predict(example_measures)

# print(prediction)
