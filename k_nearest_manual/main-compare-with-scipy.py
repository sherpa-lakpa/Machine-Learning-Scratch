from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

# plot1 = [1,3]
# plot2 = [2,5]
# euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
# print(euclidean_distance)

dataset = {'k' : [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]} #here K=> and r=>
new_features = [5,7]

# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0],ii[1], s=100, color=i)


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting group')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group]) #saves distances with each points and its class
	
	votes = [i[1] for i in sorted(distances)[:k]]	#we get top class in votes not distance since we sorted the distance
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]	#Check the most common votes in votes
	confidence = Counter(votes).most_common(1)[0][1] / k #[1]=> how many, hopes numerator be equals to k
	# print(vote_result, float(confidence))
	return vote_result, confidence

accuracies = []

for i in range(25):

	df = pd.read_csv("breast-cancer-wisconsin.data")
	df.replace('?',-99999, inplace=True)
	df.drop(['id'],1, inplace=True)
	full_data = df.astype(float).values.tolist() #if we don't make float then there exist some data in quoates
	# print(full_data[:5])
	random.shuffle(full_data)
	# print(20*'=')
	# print(full_data[:5])

	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(test_size*len(full_data))] #here data will be all except test 20%
	test_data = full_data[-int(test_size*len(full_data)):] #here data will be 20%

	for i in train_data:
		train_set[i[-1]].append(i[:-1])	#set with list to the elements up to last in according to dictionary class column i.e 2 or 4

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0
	confidence_sum = 0
	c = 0

	for group in test_set:	#for each group ie 2 and 4 in test_set
		for data in test_set[group]:	#for each data in that group
			vote,confidence = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			else:
				# print(float(confidence)) 
				confidence_sum += confidence
				c += 1
			total += 1
	accuracy = float(correct)/total	#float for specially python 2.3
	# confidence_per = confidence_sum/c
	# print('Confidence: ',confidence_per)
	# print('Accuracy : ', accuracy)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))