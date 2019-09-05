'''
This file contains data utility functions of Randomized Neural Network for Outlier Detection using Tensorflow
randnet_model is built from 《Outlier Detection with Autoencoder Ensembles》 by Jinghui Chen, SDM 2017

Created by Kunhong Yu
2019/08/26
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def data_utils(df, strategy = 'under'):
	'''Visualize data set a little bit'''
	'''
	Args : 
		--df: Pandas instance
		--strategy: 'under': under sample, 'upper': up sampling using SMOTE, default is 'under'
	return : 
		--df: processed df
	'''
	print(df.head())
	print(df.describe())
	print(df.isnull().sum().max())
	print(df.columns)
	print('No frauds ', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the data set.')
	print('Frauds ', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the data set.')
	colors = ["#0101DF", "#DF0101"]
	sns.countplot('Class', data = df, palette = colors)
	plt.title('No Fraud : 0 || Fraud : 1', fontsize = 14)
	plt.show()

	_, ax = plt.subplots(1, 2, figsize = (18, 4))
	amount_values = df['Amount'].values
	time_values = df['Time'].values

	sns.distplot(amount_values, ax = ax[0], color = 'r')
	ax[0].set_title('Distribution of Transaction Amount', fontsize = 14)
	ax[0].set_xlim([min(amount_values), max(amount_values)])

	sns.distplot(time_values, ax = ax[1], color = 'b')
	ax[1].set_title('Distribution of Transaction Time', fontsize = 14)
	ax[1].set_xlim([min(time_values), max(time_values)])
	plt.show()

	std_scaler = StandardScaler()
	rbt_scaler = RobustScaler()
	#Feature transformation#
	df['scaled_amount'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
	df['scaled_time'] = std_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
	df.drop(['Time', 'Amount'], axis = 1, inplace = True)
	scaled_amount = df['scaled_amount'].values
	scaled_time = df['scaled_time'].values
	df.drop(['scaled_time', 'scaled_amount'], axis = 1, inplace = True)
	df.insert(0, 'scaled_amount', scaled_amount)
	df.insert(1, 'scaled_time', scaled_time)
	print(df.head())

	#We then split training and testing data set#
	X = df.drop('Class', axis = 1)
	y = df['Class']
	#Load data set#
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	sss = StratifiedKFold(n_splits = 5, random_state = None, shuffle=False)
	for train_indices, test_indices in sss.split(X, y):
		print('Train: ', train_indices)
		print('Test: ', test_indices)
		X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
		y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

	X_train_array = X_train.values
	X_test_array = X_test.values
	y_train_array = y_train.values
	y_test_array = y_test.values

	y_unique_train_labels, train_counts = np.unique(y_train_array, return_counts = True)
	y_unique_test_labels, test_counts = np.unique(y_test_array, return_counts = True)

	print('Train label distribution : ', train_counts / len(y_train_array))
	print('Test label distribution : ', test_counts / len(y_test_array))

	if strategy == 'under':

		#Before we start, we have to balance the labels#
		df = df.sample(frac = 1)
		fraud_loc = df.loc[df['Class'] == 1] 
		no_fraud_loc = df.loc[df['Class'] == 0][: 492]

		df = pd.concat([fraud_loc, no_fraud_loc])
		print('Distribution of the Classes in the subsample dataset')
		print(df['Class'].value_counts() / len(df))

		sns.countplot('Class', data = df, palette = colors)
		plt.title('Equally Distributed Classes', fontsize = 14)
		plt.show()

		#Let's visualize data correlations#
		_, ax = plt.subplots(1, 1, figsize = (24, 20))
		corr = df.corr()
		sns.heatmap(corr, cmap = 'coolwarm_r', annot_kws = {'size' : 20}, ax = ax)
		ax.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
		plt.show()

		#We then have insight of each feature#
		_, ax = plt.subplots(1, 3, figsize = (20, 6))
		V14 = df['V14'].loc[df['Class'] == 1].values
		sns.distplot(V14, ax = ax[0], fit = norm, color = '#FB8861')
		ax[0].set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

		V12 = df['V12'].loc[df['Class'] == 1].values
		sns.distplot(V12, ax = ax[1], fit = norm, color = '#FB8861')
		ax[1].set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

		V10 = df['V10'].loc[df['Class'] == 1].values
		sns.distplot(V10, ax = ax[2], fit = norm, color = '#FB8861')
		ax[2].set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

		plt.show()

		#Let's remove outliers features#
		q25, q75 = np.percentile(V14, 25), np.percentile(V14, 75)
		V14_IQR = q75 - q25
		V14_high = q75 + 1.5 * V14_IQR
		V14_low = q75 - 1.5 * V14_IQR
		df = df.drop(df[(df['V14'] > V14_high) | (df['V14'] < V14_low)].index)

		q25, q75 = np.percentile(V12, 25), np.percentile(V12, 75)
		V12_IQR = q75 - q25
		V12_high = q75 + 1.5 * V12_IQR
		V12_low = q75 - 1.5 * V12_IQR
		df = df.drop(df[(df['V12'] > V12_high) | (df['V12'] < V12_low)].index)

		q25, q75 = np.percentile(V10, 25), np.percentile(V10, 75)
		V10_IQR = q75 - q25
		V10_high = q75 + 1.5 * V10_IQR
		V10_low = q75 - 1.5 * V10_IQR
		df = df.drop(df[(df['V10'] > V10_high) | (df['V10'] < V10_low)].index)

		print('Number of Instances after outliers removal: {}'.format(len(df)))

		X = df.drop('Class', axis = 1)
		y = df['Class']

		#Let's finally visualize data set#
		X_tsne = TSNE(n_components = 2, random_state = 42).fit_transform(X.values)
		X_pca = PCA(n_components = 2, random_state = 42).fit_transform(X.values)
		X_svd = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state = 42).fit_transform(X.values)

		f, ax = plt.subplots(1, 3, figsize = (24, 6))
		f.suptitle('Dimensionality Reduction Algorithms')
		blue_patch = mpatches.Patch(color = '#0A0AFF', label = 'No fraud')
		red_patch = mpatches.Patch(color = '#AF0000', label = 'Fraud')

		ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c = (y == 0), cmap = 'coolwarm', label = 'No fraud', linewidth = 1.6)
		ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c = (y == 1), cmap = 'coolwarm', label = 'Fraud', linewidth = 1.6)
		ax[0].set_title('t-sne', fontsize = 14)
		ax[0].grid(True)
		ax[0].legend(handles = [blue_patch, red_patch])

		ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c = (y == 0), cmap = 'coolwarm', label = 'No fraud', linewidth = 1.6)
		ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c = (y == 1), cmap = 'coolwarm', label = 'Fraud', linewidth = 1.6)
		ax[1].set_title('pca', fontsize = 14)
		ax[1].grid(True)
		ax[1].legend(handles = [blue_patch, red_patch])

		ax[2].scatter(X_svd[:, 0], X_svd[:, 1], c = (y == 0), cmap = 'coolwarm', label = 'No fraud', linewidth = 1.6)
		ax[2].scatter(X_svd[:, 0], X_svd[:, 1], c = (y == 1), cmap = 'coolwarm', label = 'Fraud', linewidth = 1.6)
		ax[2].set_title('svd', fontsize = 14)
		ax[2].grid(True)
		ax[2].legend(handles = [blue_patch, red_patch])

		plt.show()

	print('The data set has ' + str(len(df)) + ' items.')

	return df

def data_utils_opt(X, y):
	'''
	This function is used to preprocessed OptDigits structured data
	According to paper, we have to let labels range from 1 - 9 be class 0
	let label 0 be class 1
	'''
	'''
	Args : 
		--X: input features
		--y: output labels
	return : 
		--X: preprocessed features
		--y: preprocessed labels
	'''
	pos_indices = np.argwhere(y == 0)
	pos_indices = np.squeeze(pos_indices.tolist())

	neg_indices = np.argwhere(y != 0)
	neg_indices = np.squeeze(neg_indices.tolist())

	X_pos = X[pos_indices, :]
	y_pos = np.ones((pos_indices.shape[0], 1))
	X_neg = X[neg_indices, :]
	y_neg = np.zeros((neg_indices.shape[0], 1))

	X = np.concatenate([X_pos, X_neg], axis = 0)
	y = np.concatenate([y_pos, y_neg], axis = 0)
	y = np.squeeze(y)

	'''
	X_tsne = TSNE(n_components = 2, random_state = 23).fit_transform(X)
	f, ax = plt.subplots(1, 1, figsize = (22, 10))
	f.suptitle('Dimensionality Reduction Algorithms')
	blue_patch = mpatches.Patch(color = '#0A0AFF', label = 'Negative')
	red_patch = mpatches.Patch(color = '#AF0000', label = 'Positive')
	ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c = (y == 0), label = 'Negative', cmap = 'coolwarm', linewidth = 1.6)
	ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c = (y == 1), label = 'Positive', cmap = 'coolwarm', linewidth = 1.6)
	ax.grid(True)
	ax.legend(handles = [blue_patch, red_patch])

	plt.show()'''

	return X, y

def data_utils_ecoli(path, mode = 'train'):
	'''
	This function is used to load Ecoli data set
	According to paper, we have to let labels 'omL', 'imL', 'imS' be class 0
	let other labels be class 1
	Original labels:
		cp  (cytoplasm)                                    143
  		im  (inner membrane without signal sequence)        77               
  		pp  (perisplasm)                                    52
  		imU (inner membrane, uncleavable signal sequence)   35
  		om  (outer membrane)                                20
  		omL (outer membrane lipoprotein)                     5
  		imL (inner membrane lipoprotein)                     2
  		imS (inner membrane, cleavable signal sequence)      2
	'''
	'''
	Args : 
		--path: data set path
		--mode: 'train' for training, 'test' for testing
	return : 
		--X: preprocessed features
		--y: preprocessed labels
	'''
	features = []
	labels = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip('\n').split()
			
			feature = line[1 : -1]#We discard first feature
			label = line[-1]

			feature = list(map(float, feature))
			features.append(feature)
			if label == 'omL' or label == 'imL' or label == 'imS':
				labels.append(1)
			else:
				labels.append(0)

		#Features are all between [0, 1], so we don't have to normalize them#

	#Then, we let training set have all examples with good labels, test data set has mixed good and bad labels
	X = np.array(features)
	y = np.array(labels)

	good_indices = np.argwhere(y == 0)
	good_indices = np.squeeze(good_indices.tolist())
	bad_indices = np.argwhere(y == 1)
	bad_indices = np.squeeze(bad_indices.tolist())

	X_train = X[good_indices[: int(len(good_indices) * 0.8)], :]
	y_train = y[good_indices[: int(len(good_indices) * 0.8)]]

	X_test_good = X[good_indices[int(len(good_indices) * 0.8) : ], :]
	y_test_good = y[good_indices[int(len(good_indices) * 0.8) : ]]
	X_test_anomaly = X[bad_indices, :]
	y_test_anomaly = y[bad_indices]

	X_test = np.concatenate([X_test_good, X_test_anomaly], axis = 0)
	y_test = np.concatenate([y_test_good, y_test_anomaly], axis = 0)

	if mode == 'train':
		print('Training features size : ', X_train.shape)
		print('Training labels size : ', y_train.shape)

		return X_train, y_train
	elif mode == 'test':
		print('Testing features size : ', X_test.shape)
		print('Testing labels size : ', y_test.shape)

		return X_test, y_test

def data_utils_lympho(path, mode = 'train'):
	'''
	This function is used to load Lympo data set
	According to paper, we have to let labels 1 and 4 be class 1
	let other labels be class 0
	Original labels:
		    normal find:  2
    		metastases:   81
    		malign lymph: 61
    		fibrosis:     4
	features : 
    2. lymphatics: normal, arched, deformed, displaced
    3. block of affere: no, yes
    4. bl. of lymph. c: no, yes
    5. bl. of lymph. s: no, yes
    6. by pass: no, yes
    7. extravasates: no, yes
    8. regeneration of: no, yes
    9. early uptake in: no, yes
    10. lym.nodes dimin: 0-3
    11. lym.nodes enlar: 1-4
    12. changes in lym.: bean, oval, round
    13. defect in node: no, lacunar, lac. marginal, lac. central
    14. changes in node: no, lacunar, lac. margin, lac. central
    15. changes in stru: no, grainy, drop-like, coarse, diluted, reticular, 
                        stripped, faint, 
    16. special forms: no, chalices, vesicles
    17. dislocation of: no, yes
    18. exclusion of no: no, yes
    19. no. of nodes in: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70
    max_features = [4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 4, 4, 8, 3, 2, 2, 10]
	'''
	'''
	Args : 
		--path: data set path
		--mode: 'train' for training, 'test' for testing
	return : 
		--X: preprocessed features
		--y: preprocessed labels
	'''
	features = []
	labels = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip('\n').split(',')
			
			feature = line[1 :]
			label = line[0]

			feature = list(map(float, feature))
			features.append(feature)
			if label == '1' or label == '4':
				labels.append(1)
			else:
				labels.append(0)

	#Then, we let training set have all examples with good labels, test data set has mixed good and bad labels
	X = np.array(features)
	y = np.array(labels)

	max_features = np.array([4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 4, 4, 8, 3, 2, 2, 10])
	max_features = np.expand_dims(max_features, axis = 0)
	X /= max_features#feature normalization

	good_indices = np.argwhere(y == 0)
	good_indices = np.squeeze(good_indices.tolist())
	bad_indices = np.argwhere(y == 1)
	bad_indices = np.squeeze(bad_indices.tolist())

	X_train = X[good_indices[: int(len(good_indices) * 0.8)], :]
	y_train = y[good_indices[: int(len(good_indices) * 0.8)]]

	X_test_good = X[good_indices[int(len(good_indices) * 0.8) : ], :]
	y_test_good = y[good_indices[int(len(good_indices) * 0.8) : ]]
	X_test_anomaly = X[bad_indices, :]
	y_test_anomaly = y[bad_indices]

	X_test = np.concatenate([X_test_good, X_test_anomaly], axis = 0)
	y_test = np.concatenate([y_test_good, y_test_anomaly], axis = 0)

	if mode == 'train':
		print('Training features size : ', X_train.shape)
		print('Training labels size : ', y_train.shape)

		return X_train, y_train
	elif mode == 'test':
		print('Testing features size : ', X_test.shape)
		print('Testing labels size : ', y_test.shape)

		return X_test, y_test

def data_utils_waveform(path, mode = 'train'):
	'''
	This function is used to load Waveform data set
	According to paper, we have to let labels 0 be class 1
	let other labels be class 0
	'''
	'''
	Args : 
		--path: data set path
		--mode: 'train' for training, 'test' for testing
	return : 
		--X: preprocessed features
		--y: preprocessed labels
	'''
	features = []
	labels = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip('\n').split(',')
			
			feature = line[: -1]
			label = line[-1]

			feature = list(map(float, feature))
			features.append(feature)
			if label == '0':
				labels.append(1)
			else:
				labels.append(0)

	#Then, we let training set have all examples with good labels, test data set has mixed good and bad labels
	X = np.array(features)
	y = np.array(labels)

	good_indices = np.argwhere(y == 0)
	good_indices = np.squeeze(good_indices.tolist())
	bad_indices = np.argwhere(y == 1)
	bad_indices = np.squeeze(bad_indices.tolist())

	X_train = X[good_indices[: int(len(good_indices) * 0.9)], :]
	y_train = y[good_indices[: int(len(good_indices) * 0.9)]]

	X_test_good = X[good_indices[int(len(good_indices) * 0.9) : ], :]
	y_test_good = y[good_indices[int(len(good_indices) * 0.9) : ]]
	X_test_anomaly = X[bad_indices[: int(len(bad_indices) * 0.1)], :]
	y_test_anomaly = y[bad_indices[: int(len(bad_indices) * 0.1)]]#10% data is anomaly

	X_test = np.concatenate([X_test_good, X_test_anomaly], axis = 0)
	y_test = np.concatenate([y_test_good, y_test_anomaly], axis = 0)

	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)

	if mode == 'train':
		print('Training features size : ', X_train.shape)
		print('Training labels size : ', y_train.shape)

		return X_train, y_train
	elif mode == 'test':
		X_test = scaler.fit_transform(X_test)
		print('Testing features size : ', X_test.shape)
		print('Testing labels size : ', y_test.shape)

		return X_test, y_test

def data_utils_yeast(path, mode = 'train'):
	'''
	This function is used to load Yeast data set
	According to paper, we have to let labels ME3, MIT, NUC and CYT be class 0
	let 5% other labels be class 1
	'''
	'''
	Args : 
		--path: data set path
		--mode: 'train' for training, 'test' for testing
	return : 
		--X: preprocessed features
		--y: preprocessed labels
	'''
	features = []
	labels = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip('\n').split()
			
			feature = line[1: -1]
			label = line[-1]

			feature = list(map(float, feature))
			features.append(feature)
			if label == 'ME3' or label == 'MIT' or label == 'NUC' or label == 'CYT':
				labels.append(0)
			else:
				labels.append(1)

	#Then, we let training set have all examples with good labels, test data set has mixed good and bad labels
	X = np.array(features)
	y = np.array(labels)

	good_indices = np.argwhere(y == 0)
	good_indices = np.squeeze(good_indices.tolist())
	bad_indices = np.argwhere(y == 1)
	bad_indices = np.squeeze(bad_indices.tolist())

	X_train = X[good_indices[: int(len(good_indices) * 0.9)], :]
	y_train = y[good_indices[: int(len(good_indices) * 0.9)]]

	X_test_good = X[good_indices[int(len(good_indices) * 0.9) : ], :]
	y_test_good = y[good_indices[int(len(good_indices) * 0.9) : ]]
	X_test_anomaly = X[bad_indices[: int(len(bad_indices) * 0.1)], :]
	y_test_anomaly = y[bad_indices[: int(len(bad_indices) * 0.1 )]]#5% data is anomaly

	X_test = np.concatenate([X_test_good, X_test_anomaly], axis = 0)
	y_test = np.concatenate([y_test_good, y_test_anomaly], axis = 0)

	if mode == 'train':
		print('Training features size : ', X_train.shape)
		print('Training labels size : ', y_train.shape)

		return X_train, y_train
	elif mode == 'test':
		print('Testing features size : ', X_test.shape)
		print('Testing labels size : ', y_test.shape)

		return X_test, y_test
