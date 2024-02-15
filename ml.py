
""" 
	This file is altered from original file because of most of variable names
	and comments were in my motherlanguage(Turkish). For simplicity all code compiled into 3
	functions, each has its docstring.
	As a physiotherapist working in ICU learning coding and implenting in daily activities and job related tasks
	this is my first time writng code for a research paper, I hope my code won't hurt your eyes too much.
"""
# Due to ethical commite decision we can not share patient data with you

import numpy as np
import pandas as pd
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# working folder of patient data
WORK_FOLDER = "example\\folder\\"

# column names for the data rows
sutunlar = [
	"yaş", "cinsiyet", "BMI", "carlson", "BK", "lenfosit", "hct", "plat", "D-dimer", "Ferritin",
	"CRP", "Procalsitonin", "creatinin", "proBNP", "pH", "PaO2", "PaCO2", "SaO2", "FiO2", "P/F", "Solunum desteği",
	"sistolik KB", "diastolik KB", "Solunum", "ateş", "nabız", "YB çıkış şekli"
]
# column names of categorical variables
kategorikler = ["cinsiyet", "Solunum desteği"]
# possible values of categorical vars
kategorikDegerler = [[1, 2], [1, 2, 3]]
# column names of numeric variables
numerikler = [
	"yaş", "BMI", "carlson", "BK", "lenfosit", "hct", "plat", "D-dimer", "Ferritin",
	"CRP", "Procalsitonin", "creatinin", "proBNP", "pH", "PaO2", "PaCO2", "SaO2", "FiO2", "P/F",
	"sistolik KB", "diastolik KB", "Solunum", "ateş", "nabız"
]
# possible range for numeric values(will be used for min-max normalization)
numerikAralik = [
	[18, 100], [10, 60], [0, 37], [0, 100000], [0, 5000], [15, 50], [10, 1000], [0, 10000], [0, 3000],
	[0, 400], [0.05, 100], [0.1, 10], [0, 35000], [6.8, 7.8], [30, 200], [20, 150], [50, 100], [20, 100], [0, 400],
	[60, 220], [30, 120], [5, 50], [30, 42], [40, 220]
]


# loading, preparation and normalization of patient raw data
def loadData(dosyaAd):
	"""load patient data from given file name as csv
		for normalization min-max normalization is used
		and data was returned as 2 numpy arrays

	Args:
		dosyaAd (str): file to laod patient data from

	Returns:
		np_array: mlData : array of normalized patient data last value is y
		np_array: basliklar : column names of given data
	"""
	dataframe = pd.read_csv(WORK_FOLDER + "rawData/" + dosyaAd + ".csv", usecols=sutunlar)
	data = []
	basliklar = []
	# min-max normaliziation
	for n in range(len(numerikler)):
		# data to numpy
		g = dataframe[numerikler[n]].to_numpy()
		# missing data handling
		# choosing non empty data
		gx = g[np.logical_not(np.isnan(g))]
		# mean of non empty data
		mean = np.mean(gx)
		# if mean is nan it will be accepted as 0.5
		if np.isnan(mean): # We did not have a case of a nan mean but code was written before all data was available
			mean = 0.5
		# mean is putted in nan values place
		g[np.isnan(g)] = mean
		# min-max normalization for numeric vars
		g = (g - numerikAralik[n][0])/(numerikAralik[n][1] - numerikAralik[n][0])
		data.append(g)
		basliklar.append(numerikler[n])

	# loop for categorical variables
	# categorical variables are one-hot encoded
	for k in range(len(kategorikler)):
		gecici = []
		for n in range(len(kategorikDegerler[k])):
			if len(kategorikDegerler[k]) == 2:
				basliklar.append(kategorikler[k])
				gecici.append([])
				break
			basliklar.append(kategorikler[k] + " " + str(kategorikDegerler[k][n]))
			gecici.append([])
		for x in range(len(dataframe[kategorikler[k]])):
			for y in range(len(kategorikDegerler[k])):
				if len(kategorikDegerler[k]) == 2:
					if kategorikler[k] == "cinsiyet":
						gecici[0].append(dataframe[kategorikler[k]][x] - 1)
					else:
						gecici[0].append(dataframe[kategorikler[k]][x])
					break
				if kategorikDegerler[k][y] == dataframe[kategorikler[k]][x]:
					gecici[y].append(1)
				else:
					gecici[y].append(0)
		for grup in gecici:
			data.append(gecici[0])

	# numpy convert
	mlData = np.zeros((len(data[0]), len(data)+1))
	for satir in range(len(data[0])):
		for sutun in range(len(data)):
			mlData[satir, sutun] = data[sutun][satir]
	# defining y values 
	for y in range(len(dataframe["YB çıkış şekli"])):
		if dataframe["YB çıkış şekli"][y] == 1:
			mlData[y,28] = 1
		else:
			mlData[y,28] = 0
	
	return mlData, basliklar


# loading training data
mlData, basliklar = loadData("trainGroup")
X_train = mlData[:, :28]
y_train = mlData[:, 28:]


# loading validation data
mlDataPI, basliklarPI = loadData("valGroup")
X_test = mlDataPI[:, :28]
y_test = mlDataPI[:, 28:]


def modelSelectionTraining():
	# some predefined arbitarry hidden layer configurations
	mimariler = [
		[10],
		[10,10,10],
		[10,10,10,10],
		[20],
		[20,20],
		[20,20,20],
		[20,20,20,20],
		[30],
		[30,30],
		[30,30,30],
		[30,30,30,30],
		[40],
		[40,40],
		[40,40,40],
		[40,40,40,40],
		[50],
		[50,50],
		[50,50,50],
		[50,50,50,50],
		[60],
		[60,60],
		[60,60,60],
		[60,60,60,60],
	]

	katmanMaxSayi = 4
	katmanMinSayi = 1
	nodeMaxSayi = 45
	nodeMinSayi = 15

	# adding multiple random hidden layers and nodes for testing
	for g in range(500):
		katman = np.random.randint(katmanMinSayi, katmanMaxSayi)
		node = np.random.randint(nodeMinSayi, nodeMaxSayi)
		mimariler.append([node] * katman)

	precisionScore = -1 # best precision score
	maxScoreKey = -1 # corresponding key
	# training every model and choosing precision wise best performer model
	for m in range(len(mimariler)):
		clf = MLPClassifier(hidden_layer_sizes=mimariler[m], activation='relu', solver='adam', verbose=False, validation_fraction=0.1)
		clf.fit(X_train, y_train.ravel())
		scores = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='precision')
		results = clf.predict(X_test)
		print(f"Mimari: {mimariler[m]}, Avg. Prec: {average_precision_score(y_test, results)}, Prec: {precision_score(y_test, results)}, CV mean: {scores.mean()}")
		# best model so far
		if np.abs(precision_score(y_test, results) - scores.mean()) < 0.1 and precisionScore <  precision_score(y_test, results):
			precisionScore = precision_score(y_test, results)
			maxScoreKey = m
			# save the model to disk
			filename = WORK_FOLDER + 'final.sav'
			pickle.dump(clf, open(filename, 'wb'))
	print(f"Best found model layer&nodes: {mimariler[maxScoreKey]}, prec: {precisionScore}")



def modelTesting():
	""" calculates permutaion importance for the saved model
	"""
	mlDataPI, basliklarPI = loadData("valGroup")
	X_test = mlDataPI[:, :28]
	y_test = mlDataPI[:, 28:]
	# load the model from disk
	loaded_model = pickle.load(open(WORK_FOLDER + 'final.sav', 'rb'))
	from sklearn.inspection import permutation_importance
	r = permutation_importance(loaded_model, X_test, y_test, n_repeats=30, random_state=0)
	for i in r.importances_mean.argsort()[::-1]:
		if r.importances_mean[i] - 2 * r.importances_std[i] > 0 or 1>0:
			print(f"{basliklar[i]} : "
				f"{r.importances_mean[i]:.3f}"
				f" +/- {r.importances_std[i]:.3f}")
