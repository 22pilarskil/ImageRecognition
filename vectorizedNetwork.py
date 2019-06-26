import numpy as np 
import random
import codecs
import matplotlib.pyplot as plt
import json

class crossEntropyCost(object):
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	@staticmethod
	def delta(z, a, y):
		return (a.transpose()-y)

def toInt(b):
	return int(codecs.encode(b, "hex"), 16)

def normalize(rawArray, range_):
	array = np.copy(rawArray).astype(np.float32)
	if range_ == (0, 1):
		return range_
	array-=range_[0]
	dist = abs(range_[0])+abs(range_[1])
	array /= dist
	return array

def vectorize(num):
	array = np.zeros(10)
	array[num] = 1
	return array

def displayImage(pixels, label = None):
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  if label != None: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")  
  plt.imshow(pixels, cmap = "gray")
  plt.show()

def loadFile(fileName, mode="rb"):
	with open(fileName, mode) as raw:
		data = raw.read()
		magicNumber = toInt(data[:4])
		length = toInt(data[4:8])
		if magicNumber==2049:
			#labels
			parsed = np.frombuffer(data, dtype=np.uint8, offset = 8)
		elif magicNumber==2051:
			#images
			numOfRows = toInt(data[8:12])
			numOfColumns = toInt(data[12:16])
			parsed = normalize(np.frombuffer(data, dtype=np.uint8, offset = 16).reshape(length, numOfRows*numOfColumns), (0, 255))
		else: return -1
		return parsed

def loadData(vector=True):
	print("Loading Data...")
	'''
	Before using this function, you must first download all four of the binary array files from this link (http://yann.lecun.com/exdb/mnist/)
	Drag them to your directory without double clicking them, and only once you have placed them in a separate directory (in my case, 'data') do you double click them
	Change the argument in the laodFile function to the directory paths of each of the files
	'''
	data = {"train":[], "validation":[], "test":[]}
	trainImages = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/train-images-idx3-ubyte")
	trainLabels = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/train-labels-idx1-ubyte")
	if vector:
		data["train"] = np.asarray(list(zip(trainImages, np.asarray([vectorize(i) for i in trainLabels]))))
	else: data["train"] = np.asarray(list(zip(trainImages, np.asarray([i for i in trainLabels]))))
	testLabels = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/t10k-labels-idx1-ubyte")
	testImages = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/t10k-images-idx3-ubyte")
	data["test"] = np.asarray(list(zip(testImages, np.asarray(testLabels))))
	return data

def sigmoid(x):
	return float(1)/(float(1)+np.exp(-x))

def derivSigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def retreiveNetwork():
	biases = {}
	weights = {}
	b = []
	w = []
	def take(fileName, mode, dictionary, listName):
		with open(fileName, mode) as JSONFile:
			data = json.load(JSONFile)
			for i in data:
				dictionary[i] = []
				for j in range(len(data[i])):
					dictionary[i].append(data[i][j])
			for i in dictionary:
				dictionary[i] = np.asarray(dictionary[i])
		placeHolder = 0
		while (placeHolder<(len(dictionary))):
			for i in dictionary:
				if (int(i)==placeHolder):
					listName.append(dictionary[i])
					placeHolder+=1
	take("weights.txt", 'r', weights, w)
	take("biases.txt", 'r', biases, b)
	return w, b

class Network():

	def __init__ (self, sizes, trainedWeights=None, trainedBiases=None, saveNetworkStuff=True, cost=crossEntropyCost):
		self.numOfLayers = len(sizes)
		if (trainedWeights==None):
			self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
			self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		else:
			self.weights = trainedWeights
			self.biases = trainedBiases
		self.saveNetworkStuff = saveNetworkStuff
		self.cost = cost

	def saveNetwork(self):
		trainedBiases = {}
		trainedWeights = {}
		for i in range(len(self.weights)):
			trainedBiases[i] = []
			trainedWeights[i] = []
			for j, k in zip(self.biases[i], self.weights[i]):
				trainedBiases[i].append(j.tolist())
				trainedWeights[i].append(k.tolist())
		with open("weights.txt", 'w+') as JSONFile:
			json.dump(trainedWeights, JSONFile)
		with open("biases.txt", 'w+') as JSONFile:
			json.dump(trainedBiases, JSONFile)

	def SGD(self, trainingData, miniBatchSize, epochs, eta, testData):
		print("Starting Stochastic Gradient Descent...")
		printNumber = 1
		for j in range(epochs):
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch, eta)
			totalCorrect = 0
			totalPercentx = 0
			blah = trainingData
			numOfTests = len(blah)
			for x, y in blah:
				totalPercent, correct = self.mse(x.reshape(784, 1), y)
				totalCorrect+=correct
				totalPercentx+=totalPercent
			totalPercentx/=numOfTests
			if (j%printNumber==0):
				print("Epoch %d complete. Percent Error: %.8f. Total Correct: %d/%d" %(j, totalPercentx*100, totalCorrect, numOfTests))
		if(self.saveNetworkStuff):
			self.saveNetwork()
			print("Weights and Biases Saved")

	def updateMiniBatch(self, miniBatch, eta):
		weightError = [np.zeros(w.shape) for w in self.weights]
		biasError = [np.zeros(b.shape) for b in self.biases]
		for x, y in miniBatch:
			x = x.transpose()
			deltaWeightError, deltaBiasError = self.backprop(x, y)
			weightError = [we+dwe for we, dwe in zip(weightError, deltaWeightError)]
			biasError = [be+dbe for be, dbe in zip(biasError, deltaBiasError)]
		#try changing eta/len to just eta
		self.weights = [w-(((float(eta)/len(miniBatch))*we)) for w, we in zip(self.weights, weightError)]
		self.biases = [b-(((float(eta)/len(miniBatch))*be)) for b, be in zip(self.biases, biasError)]

	def backprop(self, x, y):
		weightError = [np.zeros(w.shape) for w in self.weights]
		biasError = [np.zeros(b.shape) for b in self.biases]
		delta = [np.zeros(b.shape) for b in self.biases]
		activation = x.reshape(784, 1)
		activations = [activation]
		zs = []
		z = ""
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta[-1] = (self.cost).delta(zs[-1], activations[-1], y)
		weightError[-1] = np.dot(activations[-2], delta[-1]).T
		biasError[-1] = delta[-1].T
		for i in range(2, self.numOfLayers):
			z = zs[-i]
			delta[-i] = np.dot(delta[-i+1], self.weights[-i+1]) * derivSigmoid(z).T
			biasError[-i] = delta[-i].T
			weightError[-i] = (activations[-i-1]*delta[-i]).T
		return weightError, biasError
	
	def feedforward(self, activation):
		for w, b in zip(self.weights, self.biases): activation = sigmoid(np.dot(w, activation)+b)
		return(activation)

	def mse(self, x, y):
		prediction = self.feedforward(x).reshape(10)
		correct = 0
		totalPercent = (np.linalg.norm(y.reshape(1, 10)-prediction) ** 2) 
		if np.argmax(prediction) == np.argmax(y): correct = 1
		return totalPercent, correct

	def classify(self, trainingData):
		x = self.feedforward(trainingData[0].reshape(784, 1)).reshape(1, 10)
		print("Network prediction: %d. Actual: %d" %(np.argmax(x), trainingData[1]))
		displayImage(trainingData[0].reshape(28, 28))
		
def costDerivative(y, activation):
	whatINeed = y-activation
	return whatINeed
#Parameter Declarations-------------
numOfInputs = 784
epochs = 4
sizeOfMinis = 10
learnRate = 2
sizes = np.array([numOfInputs,30,10])
#-----------------------------------

#Learning/Classify------------------
w, b = retreiveNetwork()
network = Network(sizes, trainedWeights=w, trainedBiases=b, saveNetworkStuff=False)
trainingData = loadData()
network.classify(trainingData["test"][14])
#network.SGD(trainingData["train"], sizeOfMinis, epochs, learnRate, trainingData["test"])
#----------------------------------
