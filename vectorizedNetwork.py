import numpy as np 
import random
import math
import codecs
#import matplotlib.pyplot as plt
import json

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
	return 1/(1+np.exp(-x))
def derivSigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def retreiveNetwork():
	biases = {}
	weights = {}
	with open("weights.txt", 'r') as JSONFile:
		data = json.load(JSONFile)
		for i in data:
			weights[i] = []
			for j in range(len(data[i])):
				weights[i].append(data[i][j])
		for i in weights:
			weights[i] = np.asarray(weights[i])
	with open("biases.txt", 'r') as JSONFile:
		data = json.load(JSONFile)
		for i in data:
			biases[i] = []
			for j in range(len(data[i])):
				biases[i].append(data[i][j])
		for i in biases:
			biases[i] = np.asarray(biases[i])	
	b = []
	w = []
	placeHolder = 0
	while (placeHolder<(len(weights))):
		for i in weights:
			if (int(i)==placeHolder):
				w.append(weights[i])
				placeHolder+=1
	placeHolder = 0
	while (placeHolder<(len(biases))):
		for i in biases:
			if (int(i)==placeHolder):
				b.append(biases[i])
				placeHolder+=1
	return w, b

class Network():
	def __init__ (self, sizes, trainedWeights=None, trainedBiases=None, saveNetworkStuff=True):
		self.numOfLayers = len(sizes)
		if (trainedWeights==None):
			self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
			self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		else:
			self.weights = trainedWeights
			self.biases = trainedBiases
		self.saveNetworkStuff = saveNetworkStuff
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
	def SGD(self, trainingData, miniBatchSize, epochs, eta, testPixels, testNumbers):
		print("Starting Stochastic Gradient Descent...")
		printNumber = 1
		for j in range(epochs):
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch, eta)
			error = 0
			numOfTests = len(testPixels)
			totalCorrect = 0
			totalPercent = 0
			for i in range(numOfTests):
				correct, c = self.mse(testPixels[i].transpose(), testNumbers[i])
				totalCorrect+=correct
				totalPercent+=np.asscalar(c)
			totalPercent/=numOfTests
			if (j%printNumber==0):
				print("Epoch %d complete. Percent Error: %.8f. Total Correct: %d/%d" %(j, totalPercent*100, totalCorrect, numOfTests))
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
		self.weights = [w-(float(eta)/len(miniBatch)*we) for w, we in zip(self.weights, weightError)]
		self.biases = [b-(float(eta)/len(miniBatch)*be) for b, be in zip(self.biases, biasError)]
	def backprop(self, x, y):
		weightError = [np.zeros(w.shape) for w in self.weights]
		biasError = [np.zeros(b.shape) for b in self.biases]
		delta = [np.zeros(b.shape) for b in self.biases]
		activations = [x]
		activation = x
		zs = []
		z = ""
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta[-1] = -costDerivative(y, activations[-1].T)*derivSigmoid(zs[-1]).T
		weightError[-1] = np.dot(activations[-2], delta[-1]).T
		biasError[-1] = delta[-1].T
		for i in range(2, self.numOfLayers):
			z = zs[-i]
			delta[-i] = np.dot(delta[-i+1], self.weights[-i+1]) * derivSigmoid(z).T
			biasError[-i] = delta[-i].T
			weightError[-i] = (activations[-i-1]*delta[-i]).T
		return weightError, biasError
	
	def feedforward(self, activation):
		zs = []
		activations = []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		return(activation)
	def mse(self, x, y):
		whatINeed = self.feedforward(x)
		xplaceholder = 0
		xval = 0
		yplaceholder = 0
		yval = 0
		correct = 0
		for i in range(len(whatINeed)):
			if (whatINeed[i]>xplaceholder): 
				xplaceholder = whatINeed[i]
				xval = i
		for i in range(len(y[0])):
			if (y[0][i]>yplaceholder): 
				yplaceholder = y[0][i]
				yval = i
		if (yval==xval): correct = 1
		return correct, 1-(whatINeed[yval]/yplaceholder)
def costDerivative(y, activation):
	whatINeed = y-activation
	whatINeed = whatINeed*whatINeed/whatINeed
	return whatINeed
def divideData(data, trainOrTest):
	pixels = np.asarray([np.array([data[trainOrTest][i][0]]) for i in range(60000)])
	numbers = np.asarray([np.array([data[trainOrTest][i][1]]) for i in range(60000)])
	trainingData = zip(pixels, numbers)
	testPixels = pixels[:-59000]
	for i in testPixels:
		i = i.transpose()
	testNumbers = numbers[:-59000]
	return trainingData, testPixels, testNumbers

#Parameter Declarations-------------
numOfInputs = 784
epochs = 700
sizeOfMinis = 10000
learnRate = 2
sizes = np.array([numOfInputs,2,3,10])
#-----------------------------------

#Learning--------------------------
trainingData, testPixels, testNumbers = divideData(loadData(), "train")
network = Network(sizes, trainedWeights=None, trainedBiases=None, saveNetworkStuff=True)
network.SGD(trainingData, sizeOfMinis, epochs, learnRate, testPixels, testNumbers)
#----------------------------------
'''
#trained---------------------------
w, b = retreiveNetwork()
example = np.random.rand(784, 1)
trainedNetwork = Network(sizes)
answer = trainedNetwork.feedforward(example)
#----------------------------------
'''
