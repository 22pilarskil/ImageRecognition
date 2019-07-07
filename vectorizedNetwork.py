import numpy as np 
import random
import codecs
import matplotlib.pyplot as plt
import json

def displayImage(pixels, label = None):
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  if label != None: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")  
  plt.imshow(pixels, cmap = "gray")
  plt.show()

def compressImage(pixels, imageWidth, imageHeight, newWidth, newHeight):
	widthScalar = newWidth/imageWidth
	heightScalar = newHeight/imageHeight
	pixels = pixels.reshape(imageHeight, imageWidth)



def createCoefficient(we):
	return(np.nan_to_num(we/we))

def loadData():
	print("Loading Data...")
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
	#add validation if needed
	data = {"train":[], "test":[]}
	trainImages = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/train-images-idx3-ubyte")
	trainLabels = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/train-labels-idx1-ubyte")
	data["train"] = np.asarray(list(zip(trainImages, np.asarray([vectorize(i) for i in trainLabels]))))
	testLabels = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/t10k-labels-idx1-ubyte")
	testImages = loadFile("/Users/MichaelPilarski1/Desktop/Neural_Network/data/t10k-images-idx3-ubyte")
	data["test"] = np.asarray(list(zip(testImages, np.asarray([vectorize(i) for i in testLabels]))))
	return data

	'''
	Before using this function, you must first download all four of the binary array files from this link (http://yann.lecun.com/exdb/mnist/)
	Drag them to your directory without double clicking them, and only once you have placed them in a separate directory (in my case, 'data') do you double click them
	Change the argument in the laodFile function to the directory paths of each of the files
	'''

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

	def __init__ (self, sizes, trainedWeights=None, trainedBiases=None, saveNetworkStuff=True):
		self.numOfLayers = len(sizes)
		if (trainedWeights==None):
			self.weights = [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])]
			self.storedWeights = np.copy(self.weights)
			self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
			self.storedBiases = np.copy(self.biases)
		else:
			self.weights = trainedWeights
			self.biases = trainedBiases
		self.saveNetworkStuff = saveNetworkStuff
		self.minimum = 100
		self.streak = 0
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

	def earlyStop(self, totalPercent, epochs):
		specialNum = .05*epochs+4
		if (totalPercent>self.minimum): self.streak+=1
		else: 
			self.minimum = totalPercent
			self.streak = 0
			self.storedWeights = np.copy(self.weights)
			self.storedBiases = np.copy(self.biases)
		if (self.streak>=specialNum): return True
		else: return False

	def SGD(self, trainingData, miniBatchSize, epochs, eta, testData, printNumber, lmbda=0):
		print("Starting Stochastic Gradient Descent...")
		self.totalCorrect = 0
		self.totalPercent = 0
		def reset():
			self.totalPercent = 0
			self.totalCorrect = 0
		def makeCheck(label):
			reset()
			self.totalCorrect = 0
			for x, y in testData:
				percent, correct = self.mse(x.reshape(784, 1), y)
				self.totalCorrect+=correct
				self.totalPercent+=percent
			self.totalPercent/=(len(testData)/100)
			print(label)
			print("Percent Error: %.8f. Total Correct: %d/%d" %(self.totalPercent, self.totalCorrect, len(testData)))
			reset()
		makeCheck("Initialization:")
		for j in range(epochs):
			reset()
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
			for miniBatch in miniBatches:
				self.updateMiniBatch(miniBatch, eta, lmbda, len(trainingData))
			for x, y in testData:
				percent, correct = self.mse(x.reshape(784, 1), y)
				self.totalCorrect+=correct
				self.totalPercent+=percent
			self.totalPercent/=(len(testData)/100)
			if (j%printNumber==0):
				print("Epoch %d complete. Percent Error: %.8f. Total Correct: %d/%d" %(j, self.totalPercent, self.totalCorrect, len(testData)))
			if (self.earlyStop(self.totalPercent, epochs)):
				print("Network oversaturated- exiting SGD")
				self.weights = np.copy(self.storedWeights)
				self.biases = np.copy(self.storedBiases)
				break
		makeCheck("Final Status:")
		if(self.saveNetworkStuff):
			self.saveNetwork()
			print("Weights and Biases Saved")

	def updateMiniBatch(self, miniBatch, eta, lmbda, n):
		weightError = [np.zeros(w.shape) for w in self.weights]
		biasError = [np.zeros(b.shape) for b in self.biases]
		for x, y in miniBatch:
			x = x.transpose()
			deltaWeightError, deltaBiasError = self.backprop(x, y)
			weightError = [we+dwe+(lmbda/len(miniBatch)*w)for we, dwe, w in zip(weightError, deltaWeightError, self.weights)]
			#weightError = [we+dwe for we, dwe in zip(weightError, deltaWeightError)]
			biasError = [be+dbe for be, dbe in zip(biasError, deltaBiasError)]
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
		delta[-1] = self.costDerivative(activations[-1], y)
		weightError[-1] = np.dot(activations[-2], delta[-1]).T
		biasError[-1] = delta[-1].T
		for i in range(2, self.numOfLayers):
			z = zs[-i]
			delta[-i] = np.dot(delta[-i+1], self.weights[-i+1]) * derivSigmoid(z).T
			biasError[-i] = delta[-i].T
			weightError[-i] = (activations[-i-1]*delta[-i]).T
			'''
			negs = 0
			x = 0
		for w in weightError:
			for ww in w:
				for www in ww:
					if (www==0): negs+=1
		print("%d/%d" %(negs, 30*784+10*30))
		'''
		return weightError, biasError
	
	def feedforward(self, activation):
		for w, b in zip(self.weights, self.biases): activation = sigmoid(np.dot(w, activation)+b)
		return(activation)

	def mse(self, x, y):
		prediction = self.feedforward(x).reshape(10)
		correct = 0
		percent = (np.linalg.norm(y.reshape(1, 10)-prediction) ** 2) 
		if np.argmax(prediction) == np.argmax(y): correct = 1
		return percent, correct

	def classify(self, trainingData):
		x = self.feedforward(trainingData[0].reshape(784, 1)).reshape(1, 10)
		print("Network prediction: %d. Actual: %d" %(np.argmax(x), np.argmax(trainingData[1])))
		displayImage(trainingData[0].reshape(28, 28))
		
	def costDerivative(self, activation, y):
		return activation.T-y
#Parameter Declarations-------------
numOfInputs = 784
epochs = 20
sizeOfMinis = 10
learnRate = 0.5
sizes = np.array([numOfInputs,30,10])
regularizationParam = 0.0
printNumber = 1
#-----------------------------------

#Learning/Classify------------------
w, b = retreiveNetwork()
network = Network(sizes, trainedWeights=w, trainedBiases=b, saveNetworkStuff=True)
trainingData = loadData()
#network.classify(trainingData["test"][100])
network.SGD(trainingData["train"], sizeOfMinis, epochs, learnRate, trainingData["test"], printNumber, regularizationParam)
#----------------------------------
