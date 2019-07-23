import numpy as np 
import random
import codecs
import matplotlib.pyplot as plt
import json
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
import time
import math
def displayImage(pixels, label = None):
  figure = plt.gcf()
  figure.canvas.set_window_title("Number display")
  if label != None: plt.title("Label: \"{label}\"".format(label = label))
  else: plt.title("No label")  
  plt.imshow(pixels, cmap = "gray")
  plt.show()
  plt.close()

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
  trainImages = loadFile("/Users/michaelpilarski/Desktop/CNNs/train-images.idx3-ubyte")
  trainLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/train-labels.idx1-ubyte")
  data["train"] = np.asarray(list(zip(trainImages, np.asarray([vectorize(i) for i in trainLabels]))))
  testLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/t10k-labels.idx1-ubyte")
  testImages = loadFile("/Users/michaelpilarski/Desktop/CNNs/t10k-images.idx3-ubyte")
  data["test"] = np.asarray(list(zip(testImages, np.asarray([vectorize(i) for i in testLabels]))))
  return data

data = loadData()
trainingData = data["train"]
testData = data["test"]
kernelDimensions = (5, 5)
sizes = [30, 10]

def times():
  return time.process_time()
  
class featureMap():
  def __init__ (self, kernelDimensions):
    self.kernelDimensions = kernelDimensions
    self.weights = np.random.randn(kernelDimensions[0], kernelDimensions[1])
    self.biases = np.random.randn((1))
    self.Image = None
    self.Chunks = None
    self.Pools = None
    self.num = None
    self.weightError = np.zeros(self.weights.shape)
    self.biasError = np.zeros((1))
  def convolve(self, image, num, strideLength=1):
    self.num = (num+1)*144
    self.Image = image.reshape(28, 28)
    self.Chunks = view_as_windows(self.Image, self.kernelDimensions, strideLength).reshape(576, 25)*self.weights.reshape(25)
    chunks = np.asarray([np.sum(self.Chunks, axis=1)]).reshape(24, 24)
    return self.pool(chunks, strideLength)
  def pool(self, chunks, strideLength):
    pools = view_as_blocks(chunks, (2, 2)).reshape(144, 4)**2
    pieces = np.asarray([np.sum(pools, axis=1)]).reshape(12, 12)
    self.Pools = pools.reshape(144, 2, 2)
    return np.sqrt(pieces)
  def getPoolDerivatives(self, delta):
    pools = self.Pools.reshape(144, 4)
    rootSummedSquares = np.sqrt(np.sum(pools**2, axis = 1))
    poolDerivatives = (pools.T/rootSummedSquares).T.reshape(144, 2, 2)
    return poolDerivatives
  def reconstructShape(self, poolDerivatives):
    remade = poolDerivatives.reshape(288, 2)
    z = [remade[i:i+24] for i in range(0, 288, 24)]
    remade = np.asarray(np.concatenate([zs for zs in z], axis = 1))
    return remade.T
  def backpropCONV(self, delta, inputWeights):
    weights = np.take(inputWeights, [i for i in range(self.num-144, self.num)], axis=1)
    delta = np.dot(delta, weights).reshape(144)
    poolDerivatives = self.getPoolDerivatives(delta)
    alignedPoolDerivatives = self.reconstructShape(poolDerivatives.transpose(0, 2, 1)).flatten()
    weightError = np.sum((self.Chunks.T*alignedPoolDerivatives).T, axis=0).reshape(5, 5)
    biasError = np.sum(alignedPoolDerivatives)
    self.weightError+=weightError
    self.biasError+=biasError
  def updateMiniBatchCONV(self, miniBatch, eta):
    self.weights = [w-(float(eta)/len(miniBatch))*we for w, we in zip(self.weights, self.weightError)]
    self.biases = [b-(float(eta)/len(miniBatch))*be for b, be in zip(self.biases, self.biasError)]
    
def sigmoid(x):
  return 1/(1+np.exp(-x))
def derivSigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))
    
class Network():
  def __init__ (self, numFeatures, inputSizes, kernelDimensions):
    self.numFeatures = numFeatures
    self.sizes = np.asarray([numFeatures*144, inputSizes[0], inputSizes[1]])
    self.weights = [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.features = [featureMap(kernelDimensions) for i in range(numFeatures)]
  def feedforward(self):
    activation = np.asarray([self.features[f].convolve(imageData[0], f) for f in range(len(self.features))]).reshape(self.numFeatures*144, 1)
    for w, b in zip(self.weights, self.biases): activation = sigmoid(np.dot(w, activation)+b)
    return activation
  def backpropMLP(self, image):
    activation = np.asarray([self.features[f].convolve(image[0], f) for f in range(len(self.features))]).reshape(self.numFeatures*144, 1)
    activations = [activation]
    zs = []
    biasError = [np.zeros(b.shape) for b in self.biases]
    weightError = [np.zeros(w.shape) for w in self.weights]
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = [np.zeros(b.shape) for b in self.biases]
    delta[-1] = activation.T-image[1]
    weightError[-1] = np.dot(activations[-2], delta[-1]).T
    biasError[-1] = delta[-1].T
    for i in range(2, len(self.sizes)):
      z = zs[-i]
      delta[-i] = np.dot(delta[-i+1], self.weights[-i+1]) * derivSigmoid(z).T
      biasError[-i] = delta[-i].T
      weightError[-i] = (activations[-i-1]*delta[-i]).T
    for feature in self.features: feature.backpropCONV(delta[0], self.weights[0])
    return weightError, biasError
  def feedforward(self, x):
    activation = np.asarray([self.features[f].convolve(x, f) for f in range(len(self.features))]).reshape(self.numFeatures*144, 1)
    for w, b in zip(self.weights, self.biases): activation = np.dot(w, activation)+b
    return activation
  def mse(self, x, y):
    prediction = self.feedforward(x).reshape(10)
    correct = 0
    percent = (np.linalg.norm(y.reshape(1, 10)-prediction) ** 2) 
    if np.argmax(prediction) == np.argmax(y): correct = 1
    return percent, correct
  def updateMiniBatch(self, miniBatch, lmbda, eta):
    weightError = [np.zeros(w.shape) for w in self.weights]
    biasError = [np.zeros(b.shape) for b in self.biases]
    for image in miniBatch:
      deltaWeightError, deltaBiasError = self.backpropMLP(image)
      weightError = [we+dwe for we, dwe in zip(weightError, deltaWeightError)]
      biasError = [be+dbe for be, dbe in zip(biasError, deltaBiasError)]
    self.weights = [(1-eta*lmbda/len(trainingData))*w-(((float(eta)/len(miniBatch))*we)) for w, we in zip(self.weights, weightError)]
    self.biases = [b-(((float(eta)/len(miniBatch))*be)) for b, be in zip(self.biases, biasError)]
    for feature in self.features: feature.updateMiniBatchCONV(miniBatch, eta)
  def SGD(self, lmbda, eta, trainingData, testData, epochs, miniBatchSize):
    for j in range(epochs):
      random.shuffle(trainingData)
      miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
      for miniBatch in miniBatches: self.updateMiniBatch(miniBatch, lmbda, eta)

network = Network(3, sizes, kernelDimensions)
'''
feature = featureMap(kernelDimensions)
start = time.process_time()
feature.convolve(trainingData[1][0], 1)
end = time.process_time()
print(end-start)
'''

'''
start = time.process_time()
network.backpropMLP(trainingData[0])
end = time.process_time()
print((end-start))
'''

start = time.process_time()
for i in range(1):
  network.backpropMLP(trainingData[0])
end = time.process_time()
#print((end-start))


#network.SGD(0, .5, trainingData, testData, 20, 10)
