#math
import numpy as np 
import random
import math
#visual
import matplotlib.pyplot as plt
from tkinter import *
import PIL
from PIL import ImageTk, Image, ImageDraw
#functionality
import codecs
import json
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
#time
import time

def draw(fileName):
    def save():
        image1.save(fileName)
    def drawIm(event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        cv.create_oval(x1, y1, x2, y2, width=5, fill="black")
        draw.line(((x2,y2),(x1,y1)), fill="black", width=10)
    width = 200
    height = 200
    white = (255, 255, 255)
    root = Tk()
    cv = Canvas(root, width=width, height=height, bg="white")
    cv.pack()
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.bind("<B1-Motion>", drawIm)
    button=Button(text="save", command=save)
    button.pack()
    root.mainloop()
    
def getActivation(fileName):
    img = Image.open(fileName)
    img = img.resize((28, 28))
    img = np.take(np.asarray(img), [0], axis = 2).reshape(28, 28)
    return np.abs(img-255)

def saveNetwork(fileNames, weights, biases, layer="Dense"):
    trainedBiases = {}
    trainedWeights = {}
    if layer=="Dense":
      for i in range(len(weights)):
        trainedBiases[i] = []
        trainedWeights[i] = []
        for j, k in zip(biases[i], weights[i]):
          trainedBiases[i].append(j.tolist())
          trainedWeights[i].append(k.tolist())
    else:
      trainedWeights[0] = []
      trainedBiases[0] = []
      trainedWeights[0].append(weights.tolist())
      trainedBiases[0].append(biases.tolist())
    with open (fileNames[0], 'w+') as JSONFile:
      json.dump(trainedWeights, JSONFile)
    with open(fileNames[1], 'w+') as JSONFile:
      json.dump(trainedBiases, JSONFile)

def retreiveNetwork(fileNames):
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
  take(fileNames[0], 'r', weights, w)
  take(fileNames[1], 'r', biases, b)
  return w, b

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
        parsed = np.frombuffer(data, dtype=np.uint8, offset = 8)
      elif magicNumber==2051:
        numOfRows = toInt(data[8:12])
        numOfColumns = toInt(data[12:16])
        parsed = normalize(np.frombuffer(data, dtype=np.uint8, offset = 16).reshape(length, numOfRows*numOfColumns), (0, 255))
      else: return -1
      return parsed
  data = {"train":[], "test":[]}
  trainImages = loadFile("/Users/michaelpilarski/Desktop/CNNs/train-images.idx3-ubyte")
  trainLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/train-labels.idx1-ubyte")
  data["train"] = np.asarray(list(zip(trainImages, np.asarray([vectorize(i) for i in trainLabels]))))
  testLabels = loadFile("/Users/michaelpilarski/Desktop/CNNs/t10k-labels.idx1-ubyte")
  testImages = loadFile("/Users/michaelpilarski/Desktop/CNNs/t10k-images.idx3-ubyte")
  data["test"] = np.asarray(list(zip(testImages, np.asarray([vectorize(i) for i in testLabels]))))
  return data
  
class featureMap():
  def __init__ (self, kernelDimensions, trainedNet=None):
    self.kernelDimensions = kernelDimensions
    if trainedNet==None:
      self.weights = np.random.randn(5, 5)/np.sqrt(5)
      self.biases = np.random.randn((1))
    else:
      weights, biases = trainedNet
      self.weights = weights[0][0]
      self.biases = biases[0][0]
    self.weightError = np.zeros(self.weights.shape)
    self.biasError = np.zeros((1))
    self.storedWeights = self.weights
    self.storedBiases = self.biases
  def reset(self):
    self.Image = None
    self.Chunks = None
    self.Pools = None
    self.num = None
    self.Z = None
  def convolve(self, image, num, strideLength=1):
    self.num = (num+1)*144
    self.Image = image.reshape(28, 28)
    self.Chunks = view_as_windows(self.Image, self.kernelDimensions, strideLength).reshape(576, 25)*self.weights.reshape(25)
    self.Z = np.array([np.sum(self.Chunks, axis=1)]).reshape(24, 24)+self.biases
    return self.pool(sigmoid(self.Z))
  def pool(self, chunks):
    pools = view_as_blocks(chunks, (2, 2)).reshape(144, 4)**2
    pieces = [np.sum(pools, axis=1)]
    self.Pools = pools.reshape(144, 2, 2)
    return np.sqrt(pieces).reshape(12, 12)
  def getPoolDerivatives(self, delta):
    pools = self.Pools.reshape(144, 4)
    rootSummedSquares = np.sqrt(np.sum(pools**2, axis = 1))
    poolDerivatives = (pools.T/rootSummedSquares).T.reshape(144, 2, 2)
    return (delta*poolDerivatives.T).T
  def reconstructShape(self, poolDerivatives):
    remade = poolDerivatives.reshape(288, 2)
    z = [remade[i:i+24] for i in range(0, 288, 24)]
    remade = np.concatenate([zs for zs in z], axis = 1)
    return remade.T
  def backpropCONV(self, delta, inputWeights):
    weights = inputWeights.T[self.num-144:self.num].T
    delta = np.dot(delta, weights).reshape(144)
    poolDerivatives = self.getPoolDerivatives(delta)
    alignedPoolDerivatives = self.reconstructShape(poolDerivatives.transpose(0, 2, 1)).reshape(576)
    weightError = np.sum((self.Chunks.T*(alignedPoolDerivatives*derivSigmoid(self.Z).reshape(576))).T, axis=0).reshape(5, 5)/576
    biasError = np.sum(alignedPoolDerivatives)/576
    self.weightError+=weightError
    self.biasError+=biasError
    start = times()
    self.reset()
  def updateMiniBatchCONV(self, miniBatch, eta, lmbda, trainingData):
    self.weights = (1-eta*lmbda/len(trainingData))*self.weights-(float(eta)/len(miniBatch))*self.weightError
    self.biases = self.biases-(float(eta)/len(miniBatch))*self.biasError
    self.weightError = np.zeros(self.weights.shape)
    self.biasError = np.zeros((1))
  def update(self):
    self.storedWeights = self.weights
    self.storedBiases = self.biases
  def get(self, what):
    if what=='w': return self.storedWeights
    else: return self.storedBiases
  def display(self):
      displayImage(self.weights)
    
def sigmoid(x):
  return 1/(1+np.exp(-x))
def derivSigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))
def times():
  return time.process_time()
    
class Network():
  def __init__ (self, numFeatures, inputSizes, kernelDimensions, saveNet=False, takeNet=False):
    self.numFeatures = numFeatures
    self.sizes = np.asarray([numFeatures*144, inputSizes[0], inputSizes[1]])
    self.saveNet = saveNet
    if self.saveNet: self.fileNames = [[("weights%d20.txt" %(i+1)), ("biases%d20.txt" %(i+1))] for i in range(numFeatures)]
    if takeNet:
      self.weights, self.biases = retreiveNetwork(["weights20.txt", "biases20.txt"])
      self.features = [featureMap(kernelDimensions, retreiveNetwork([("weights%d20.txt" %(i+1)), ("biases%d20.txt" %(i+1))])) for i in range(numFeatures)]
    else:
      self.weights = [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
      self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
      self.features = [featureMap(kernelDimensions) for i in range(numFeatures)]
    self.storedWeights = self.weights
    self.storedBiases = self.biases
    self.highestCorrect = 0
    self.streak = 0
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
    for w, b in zip(self.weights, self.biases): activation = sigmoid(np.dot(w, activation)+b)
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
    for feature in self.features: feature.updateMiniBatchCONV(miniBatch, eta, lmbda, trainingData)
  def earlyStop(self, totalCorrect):
    if totalCorrect>self.highestCorrect:
      self.storedWeights = self.weights
      self.storedBiases = self.biases
      self.highestCorrect = totalCorrect
      self.streak = 0
      for feature in self.features: feature.update()
    else: self.streak+=1
    if self.streak>=4: #return True
        return False
    else: return False
  def display(self):
    for feature in self.features: feature.display()
  def SGD(self, lmbda, eta, trainingData, testData, epochs, miniBatchSize):
    self.lmbda=lmbda
    self.eta=eta
    for j in range(epochs):
      self.lmbda=lmbda*(1-j/2/epochs)
      self.eta=eta*(1-j/2/epochs)
      start = times()
      random.shuffle(trainingData)
      miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
      for miniBatch in miniBatches: self.updateMiniBatch(miniBatch, self.lmbda, self.eta)
      totalCorrect = 0
      totalPercent = 0
      for x, y in testData:
        percent, correct = network.mse(x, y)
        totalCorrect+=correct
        totalPercent+=percent
      totalPercent/=(len(testData))
      print("Epoch %d - Percent Error: %.8f. Total Correct: %d/%d" %(j+1, totalPercent, totalCorrect, len(testData)))
      print(times()-start)
      if self.earlyStop(totalCorrect): break
    if self.saveNet:
      saveNetwork(["weights20.txt", "biases20.txt"], self.storedWeights, self.storedBiases, layer="Dense")
      for feature, fileName in zip(self.features, self.fileNames): saveNetwork([fileName[0], fileName[1]], feature.get('w'), feature.get('b'), layer="Conv")
  def classify(self, image):
        x = self.feedforward(image.reshape(784, 1)).reshape(1, 10)
        print("Network prediction: %d" %(np.argmax(x)))
        displayImage(image.reshape(28, 28))
      
    
data = loadData()
trainingData = data["train"]
testData = data["test"]
kernelDimensions = (5, 5)
sizes = [30, 10]
def getImage():
    fileName = "image.png"
    draw(fileName)
    activation = getActivation(fileName)
    return activation


network = Network(5, sizes, kernelDimensions, saveNet=True, takeNet=False)
network.SGD(.8, .3, trainingData, testData, 20, 10)
#network.display()
#network.classify(getImage())
