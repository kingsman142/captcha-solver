from numpy import array, append, dot, random
import numpy
import math
from scipy import misc

class ConvLayer():
    def __init__(self, filterXY, filterDepth, filters, padding, stride):
        self.filter = random.random((filterXY, filterXY, filterDepth))
        self.filter_size = filterXY
        self.padding = padding
        self.stride = stride
        self.title = 'conv'

    def conv(self, input):
        self.outputX = (input.shape[0] - self.filter_size + (2 * self.padding)) / self.stride + 1
        self.outputY = input.shape[1] - self.filter_size + 1
        outputMatrix = numpy.empty([self.outputX, self.outputY])

        for x in range(0, outputMatrix.shape[0]):
            for y in range(0, outputMatrix.shape[1]):
                upperX = x + self.filter_size
                upperY = y + self.filter_size
                subMatrix = input[x:upperX, y:upperY, :]
                #print outputMatrix.shape, " and ", input.shape, " and ", x, upperX, " and ", y, upperY, " and ", subMatrix.shape
                output = sum(numpy.multiply(self.filter, subMatrix).flatten()) # Sum of element-wise multiplications
                if output < 0:
                    output = 0
                elif output > 1:
                    output = 1
                #print output
                outputMatrix[x, y] = output

        return outputMatrix

class PoolLayer():
    def __init__(self, poolSize, filters):
        self.pool_size = poolSize
        self.filters = filters
        self.title = 'pool'

    def pool(self, input):
        outputX = input.shape[0] / 2
        outputY = input.shape[1] / 2
        outputMatrix = numpy.empty([outputX, outputY, self.filters])
        #for i in xrange(self.filters):
        for x in xrange(outputX):
            for y in xrange(outputY):
                lowerX = x * self.pool_size
                lowerY = y * self.pool_size
                #lowerZ = i
                upperX = lowerX + self.pool_size
                upperY = lowerY + self.pool_size
                #upperZ = lowerZ + 1
                #subMatrix = input[lowerX:upperX, lowerY:upperY, lowerZ:upperZ]
                subMatrix = input[lowerX:upperX, lowerY:upperY]
                maxValue = numpy.amax(subMatrix)
                #outputMatrix[x, y, i] = maxValue
                outputMatrix[x, y] = maxValue
        return outputMatrix

class ReLULayer():
    def __init__(self, filters):
        self.filters = filters
        self.title = 'relu'

    def relu(self, input):
        inputX = input.shape[0]
        inputY = input.shape[1]
        for x in xrange(inputX):
            for y in xrange(inputY):
                input[x, y] = max(0, input[x, y])
        return input

class FullyConnectedLayer():
    def __init__(self, hiddenNeurons, outputLength):
        self.weights = random.random((hiddenNeurons, outputLength))
        self.title = 'full'

    def connect(self, input):
        input = input.flatten()
        return dot(numpy.transpose(self.weights), input)

class SoftmaxLayer():
    def __init__(self):
        self.title = 'soft'

    def softmax(self, input):
        vecExp = [math.exp(i) for i in input]
        vecExpSum = sum(vecExp)
        softmaxResult = [(i / vecExpSum) for i in vecExp]
        return softmaxResult

class CNN():
    def __init__(self, layers):
        # Random seed for weights
        random.seed()

        self.learning_rate = 0.8
        # self.weights = 2 * random.random((36, 1)) - 1
        #self.weights = random.random((84, 36))
        #self.filter = random.random((self.filter_size, self.filter_size, 3))
        self.layers = layers
        self.total_loss = 0.0
        self.num_trained_images = 0
        #print self.filter

    def calcError(self, predicted, output):
        # Mean-Squared Error
        # Error = Summation( 1/2 * (target - output)**2)
        # loss = sum(.5*(predicted-output)**2)
        # MSE isn't bad, but it gives too much emphasis for incorrect outputs compared to Cross-Entropy Loss

        # Cross-Entropy Loss
        # Loss = -Summation( log(p(x)) ) where p(x) is the probability of the predicted label
        lnPredicted = [math.log(i) for i in predicted] # Calculate the natural logarithm for each element in the predicted list
        loss = -sum(lnPredicted) # Carry out the rest of cross-entropy loss equation

        current_total_loss = self.total_loss * self.num_trained_images
        current_total_loss += loss
        self.num_trained_images += 1

        newLoss = current_total_loss / self.num_trained_images
        self.total_loss = newLoss  # Set the CNN's new average loss across all trained images
        return newLoss

    def adjustWeights(self, error):
        '''for i in range(0, self.filter_size):
            for j in range(0, self.filter_size):
                    self.weights[i, j] -= self.learning_rate * error'''

    def train(self, iterations, input, output, imageX, imageY):
        for it in xrange(iterations): # for each image in the input
            imageIndex = it % len(input)
            imageCopy = input[imageIndex]
            for layer in self.layers: # forward pass
                if layer.title ==   'conv':
                    imageCopy = layer.conv(imageCopy)
                elif layer.title == 'relu':
                    imageCopy = layer.relu(imageCopy)
                elif layer.title == 'pool':
                    imageCopy = layer.pool(imageCopy)
                elif layer.title == 'full':
                    imageCopy = layer.connect(imageCopy)
                elif layer.title == 'soft':
                    imageCopy = layer.softmax(imageCopy)

            if it % 100 == 0:
                print it
            #print imageCopy
            error = self.calcError(output[it % len(output)], imageCopy) # calculate loss
            self.adjustWeights(error) # backward pass and weight update

train_input = []
train_output = []
print len(train_input), " and ", len(train_output)
with open("./labels.txt", "r") as labelsFile:
    for label in labelsFile:
        stripped_label = label.rstrip()
        char = stripped_label[0]
        index = -1
        if char.isalpha():
            index = ord(char) - 65
        else:
            index = (ord(char) - 48) + 26
        output = numpy.zeros(36)
        output[index] = 1
        image = misc.imread("./captchas/" + stripped_label + ".jpg")
        image = numpy.divide(image, 256.0)
        train_input.append(image)
        train_output.append(output)
print len(train_input), " and ", len(train_output)

# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
# neural_network.train(training_set_inputs, training_set_outputs, 10000)
layers = [ConvLayer(5, 3, 1, 0, 1), # One 5x5x3 filter with 0 padding and 1 stride
          ReLULayer(1), # 1 filter
          PoolLayer(2, 1), # 2x2 padding with 1 filter
          ConvLayer(5, 1, 1, 0, 1),
          ReLULayer(1),
          PoolLayer(2, 1),
          ConvLayer(5, 1, 1, 0, 1),
          ReLULayer(1),
          PoolLayer(2, 1),
          FullyConnectedLayer(84, 36), # 84 hidden layers (14x6 matrix at this point) with 36 outputs
          SoftmaxLayer()]
cnn = CNN(layers)
cnn.train(500, train_input, train_output, 140, 76)
