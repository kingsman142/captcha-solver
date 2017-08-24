from numpy import array, append, dot, random
import numpy
import math
from scipy import misc

class ConvLayer():
    def __init__(self, filterXY, filterDepth, filters, inputShape, padding, stride):
        self.filter = random.random((filterXY, filterXY, filterDepth))
        self.filter_size = filterXY
        self.outputX = inputShape[0] - filterXY
        self.outputY = inputShape[1] - filterXY

    def conv(self, input):
        outputMatrix = numpy.empty([self.outputY, self.outputX])

        for x in range(0, output.shape[0]):
            for y in range(0, output.shape[1]):
                upperX = x + self.filter_size
                upperY = y + self.filter_size
                subMatrix = input[x:upperX, y:upperY, :]
                #print newMatrix.shape, " and ", input.shape, " and ", x, upperX, " and ", y, upperY, " and ", subMatrix.shape
                output = sum(numpy.multiply(self.filter, subMatrix).flatten()) # Sum of element-wise multiplications
                if output < 0:
                    output = 0
                elif output > 1:
                    output = 1
                #print output
                outputMatrix[x, y] = output
        #print newMatrix

        newMatrix = self.relu(newMatrix) # ReLU (Rectified Linear Units) layer in between each convolutional layer
        newMatrix = self.pool(newMatrix) # 2x2 Pooling layer

        newMatrix = newMatrix.flatten() # Prepare matrix for Fully Connected layer
        newMatrix = self.fullyConnectedLayer(newMatrix)
        newMatrix = self.softmax(newMatrix)
        return newMatrix

class PoolLayer():
    def __init__(self, poolSize, filters, input):
        self.size = poolSize
        self.filters = filters

    def pool(self, input):
        outputX = input.shape[0] / 2
        outputY = input.shape[1] / 2
        outputMatrix = numpy.empty([outputX, outputY, self.filters])
        for i in xrange(self.filters):
            for x in xrange(outputX):
                for y in xrange(outputY):
                    lowerX = i * self.pool_size
                    lowerY = j * self.pool_size
                    lowerZ = i
                    upperX = lowerX + self.pool_size
                    upperY = lowerY + self.pool_size
                    upperZ = lowerZ + 1
                    subMatrix = matrix[lowerX:upperX, lowerY:upperY, lowerZ:upperZ]
                    maxValue = numpy.amax(subMatrix)
                    outputMatrix[x, y, i] = maxValue
        return outputMatrix

class ReLULayer():
    def __init__(self, filters):
        self.filters = filters

    def relu(self, input):
        inputX = input.shape[0]
        inputY = input.shape[1]
        for x in xrange(inputX):
            for y in xrange(inputY):
                input[i, j] = max(0, input[i, j])
        return input

class FullyConnectedLayer():
    def __init__(hiddenNeurons, outputLength):
        self.weights = random.random((hiddenNeurons, outputLength))

    def connect(self, input):
        return dot(numpy.transpose(self.weights), input)

class SoftmaxLayer():
    def softmax(self, input):
        vecExp = [math.exp(i) for i in input]
        vecExpSum = sum(vecExp)
        softmaxResult = [(i / vecExpSum) for i in vecExp]
        return softmaxResult

class CNN():
    def __init__(self):
        # Random seed for weights
        random.seed()

        # Set the random weights
        self.filter_size = 5
        self.layers = 3
        self.learning_rate = 0.8
        self.pool_size = 2
        # self.weights = 2 * random.random((36, 1)) - 1
        self.weights = random.random((84, 36))
        self.filter = random.random((self.filter_size, self.filter_size, 3))
        #print self.filter

    def relu(self, matrix):
        for i in range(0, self.filter_size):
            for j in range(0, self.filter_size):
                #for k in range(0, 3):
                matrix[i, j] = max(0, matrix[i, j])
        return matrix

    def think(self, input, output):
        newXDimension = (self.imageX - self.filter_size + 1) / self.stride
        newYDimension = (self.imageY - self.filter_size + 1) / self.stride
        newMatrix = numpy.empty([newYDimension, newXDimension])
        tempMatrix = []
        for i in range(0, self.layers):
            if i > 0:
                tempMatrix = numpy.empty([newMatrix.shape[0] - self.filter_size + 1, newMatrix.shape[1] - self.filter_size + 1])
            xMax = newMatrix.shape[0] if i == 0 else tempMatrix.shape[0]
            yMax = newMatrix.shape[1] if i == 0 else tempMatrix.shape[1]

            for x in range(0, xMax):
                for y in range(0, yMax):
                    # for z in range(0, 3):
                    upperX = x + self.filter_size
                    upperY = y + self.filter_size
                    # upperZ = z + 1
                    # subMatrix = input[x:upperX, y:upperY, z:upperZ]
                    subMatrix = []
                    if i == 0:
                        subMatrix = input[x:upperX, y:upperY, :]
                #print newMatrix.shape, " and ", input.shape, " and ", x, upperX, " and ", y, upperY, " and ", subMatrix.shape
                        output = sum(numpy.multiply(self.filter, subMatrix).flatten()) # Sum of element-wise multiplications
                        if output < 0:
                            output = 0
                        elif output > 1:
                            output = 1
                        #print output
                        newMatrix[x, y] = output
                    else:
                        subMatrix = newMatrix[x:upperX, y:upperY]
                        output = sum(numpy.multiply(self.filter[:,:,0:1], subMatrix).flatten()) # Sum of element-wise multiplications
                        if output < 0:
                            output = 0
                        elif output > 1:
                            output = 1
                        tempMatrix[x, y] = output
            #print newMatrix

            if i > 0:
                newMatrix = tempMatrix
            newMatrix = self.relu(newMatrix) # ReLU (Rectified Linear Units) layer in between each convolutional layer
            newMatrix = self.pool(newMatrix) # 2x2 Pooling layer

        newMatrix = newMatrix.flatten() # Prepare matrix for Fully Connected layer
        newMatrix = self.fullyConnectedLayer(newMatrix)
        newMatrix = self.softmax(newMatrix)
        return newMatrix

    def calcError(self, predicted, output):
        sum = 0
        # Mean-Squared Error
        # Error = Summation( 1/2 * (target - output)**2)
        for i in range(0, 36):
            sum += ( .5 * (predicted[0] - output[0])**2) # Calculate MSE
        return sum

    def adjustWeights(self, error):
        for i in range(0, self.filter_size):
            for j in range(0, self.filter_size):
                    self.weights[i, j] -= self.learning_rate * error

    def softmax(self, vector):
        #print vector
        vecExp = [math.exp(i) for i in vector]
        #print vecExp
        expSum = sum(vecExp)
        #print "sum: ", expSum
        normalized = [(i / expSum) for i in vecExp]
        return normalized

    def fullyConnectedLayer(self, matrix):
        #print matrix
        #print self.weights
        return dot(numpy.transpose(self.weights), matrix)

    def pool(self, matrix):
        matrixX = matrix.shape[0]
        matrixY = matrix.shape[1]
        newMatrix = numpy.empty([matrixX / self.pool_size, matrixY / self.pool_size])
        for i in range(0, matrixX / self.pool_size):
            for j in range(0, matrixY / self.pool_size):
                lowerX = i * self.pool_size
                lowerY = j * self.pool_size
                upperX = lowerX + self.pool_size
                upperY = lowerY + self.pool_size
                subMatrix = matrix[lowerX:upperX, lowerY:upperY]
                maxValue = numpy.amax(subMatrix)
                newMatrix[i, j] = maxValue
        return newMatrix

    def train(self, iterations, input, output, imageX, imageY):
        self.stride = 1
        self.padding = 0
        self.imageX = imageX
        self.imageY = imageY

        for it in xrange(iterations): # for each image in the input
            forwardPass = self.think(input[it % len(input)], output[it % len(output)]) # forward pass
            # print forwardPass
            if it % 100 == 0:
                print it
            error = self.calcError(output[it % len(output)], forwardPass) # calculate loss
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
        #image = open("./captchas/" + stripped_label + ".jpg", "r")
        image = misc.imread("./captchas/" + stripped_label + ".jpg")
        image = numpy.divide(image, 256.0)
        #print image.shape
        #train_input = append(train_input, image)
        train_input.append(image)
        train_output.append(output)
print len(train_input), " and ", len(train_output)

# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
# neural_network.train(training_set_inputs, training_set_outputs, 10000)
cnn = CNN()
cnn.train(500, train_input, train_output, 140, 76)
