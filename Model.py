import numpy as np
import pandas
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

data = pandas.read_csv("irisdata.csv",header=None)
dataset = data.values
# extract 4 data dimensions and store in irisdata, extract class information and store in classdata
irisdata = dataset[1:, :4].astype(float) 
classdata = dataset[1:, 4]

# process classdata to make a vector of 0s and 1s as desired
irisclass = []
for i in classdata:
	if i == "setosa":
		irisclass.append([1, 0, 0])
	elif i == "versicolor":
		irisclass.append([0, 1, 0])
	else:
		irisclass.append([0, 0, 1])

# transform irisclass into numpy array type. this is a new requirement of the new TensorFlow version
irisclass = np.array(irisclass)

# split irisdata, which is the input, into training and validating set
# split irisclass, which is the output, into training and validating set
inputTrain,inputVal,outputTrain,outputVal = train_test_split(irisdata, irisclass, test_size=0.25, shuffle=True)

def neuralNetwork():
	# set model to sequential since it is a model where values are continuously updates (feed-forward)
	model = Sequential()
	# set model dimension to 3 outputs for 3 classes and 4 input dimensions for 4 data dimension we are considering
	# we are using the sigmoid function. the reasons will be explained in the writeup as a tutorial
	model.add(Dense(3, input_dim=4, activation='sigmoid'))
	# the rmsprop is chosen because after researching all available optimizers, I believed rmsprop is most suitable since its optimization goal is focused towards the gradient
	# rmsprop is about dividing the gradient by the root of the average of the square of gradients to estimate the variance and optimize based on that
	# besides rmsprop, the sgd optimizer can also be a good candidate since it optimizes around gradient descent
	model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
	return model

irisNN = neuralNetwork()
irisNN.fit(x=inputTrain, y=outputTrain, epochs=2000, validation_data=(inputVal,outputVal), verbose=0)
print("Training loss/accuracy " + str(irisNN.evaluate(inputTrain, outputTrain)))
print("Validation loss/accuracy " + str(irisNN.evaluate(inputVal, outputVal)))

# process training data and validation data of each class
train1 = []
train2 = []
train3 = []
val1 = []
val2 = []
val3 = []

for i in range(len(outputTrain)):
	c = outputTrain[i]
	if c[0] == 1:
		train1.append(inputTrain[i])
	elif c[1] == 1:
		train2.append(inputTrain[i])
	else:
		train3.append(inputTrain[i])

for i in range(len(outputVal)):
	c = outputVal[i]
	if c[0] == 1:
		val1.append(inputVal[i])
	elif c[1] == 1:
		val2.append(inputVal[i])
	else:
		val3.append(inputVal[i])

# transforming into numpy arrays for plotting
train1 = np.array(train1)
train2 = np.array(train2)
train3 = np.array(train3)

val1 = np.array(val1)
val2 = np.array(val2)
val3 = np.array(val3)

# plot the obtained data
plt.figure()
plt.subplot(211)
plt.scatter(train1[:,0], train1[:,1], color='red', marker=".")
plt.scatter(val1[:,0], val1[:,1], color='red', marker="+")

plt.scatter(train2[:,0], train2[:,1], color='blue', marker=".")
plt.scatter(val2[:,0], val2[:,1], color='blue', marker="+")

plt.scatter(train3[:,0], train3[:,1], color='green', marker=".")
plt.scatter(val3[:,0], val3[:,1], color='green', marker="+")

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title("Sepal Data Training (.) vs Validation(+)")

plt.subplot(212)
plt.scatter(train1[:,2], train1[:,3], color='red', marker=".")
plt.scatter(val1[:,2], val1[:,3], color='red', marker="+")

plt.scatter(train2[:,2], train2[:,3], color='blue', marker=".")
plt.scatter(val2[:,2], val2[:,3], color='blue', marker="+")

plt.scatter(train3[:,2], train3[:,3], color='green', marker=".")
plt.scatter(val3[:,2], val3[:,3], color='green', marker="+")

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title("Petal Data Training (.) vs Validation (+)")
plt.tight_layout()

# show plot
plt.show()