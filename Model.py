import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


def load_mnist_dataset():
  # load data from tensorflow framework
  ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data() 

  # Stacking train data and test data to form single array named data
  data = np.vstack([trainData, testData]) 

  # Vertical stacking labels of train and test set
  labels = np.hstack([trainLabels, testLabels]) 

  # return a 2-tuple of the MNIST data and labels
  return (data, labels)

X , Y = load_mnist_dataset()
x_, X_test, y_, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshaping the data

x_train = x_.reshape((x_.shape[0], -1))
y_train = y_.reshape((y_.shape[0], -1))

x_test = X_test.reshape((X_test.shape[0], -1))
y_test = Y_test.reshape((Y_test.shape[0], -1))


model = Sequential([
    Dense(128, activation='relu', input_shape=(784,),name= "L1"),
    Dense(64, activation='relu',name= "L2"),
    Dense(10, activation='linear',name= "L3"),
])

# compile the model
model.compile(loss= SparseCategoricalCrossentropy(from_logits=True), optimizer='Adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=15,verbose= 1)



model.save("model.h5", save_format="h5")



    
    
