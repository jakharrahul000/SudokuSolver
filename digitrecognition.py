import matplotlib.pyplot as plt
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.keras as keras

def preprocessSudokuDigitsAndLabels(digits,labels):
  preprocessedDigits, preprocessedLabels = [],[]

  for digit in digits:
    digit=cv2.normalize(digit,0,255,cv2.NORM_MINMAX)
    if labels is None:
      digit=digit.reshape(1,28,28,1)
    else:
      digit=digit.reshape(28,28,1)
    preprocessedDigits.append(np.float32(digit))
  del digits

  if type(preprocessedDigits) is list:
    preprocessedDigits=np.array(preprocessedDigits)

  if labels is None:
    return preprocessedDigits
  
  def convertLabelToOneHotVector(label):
    oneHotVector=[0.]*10
    oneHotVector[label]=1.
    return oneHotVector
  
  preprocessedLabels=[convertLabelToOneHotVector(label) for label in labels]
  del labels
  
  if len(preprocessedDigits)!=len(preprocessedLabels):
    raise AssertionError("Error in preprocessSudokuDigitsAndLabels. Length of digits nad labels need to be same")

  if type(preprocessedLabels) is list:
    preprocessedLabels=np.array(preprocessedLabels)

  return preprocessedDigits,preprocessedLabels

# digits=[]
# values=[]

# def load_dataset():
# 	(trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
# 	return trainX, trainY, testX, testY

# def create_dataset(trainX, trainY, testX, testY):
#   trainX, trainY = preprocessSudokuDigitsAndLabels(trainX, trainY)
#   testX, testY = preprocessSudokuDigitsAndLabels(testX, testY)

#   return trainX, trainY, testX, testY

# def define_model():
# 	model = keras.models.Sequential()
# 	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
# 	model.add(keras.layers.MaxPooling2D((2, 2)))
# 	model.add(keras.layers.Flatten())
# 	model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(keras.layers.Dense(10, activation='softmax'))

# 	opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
# 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model

# trainX, trainY, testX, testY = load_dataset()

# x_train=np.array(digits*40, np.uint8)
# y_train=np.array(values*40, np.uint8)

# trainX=np.append(trainX,x_train, 0)
# trainY=np.append(trainY,y_train)

# # don't forget to shuffle the trainX and trainY

# trainX, trainY, testX, testY=create_dataset(trainX, trainY, testX, testY)

# model = define_model()
# model.fit(trainX, trainY, epochs=6)

# model.save('num_detector.model')

new_model=keras.models.load_model('/content/drive/My Drive/Colab Notebooks/sudoku/num_detector.model')

def prediction(img_predict):
  pr=new_model.predict(img_predict)
  img_class=np.argmax(pr)
  return img_class

def sudoku(cellImages):
  cell_digits=[]

  for cellImage in cellImages:
    cellImage=preprocessSudokuDigitsAndLabels([cellImage], None)[0]
    result=prediction(cellImage)
    cell_digits.append(result)

  cell_digits = [cell_digits[i:i+9] for i in range(0, len(cell_digits), 9)]
  return cell_digits