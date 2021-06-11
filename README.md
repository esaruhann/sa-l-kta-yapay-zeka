# saglıkta_yapay_zeka
from keras import applications
import os
from keras.applications.vgg19 import VGG19, preprocess_input
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
import glob
import os
from PIL import Image

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator

from numba import jit, cuda




# train data çekme
imgs = glob.glob('normal_train/*.png')

imgs1 = glob.glob('iskemi_train/*.png')

imgs2 = glob.glob('kanama_train/*.png')

#imgs = os.listdir("C:\\Users\\EDANUR\\Desktop\\normal_train")

width=100
height=100
X=[]

for img in imgs:

  filename = os.path.basename(img)
  im = np.array(Image.open(img).convert("L").resize((width,height)))
  im = im/255
  X.append(im)

for img1 in imgs1:

  filename = os.path.basename(img1)
  im = np.array(Image.open(img1).convert("L").resize((width,height)))
  im = im/255
  X.append(im)

for img2 in imgs2:

  filename = os.path.basename(img2)
  im = np.array(Image.open(img2).convert("L").resize((width,height)))
  im = im/255
  X.append(im)

X=np.array(X)
#print(X)
X=X.reshape(X.shape[0],width,height,1)

#validation data çekme

imgs = glob.glob('normal_test/*.png')

imgs1 = glob.glob('iskemi_test/*.png')

imgs2 = glob.glob('kanama_test/*.png')

#imgs = os.listdir("C:\\Users\\EDANUR\\Desktop\\normal_train")

width=100
height=100
X_test=[]

for img in imgs:

  filename = os.path.basename(img)
  im = np.array(Image.open(img).convert("L").resize((width,height)))
  im = im/255
  X_test.append(im)

for img1 in imgs1:

  filename = os.path.basename(img1)
  im = np.array(Image.open(img1).convert("L").resize((width,height)))
  im = im/255
  X_test.append(im)

for img2 in imgs2:

  filename = os.path.basename(img2)
  im = np.array(Image.open(img2).convert("L").resize((width,height)))
  im = im/255
  X_test.append(im)

X_test=np.array(X_test)
#print(X)
X_test=X_test.reshape(X_test.shape[0],width,height,1)

# re-size all the images to 100x100
IMAGE_SIZE = [100, 100] 

# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
#for layer in vgg.layers:
#    layer.trainable = False

for layer in vgg.layers[15:]:
    layer.trainable = True
    

jit(target='gpu')

  
# our layers
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
# three hidden layers
x = keras.layers.Dense(1000, activation='relu')(x)
x = keras.layers.Dense(1000, activation='relu')(x)
x = keras.layers.Dense(1000, activation='relu')(x)
x = keras.layers.Dense(1000, activation='relu')(x)
x = keras.layers.Dense(1000, activation='relu')(x)
prediction = Dense(3, activation='softmax')(x)


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
# model.compile(
#   loss='categorical_crossentropy',
#   optimizer='rmsprop',
#   metrics=['accuracy']
# )
#fine tuning
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

# train_label
A=np.zeros(3000)
#kanama yok
a=0
for i in range(0,1000):
  A[i] = a

#kanama var (inme)
b=1
for i in range(1001,2000):
   A[ i] = b

#kanama var(iskemik)
c=2
for i in range(2001,3000):
   A[i] = c

A = A.reshape(3000,1)
#A = A.reshape(-1,)
#a = Model.fit(X,A, epochs=500)
X = np.array(X)
A = np.array(A)

#test_label

A_test=np.zeros(300)
#kanama yok
a=0
for i in range(0,100):
  A_test[i] = a

#kanama var (inme)
b=1
for i in range(101,200):
   A_test[ i] = b

#kanama var(iskemik)
c=2
for i in range(201,300):
   A_test[i] = c

A_test = A_test.reshape(300,1)
#A = A.reshape(-1,)
#a = Model.fit(X,A, epochs=500)
X_test = np.array(X_test)
A_test = np.array(A_test)


# training config:
epochs = 1
batch_size = 32

train_path = 'C:\\Users\\EDANUR\\Desktop\\train'
valid_path = 'C:\\Users\\EDANUR\\Desktop\\test'
# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=3000 // batch_size,
  validation_steps=300 // batch_size,
)

import pickle              # import module first

f = open('cnn_model_kayit.pkl', 'wb')   # Pickle file is newly created where foo1.py is
pickle.dump(r, f, -1)          # dump data to f
f.close()    

'''
import pickle

dosya = "cnn_model_kayit"

pickle.dump(r, open(dosya, 'wb'))

yuklenen = pickle.load(open(dosya, 'rb'))
'''
