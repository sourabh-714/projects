from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.metrics import classification_report

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
import os



img_width = 150
img_height = 150



images_root = "/content/drive/My Drive/class"
train_root = os.path.join(images_root,'train')
eval_root = os.path.join(images_root,'validate')
test_root=os.path.join(images_root,'test')
train_samples = 120
validation_samples = 40
epochs = 5
batch_size = 8

# Check for TensorFlow or Thieno
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)




model = Sequential()
# Conv2D : Two dimenstional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Output convert into one dimension layer and will go to Dense layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))





import keras
from keras import optimizers
model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(lr=.0001),
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



print(train_generator.class_indices)


imgs, labels = next(train_generator)

from skimage import io

def imshow(image_RGB):
  io.imshow(image_RGB)
  io.show()

import matplotlib.pyplot as plt
%matplotlib inline
image_batch,label_batch = train_generator.next()

print(len(image_batch))
for i in range(0,len(image_batch)):
    image = image_batch[i]
    print(label_batch[i])
    imshow(image)
validation_generator = test_datagen.flow_from_directory(
    eval_root,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
print(validation_generator.class_indices)



history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)
model.save('/content/drive/My Drive/my_model.h5') 