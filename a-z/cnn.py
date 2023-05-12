import keras.models
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# set tensorflow to use gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# to preprocess the images, we should do some translations, rotations, and zooms to the images
# this is called image augmentation

# generate training set
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

#original target size was 64 by 64
training_set = train_datagen.flow_from_directory(
    'cnn data/training_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
)
# if multiple classes the class mode = categorical.
# here the class names are the folder names within the dataset folder

# now generate test set. this is the same transformation that will be applied to productoin imaegs
test_datagen = ImageDataGenerator(rescale=1 / 255)
test_set = test_datagen.flow_from_directory(
    'cnn data/test_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# initialize the cnn
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150, 150, 3]))
# filters is num of output neurons in convolution layer, 32 is classic number
# 3x3 kernel, input shape is 64 * 64 pixels, 3 color channels rgb

# pool 1 (max pooling , 2x2 box)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# add 2nd convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# extra code, remove if broken
# pool 2 (max pooling , 2x2 box)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# add 3rd convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))


# flattening
cnn.add(tf.keras.layers.Flatten())

# fully connected layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# extra addded code, remove if broke
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# if multiple classes, softmax


# compile the cnn, specify stochastic gradient decent
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train
cnn.fit(x=training_set, validation_data=test_set, epochs=30)

print(training_set.class_indices)
cnn.save("saved cnn 2")


