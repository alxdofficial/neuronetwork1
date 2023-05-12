import tensorflow as tf
import keras.models
import numpy as np

# predict a single image
test_image = tf.keras.preprocessing.image.load_img('cnn data/single_prediction/img.png', target_size=(150, 150))
#  here test size must be equal to the same with what wee trained with
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
# we have to duplicate the image into a batch of 32 because our network was trained with 32 image batches
test_image = np.expand_dims(test_image,axis=0)

cnn = tf.keras.models.load_model("saved cnn 2")

# we must normalize the image, since we did that for training.
result = cnn.predict(test_image / 255.0) # here we will just get 0 or 1, we need to find out if 0 or 1 means cat or dog by
# printing the class indices
# print(training_set.class_indices)

# result[0][0] means first batch and first image in batch
# we found out by printing that 1 means dog an 0 means cat
if result[0][0] > 0.5:
    print("dog")
else :
    print("cat")