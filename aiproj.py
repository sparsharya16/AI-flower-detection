from warnings import filters
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 



print("\n\n\n\n\n\n\n")

train_datagen = ImageDataGenerator(
rescale = 1. / 255,
shear_range = 0.2 ,
zoom_range = 0.2 ,
horizontal_flip = True
)

training_set = train_datagen.flow_from_directory('train',target_size=(64,64),batch_size = 32 , class_mode = 'categorical' )

test_datagen = ImageDataGenerator(rescale = 1. / 255)
test_set = test_datagen.flow_from_directory('test',target_size = (64,64) , batch_size = 32 , class_mode = 'categorical')


# building model

cnn = tf.keras.models.Sequential()

#buliding convoltional layer
cnn.add(tf.keras.layers.Conv2D( filters = 64 , kernel_size = 3, activation = 'relu', input_shape = [64 , 64  , 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2 , strides = 2))
cnn.add(tf.keras.layers.Conv2D( filters = 64 , kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2 , strides = 2))


cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128 , activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units=5,activation = 'softmax'))
cnn.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics =['accuracy'])

# epochs is like loop , in below operation total its going to be 30 times loop going to iterate with both dataset.
cnn.fit(training_set,validation_data = test_set,epochs = 1)

test_image = image.load_img('prediction/daisy.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
print(result)
if result[0][0] == 1:
    print("Daisy")
elif result[0][1] == 1 :
    print("Dandelion")
elif result[0][2] == 1 :
    print("rose")
elif result[0][3] == 1 :
    print("Sunflower")
elif result[0][4] == 1 :
    print("Tulip")
else:
    print(" Not predictable")
