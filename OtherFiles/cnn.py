# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 01:45:18 2017

@author: Turzo
"""



from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


classifier.add(MaxPooling2D(pool_size = (2, 2)))




classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units=3, activation = 'relu'))


classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


classifier.fit_generator(training_set,
                         steps_per_epoch = 20,
                         epochs =10,
                         validation_data = test_set,
                         validation_steps = 20)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
result = classifier.predict(test_image)


    
