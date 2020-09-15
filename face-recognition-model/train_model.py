# import the necessary packages
from collections import Counter
from os import getcwd, listdir, path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from build_model import BuildModel
from constraints import Constraints
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class TrainModel:

    # initialize basic parameter
    def __init__(self):
        build_model = BuildModel()
        self.model = build_model.build()

    # prepare dataset
    def prepare_image(self):
        # train datagen
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # test datagen
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        # train generator
        self.train_generator = self.train_datagen.flow_from_directory(
            Constraints.TRAIN_DATA_DIR,
            target_size=(150, 150),
            batch_size=Constraints.BATCH_SIZE,
            class_mode='categorical')

        # validation generator
        self.validation_generator = self.test_datagen.flow_from_directory(
            Constraints.VALIDATION_DATA_DIR,
            target_size=(150, 150),
            batch_size=Constraints.BATCH_SIZE,
            class_mode='categorical')

        self.test_generator = self.test_datagen.flow_from_directory(
            Constraints.VALIDATION_DATA_DIR,
            target_size=(150, 150),
            batch_size=Constraints.BATCH_SIZE,
            class_mode='categorical')

    # train model based on dataset
    def train(self):
        sleep(2)  # sleep 2 second
        self.model.summary()

        # self.opt = Adam(lr=Constraints.INIT_LR, decay=Constraints.INIT_LR / Constraints.EPOCHS)

        self.model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=.001),
                           metrics=["accuracy"])

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=Constraints.TRAIN_SAMPLE // Constraints.BATCH_SIZE,
            epochs=Constraints.EPOCHS,
            validation_data=self.validation_generator,
            validation_steps=Constraints.VALIDATION_SAMPLE // Constraints.BATCH_SIZE)

        self.model.save_weights('weights.h5')
        self.model.save('model.h5')

        # self.model.evaluate(self.test_generator)


train_model = TrainModel()
train_model.prepare_image()
train_model.train()
