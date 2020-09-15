# import the necessary packages
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
# from keras.models import Sequential
from constraints import Constraints
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

class BuildModel:

    def build(self):
        # determined image shape
        if K.image_data_format() == 'channels_first':
            self.image_shape = (Constraints.IMAGE_DEPTH,
                                Constraints.IMAGE_WIDTH, Constraints.IMAGE_HEIGHT)
        else:
            self.image_shape = (
                Constraints.IMAGE_WIDTH, Constraints.IMAGE_HEIGHT, Constraints.IMAGE_DEPTH)

        # build a model
        self.model = Sequential()

        self.model.add(
            Conv2D(32, (3, 3), padding="same", input_shape=self.image_shape))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        self.model.add(Conv2D(64, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(64, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        self.model.add(Conv2D(128, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(128, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        # softmax classifier
        self.model.add(Dense(Constraints.CLASSES))
        self.model.add(Activation("softmax"))

        # return the constructed network architecture
        return self.model


# build_model = BuildModel()
# build_model.build()
