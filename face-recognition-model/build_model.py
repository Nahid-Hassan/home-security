# import the necessary packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from constraints import Constraints
from keras import backend as K


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
        self.model.add(Conv2D(32, (3, 3), input_shape=self.image_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3))
        self.model.add(Activation('sigmoid'))

        return self.model
