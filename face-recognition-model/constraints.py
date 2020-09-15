from os import path, getcwd
from keras import backend as K

class Constraints:
    
    # About Image Shape
    IMAGE_WIDTH = 150 # 1368
    IMAGE_HEIGHT = 150 # 1000
    IMAGE_DEPTH = 3
    
    # Epochs and batch sizes
    EPOCHS = 10
    BATCH_SIZE = 25
    
    # test, train and validation data dir
    TRAIN_DATA_DIR =  path.join(getcwd(), 'dataset/train/')
    VALIDATION_DATA_DIR = path.join(getcwd(), 'dataset/validation/')
    TEST_DATA_DIR = path.join(getcwd(), 'dataset/test/')
    
    # train and validation sample size, and total classes
    TRAIN_SAMPLE = 1200
    VALIDATION_SAMPLE = 200
    CLASSES = 3 # total number of different object
    
    INIT_LR = 1e-3 # Initial learning rate