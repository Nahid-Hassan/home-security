import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import numpy as np
import argparse
from train_model import TrainModel
from keras.models import load_model

class ClassifyTestData:

    def classify_image(self):
        # TrainModel.train()
        save_model = load_model('kmodel.h5')

        l
        for file_name in file_names:
        	image_pred = image.load_img(file_name, target_size=(150,150))
        	image_pred = image.img_to_array(image_pred)
        	image_pred = np.expand_dims(image_pred, axis=0)
        	result = save_model.predict(image_pred)
        	print(result)

classify_test_data = ClassifyTestData()
classify_test_data.classify_image()
