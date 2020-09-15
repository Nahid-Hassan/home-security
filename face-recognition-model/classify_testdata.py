import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import numpy as np
import argparse
from keras.models import load_model
import os
import time

class ClassifyTestData:

    # ==========================================================================
    #                         Development Phase
    # ==========================================================================
    def classify_image(self):
        # TrainModel.train()
        self.save_model = load_model('model.h5')

        # working with python argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('file_location', help='location: image location', type=str)
        parser.add_argument('-s', '--summary', help='Show the summary of the model', action='store_true')
        parser.add_argument("-d", "--dataset", help="path: input dataset (i.e., directory of images)")
        parser.add_argument("-m", "--model", required=True, help="path to output model")
        parser.add_argument("-p", "--plot", type=str, default="plot.png", help="path: accuracy plot")

        args = parser.parse_args()

        file_name = str(args.file_location)

        if args.summary:
            self.save_model.summary()

        # load image
        if len(file_name) > 0: 
            image_pred = image.load_img(file_name, target_size=(150,150))
            # plt.imshow(image_pred)
            # plt.show()
            image_pred = image.img_to_array(image_pred)
            image_pred = np.expand_dims(image_pred, axis=0)

            # predict result
            result = save_model.predict(image_pred)

            # simply show the predicted list
            print(result)

    # ======================================================================================
    #                            Complete method
    # ======================================================================================
    def classify(self):
        time.sleep(5) # sleep 2 second
        model = load_model('model.h5')

        # file = open('pred.txt', 'a')

        file_names = os.listdir(
            '/media/nahid/data-center/project-work/home_security/rt_test_data/')

        for file_name in file_names:
            f = file_name
            name = os.path.join(os.getcwd(), 'rt_test_data')
            file_name = os.path.join(name, file_name)
            image_pred = image.load_img(str(file_name), target_size=(150, 150))
            image_pred = image.img_to_array(image_pred)
            image_pred = np.expand_dims(image_pred, axis=0)

            # predict result
            result = model.predict(image_pred)

            # # simply show the predicted list
            # if result[0][0]:
            #     file.write(str(str(f) + " --> " + str('Abir') + '\n'))
            # elif result [0][1]:
            #     file.write(str(str(f) + " --> " + str('Bobi') + '\n'))
            # elif result[0][2]:
            #     file.write(str(str(f) + " --> " + str('Rafi') + '\n'))

            print(f, result)


classify_test_data = ClassifyTestData()
# classify_test_data.classify_image() 
classify_test_data.classify()
