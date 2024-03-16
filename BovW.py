import pickle

import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.test_path_predict = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        self.images, self.trainImageCount = self.file_helper.get_files(self.train_path, 'Train')
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("Computing Features for ", word)
            for im in imlist:
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)

            label_count += 1
        self.bov_helper.format_nd(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.develop_vocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)
        self.bov_helper.plot_hist()

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)
        predictions = []
        correctClassifications = 0
        for word, imlist in self.images.items():
            print("processing ", word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })

                if self.name_dict[str(int(cl[0]))] == word:
                    correctClassifications = correctClassifications + 1

        print("Train Accuracy = " + str((correctClassifications / self.trainImageCount) * 100))

    def recognize(self, test_img, test_image_path=None):
        kp, des = self.im_helper.features(test_img)
        print(des.shape)
        vocab = np.array([[0 for i in range(self.no_clusters)]])
        vocab = np.array(vocab, 'float32')
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        for each in test_ret:
            vocab[0][each] += 1
        vocab = self.bov_helper.scale.transform(vocab)
        lb = self.bov_helper.clf.predict(vocab)
        return lb

    def test_model(self):
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.get_files(self.test_path, 'Validation')

        predictions = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in self.testImages.items():
                # print imlist[0].shape, imlist[1].shape
                print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })

                if self.name_dict[str(int(cl[0]))] == word:
                    correctClassifications = correctClassifications + 1

        print("Test Accuracy = " + str((correctClassifications / self.testImageCount) * 100))
        for each in predictions:
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()

    def test_model_predict(self):
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.getfiles_s(self.test_path_predict)

        predictions = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in imlist:
                #print(imlist[0].shape, imlist[1].shape)
                #print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })

                if self.name_dict[str(int(cl[0]))] == word:
                    correctClassifications = correctClassifications + 1

        print("Test Accuracy = " + str((correctClassifications / self.testImageCount) * 100))
        # print (predictions)
        for each in predictions:
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()

    def print_vars(self):
        pass


if __name__ == '__main__':
    dir_path = "Product Classification"
    bov = BOV(no_clusters=2000)
    bov.train_path = dir_path
    bov.test_path = dir_path
    bov.test_path_predict = 'Test Samples Classification'
    bov.trainModel()
    bov.test_model()
    """filename = "trained_model_1.plk"
    pickle.dump(bov.trainModel() , open(filename,'wb'))
    bov.test_model_predict()"""
