import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os


class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = LogisticRegression()
        # self.clf = GaussianNB()

    def cluster(self):
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def develop_vocabulary(self, n_images, descriptor_list):
        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            L = len(descriptor_list[i])
            for j in range(L):
                idx = self.kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += L
        print("Vocabulary Histogram Generated")

    def standardize(self, std=None):
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def format_nd(self, L):
        vStack = np.array(L[0])
        for remaining in L[1:]:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return

    def train(self, train_labels):
        print("Training SVM")
        print(self.clf)
        print("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)
        return self.clf
        # print("Training completed")

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plot_hist(self, vocabulary=None):

        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class FileHelpers:

    def __init__(self):
        pass

    def get_files(self, path, trainOrTest):
        imlist = {}
        count = 0
        for each in os.listdir(path):
            imlist[each] = []
            for image_file in os.listdir(path + '/' + each + '/' + trainOrTest):
                im = cv2.imread(path + '/' + each + '/' + trainOrTest + '/' + image_file, 0)
                brightness = 10
                contrast = 2.3
                im = cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)
                kernel2 = np.ones((5, 5), np.float32) / 25

                im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
                imlist[each].append(im)
                count += 1

        return [imlist, count]

    def getfiles_s(self, path):
        imlist = {}
        count = 0
        for each in os.listdir(path):
            print(" #### Reading image category ", each, " ##### ")
            imlist[each] = []
            for imagefile in os.listdir(path + '/' + each):
                print("Reading file ", imagefile)
                im = cv2.imread(path + '/' + each + '/' + imagefile, 0)
                # Adjusts the brightness by adding 10 to each pixel value
                # brightness = 10
                # # Adjusts the contrast by scaling the pixel values by 2.3
                # contrast = 2.3
                # im = cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)
                # Creating the kernel with numpy
                kernel2 = np.ones((5, 5), np.float32) / 25

                # Applying the filter
                im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
                imlist[each].append(im)
                count += 1

        return [imlist, count]
