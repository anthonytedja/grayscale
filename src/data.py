import os
from cv2 import imread, cvtColor, COLOR_BGR2Lab, resize
import numpy as np
from sklearn.model_selection import train_test_split
import config

class Data():
    def __init__(self):
        self.color_dir_path = config.COLOR_DIR_PATH
        self.gray_dir_path = config.GRAY_DIR_PATH

        self.color_filelist = os.listdir(self.color_dir_path)
        self.gray_filelist = os.listdir(self.gray_dir_path)

        self.color_filelist_train, self.color_filelist_test = train_test_split(self.color_filelist, test_size=0.2)
        self.gray_filelist_train, self.gray_filelist_test = train_test_split(self.gray_filelist, test_size=0.2)

        self.batch_size = config.BATCH_SIZE
        self.size = min(len(self.color_filelist_train), len(self.gray_filelist_train))

        self.data_index = 0

    def read_img(self, gray_filename, color_filename): 
        gray_img = imread(gray_filename, 3)
        color_img = imread(color_filename, 3)

        labimg = cvtColor(resize(gray_img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), COLOR_BGR2Lab)
        labimg_ori = cvtColor(color_img, COLOR_BGR2Lab)

        return np.reshape(labimg[:,:,0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), labimg[:, :, 1:], gray_img, labimg_ori[:,:,0]

    def generate_batch(self, train=True): 
        batch = []
        labels = []
        filelist = []
        labimg_oritList= []
        originalList = []

        color_filelist = self.color_filelist_train if train else self.color_filelist_test
        gray_filelist = self.gray_filelist_train if train else self.gray_filelist_test

        for i in range(self.batch_size):
            color_filename = os.path.join(self.color_dir_path, color_filelist[self.data_index])
            gray_filename = os.path.join(self.gray_dir_path, gray_filelist[self.data_index])

            filelist.append(color_filelist[self.data_index])

            greyimg, colorimg, original,labimg_ori = self.read_img(gray_filename, color_filename)

            batch.append(greyimg)
            labels.append(colorimg)
            originalList.append(original)
            labimg_oritList.append(labimg_ori)

            self.data_index = (self.data_index + 1) % self.size

        batch = np.asarray(batch)/255 # values between 0 and 1
        labels = np.asarray(labels)/255 # values between 0 and 1
        originalList = np.asarray(originalList)

        return batch, labels, originalList, labimg_oritList, filelist
