import os.path
import argparse
import os
import os.path
import random
import shutil
from shutil import copyfile
import re

import yaml


class DataLoader:
    def __init__(self, config_path, dataset_type="YOLOv5", train_proportion=0.7, valid_proportion=0.3):
        self.config_path = config_path
        self.dataset_type = dataset_type
        self.path = '../data/seal_full'
        self.train_proportion = train_proportion
        self.valid_proportion = valid_proportion
        self.all_number = 0
        self.train_number = 0
        self.valid_number = 0
        self.test_number = 0

        # first read configuration file .yaml to get train/valid/test set
        with open(self.config_path) as config:
            configs = yaml.safe_load(config)  # load hyps

        try:
            self.path = configs["path"]
        finally:
            pass
        self.train = configs['train']
        self.val = configs['val']
        self.nc = configs['nc']
        self.names = configs['names']

    def get_sets_number(self):
        return self.train_number, self.valid_number, self.test_number

    @staticmethod
    def train_initial():
        # make configuration file .yaml for training
        with open("data/my_data.yaml", "w+") as config:
            config.seek(0)
            # config.write("path: data/my_data/\n")  # train images (relative to 'path')
            config.write("train: data/my_data/train/images \n")  # train images (relative to 'path')
            config.write("val: data/my_data/valid/images \n")
            config.write("\n")
            config.write("nc: 3 \n")
            config.write("names: ['Baby', 'Female', 'Male']  # class name")
            config.truncate()

        try:
            exps = os.listdir("runs/train")
            for exp in exps:
                if re.search("exp.*", exp) is not None:
                    shutil.rmtree("runs/train/" + exp)
        except:
            pass
    def distribute_k_folder(self):
        """
        This function will build a k-folder validation for training.
        """

        def images_list(path, type):
            """
            This function take path as param, returns how many lines in the file depend of the dataset type
            """
            image_list = []
            image_path = path
            # label_path = path + '/labels'
            if type == 'YOLOv5':
                if os.path.isdir(image_path):
                    for name in os.listdir(image_path):
                        image_list.append(name)
                else:
                    print("path is not a folder")
                    exit()
            return image_list

        try:
            shutil.rmtree("data/my_data/")
        except:
            pass

        # mkdir for train data
        os.makedirs("data/my_data/train/images")
        os.makedirs("data/my_data/train/labels")
        os.makedirs("data/my_data/valid/images")
        os.makedirs("data/my_data/valid/labels")
        os.makedirs("data/my_data/test/images")
        os.makedirs("data/my_data/test/labels")

        # get images and distribute dataset randomly
        images = images_list(self.path + '/' + self.train, self.dataset_type)
        self.all_number = len(images)
        self.train_number = round(self.train_proportion * self.all_number)
        self.valid_number = round(self.valid_proportion * self.all_number)
        self.test_number = self.all_number - self.train_number - self.valid_number
        random.shuffle(images)

        # move images and labels to train, valid and test set
        for image in images[0: self.train_number]:
            copyfile(self.path + "/images/" + image, "data/my_data/train/images/" + image)
            copyfile(self.path + "/labels/" + os.path.splitext(image)[0] + ".txt",
                     "data/my_data/train/labels/" + os.path.splitext(image)[0] + ".txt")

        for image in images[self.train_number + 1: self.train_number + self.valid_number]:
            copyfile(self.path + "/images/" + image, "data/my_data/valid/images/" + image)
            copyfile(self.path + "/labels/" + os.path.splitext(image)[0] + ".txt",
                     "data/my_data/valid/labels/" + os.path.splitext(image)[0] + ".txt")

        for image in images[self.train_number + self.valid_number + 1: -1]:
            copyfile(self.path + "/images/" + image, "data/my_data/test/images/" + image)
            copyfile(self.path + "/labels/" + os.path.splitext(image)[0] + ".txt",
                     "data/my_data/test/labels/" + os.path.splitext(image)[0] + ".txt")


if __name__ == '__main__':
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help='data configuration path for training')
    parser.add_argument("--dataset_type", type=str, default="YOLOv5", help='dataset type')
    parser.add_argument("--train_proportion", type=int, default=0.7, help='proportion of training data')
    parser.add_argument("--valid_proportion", type=int, default=0.3, help='proportion of validation data')
    parser.add_argument("--epoch", type=int, default=3, help='epoch number, must be multiple of 10')
    parser.add_argument("--mini_epoch", type=int, default=100, help='epoch number, must be multiple of 10')
    parser.add_argument("--batch", type=int, default=8, help='mini batch')
    parser.add_argument("--weights", type=str, default="yolov5m.pt", help='data configuration path for training')
    ops = parser.parse_args()

    dataset = DataLoader(ops.config_path)

    # train model

    dataset.train_initial()  # initial dataset, delete previous run and make new folder for my_data test
    for i in range(0, ops.epoch):
        dataset.distribute_k_folder()  # each epoch reshuffle all the images and rebuild dataset

        if i == 0:
            os.system("python3 train.py --img 640 --batch " + str(ops.batch) + " --epochs " + str(
                ops.mini_epoch) + " --data data/my_data.yaml --weights " + ops.weights)
        elif i == 1:
            os.system("python3 train.py --img 640 --batch " + str(ops.batch) + " --epochs " + str(
                ops.mini_epoch) + " --data data/my_data.yaml --weights " + "runs/train/exp/weights/best.pt")
        else:
            os.system("python3 train.py --img 640 --batch " + str(ops.batch) + " --epochs " + str(
                ops.mini_epoch) + " --data data/my_data.yaml --weights " + "runs/train/exp" + str(
                i) + "/weights/best.pt")

        print("######################################################")
        print("###################### epoch " + str(i) + " end ################")
        print("######################################################")

