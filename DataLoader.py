import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    filename=os.path.join("logs", ''), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

#custom exception class
class DataLoaderException(Exception):
    pass


class Dataset:
    def __init__(self,root : str = None) -> None:   
        logging.info("Initializing DataLoader")
        self.root = root
        self.label_encoder = LabelEncoder()
        pass

    def read_images(self,size : tuple = (224,224),gray : bool = False):
        # Read all images in the directory
        images, labels = [],[]
        for label in os.listdir(self.root):
            for filename in os.listdir(self.root + '/' +label):
                try:
                    img = cv2.imread(self.root + '/' + label + '/' + filename,cv2.IMREAD_COLOR)
                    img = cv2.resize(img,size)
                    images.append(img)
                    labels.append(label)
                    logging.info("Read image: " + filename)
                except:
                    raise DataLoaderException('Error reading image')
                
        if gray:
            images = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in images]
        logging.info("Image loaded {}".format('grayscale' if gray else 'color'))
        return np.array(images),np.array(labels)

    def normalize(self,images):
        # Normalize the images
        images = images / 255.0
        return images

    def to_categorical(self,labels):
        logging.info("converted to categorical")
        # Convert labels to categorical
        labels = self.label_encoder.fit_transform(labels)
        return labels
    
    def to_class(self,labels):
        # Convert labels to class
        labels = self.label_encoder.inverse_transform(labels)
        return labels

    def plot_img(self,img):
        # Plot the image
        plt.imshow(img)
        plt.show()
    
    def plot_img_with_label(self,img,label):
        # Plot the image with label
        plt.imshow(img)
        plt.title(label)
        plt.show()

    def save_npy(self,images,label):
        np.save('X.npy',images)
        np.save('Y.npy',label)
        return f'X.npy and Y.npy saved'


class Process_image(Dataset):
    def __inti__(self,root : str = None) -> None:
        super().__init__(root)
        pass

    def imread(self,path : str,shape : tuple = (224,224)):
        # Read the image
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,shape)
        return Dataset.normalize(np.array(img)).reshape(1,shape[0],shape[1],3)

    

