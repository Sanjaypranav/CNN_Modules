import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
import numpy as np

class CNN:
    def __init__(self) -> None:
        self.model = Sequential()
        self.base_model = VGG16(weights = 'imagenet',input_shape = (224,224,3),include_top = False)
        for layer in self.base_model.layers:
            layer.trainable = False
        pass

    def build_model(self):
        self.model.add(self.base_model)
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = "relu"))
        self.model.add(Dense(2))
        self.model.add(Activation("softmax"))
        return self.model
    


