import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from keras.applications import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input, decode_predictions, ResNet152V2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, Activation, Input, UpSampling2D
from keras.activations import relu

CLASSES=10
INPUT_SHAPE=224
EPOCHS=5
RESIZE=7

def preprocess_data(inputs):
    '''
    :param inputs:
    :return:
    '''

    data=np.array(inputs)
    data=data.astype(np.float32)
    data=data/255
    data=preprocess_input(data)
    return data

def resize_data(inputs, size_mp):
    '''
    :param inputs:
    :param size_mp:
    :return:
    '''

    resized_data=UpSampling2D(size=(size_mp,size_mp))(inputs)
    return resized_data

def resnet_model(input_shape, inputs):
    '''
    :param input_shape: int
    :return: model: ResNet 152 V2 model without top layer
    '''

    #inputs=preprocess_data(inputs)
    #inputs=resize_data(inputs,size_mp=RESIZE)
    model=ResNet152V2(include_top=False, weights='imagenet', input_shape=(input_shape, input_shape,3))(inputs)
    return model

def output_classifier(inputs, classes):
    '''
    :param inputs:
    :param classes: number of classification classes
    :return: output: model top layer for classification
    '''

    c=GlobalAveragePooling2D()(inputs)
    c=Flatten()(c)
    c=Dense(2048)(c)
    c=Dropout(0.4)(c)
    c=Activation(relu)(c)
    c=Dense(1024)(c)
    c=Dropout(0.3)(c)
    c=Activation(relu)(c)
    c=Dense(512)(c)
    c=Dropout(0.2)(c)
    c=Activation(relu)(c)
    c=Dense(256, activation='relu')(c)
    output=Dense(classes, activation='softmax')(c)
    return output

def model_compile(classes, input_shape):
    '''
    :param classes:
    :param input_shape:
    :return:
    '''

    input_layer = Input(shape=(32, 32, 3))
    resnet=resnet_model(input_shape=input_shape,inputs=input_layer)
    output_layer=output_classifier(resnet,classes=classes)

    model=Model(inputs=input_layer,outputs=output_layer)
    model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

model=model_compile(classes=CLASSES,input_shape=INPUT_SHAPE)
print(model.summary())


#history=model.fit()
