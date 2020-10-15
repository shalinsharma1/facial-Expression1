# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 02:41:02 2020

@author: kunal
"""

from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

#Below is to setup the GPU for tensorflow session
'''
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.15
session=tf.compat.v1.Session(config=config)
'''

class FacialExpressionModel(object):
    EMOTIONS_LIST=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise","Neutral"]
    
    def __init__(self,model_json_file,model_weights_file):
        with open(model_json_file,"r") as json_file:
            loaded_model_json=json_file.read()
            self.loaded=model_from_json(loaded_model_json)
        print("Model loaded")
        self.loaded.load_weights(model_weights_file)
        print("wts not loaded")
        self.loaded._make_predict_function()
        
    def predict_emotion(self,img):
        self.preds=self.loaded.predict(img)
        ##DOUBT
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
    
    