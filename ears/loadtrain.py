#loadtrained.py
import json
import logging
import os
import time
import warnings

import librosa
import numpy as np
import numpy.core.multiarray
import pandas as pd
import pydub
import keras
import sklearn.preprocessing
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras.models import model_from_json
#from tensorflow.contrib import lite


pathtojson = 'model.json'

def GetModel(jsonfile):

    model = model_from_json(jsonfile)

    return model

if __name__ == '__main__':
    np.random.seed(1)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(os.getcwd())

    #model = GetModel(pathtojson)
    converter = tf.lite.TocoConverter.from_keras_model_file('./wizzle.h5')
    #converter = tf.lite.TFLiteConverter.from_saved_model('./wizzlewizzlewizzle.ckpt')
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS]
    #converter.data_format = 'NHWC'

    tflite_model = converter.convert()
    open("wizzel.tflite",'wb').write(tflite_model)


    with tf.Session() as sess:
  # Restore variables from disk.
        #model =saver.restore(sess, "./wizzlewizzlewizzle/model.ckpt")
        #model.save('backup.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model_file('./wizzle.h5')
        #converter = tf.lite.TFLiteConverter.from_saved_model('./wizzlewizzlewizzle.ckpt')
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                            tf.lite.OpsSet.SELECT_TF_OPS]
        #converter.data_format = 'NHWC'

        tflite_model = converter.convert()
        open("wizzel.tflite",'wb').write(tflite_model)


    #attempt to convert to tf.graph
    #to apply to android device
    # model_wizzle = keras.models.load_model('wizzle.h5')
    # model_wizzle.load_weights('model.h5')
    # model_wizzle.summary()

    #saved_model_path = tf.contrib.saved_model.save_keras_model(model_wizzle, "wizzleaved_models")
