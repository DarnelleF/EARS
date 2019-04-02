# -*- coding: utf-8 -*-

import json
import logging
import os
import time
import warnings
import tensorflow as tf
import librosa
import numpy as np
import numpy.core.multiarray
import pandas as pd
import pydub
import sklearn.preprocessing
from tqdm import tqdm
from keras import backend as K


THEANO_FLAGS = ('device=cuda*,'
                'floatX=float32,'
                'dnn.conv.algo_bwd_filter=deterministic,'
                'dnn.conv.algo_bwd_data=deterministic')

os.environ['THEANO_FLAGS'] = THEANO_FLAGS
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.callbacks import ModelCheckpoint
keras.backend.set_image_dim_ordering('th')
from keras.layers.convolutional import Conv2D as Conv
from keras.layers.convolutional import MaxPooling2D as Pool
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.regularizers import l2 as L2


from config import *


def to_one_hot(targets, class_count):
    """Encode target classes in a one-hot matrix.
    """
    one_hot_enc = np.zeros((len(targets), class_count))

    for r in range(len(targets)):
        one_hot_enc[r, targets[r]] = 1

    return one_hot_enc


def extract_segment(filename):
    """Get one random segment from a recording.
    """
    spec = np.load('dataset/tmp/' + filename + '.spec.npy').astype('float32')

    offset = np.random.randint(0, np.shape(spec)[1] - SEGMENT_LENGTH + 1)
    spec = spec[:, offset:offset + SEGMENT_LENGTH]

    return np.stack([spec])


def iterrows(dataframe):
    """Iterate over a random permutation of dataframe rows.
    """
    while True:
        for row in dataframe.iloc[np.random.permutation(len(dataframe))].itertuples():
            yield row


def iterbatches(batch_size, training_dataframe):
    """Generate training batches.
    """
    itrain = iterrows(training_dataframe)

    while True:
        X, y = [], []

        for i in range(batch_size):
            row = next(itrain)
            X.append(extract_segment(row.filename))
            y.append(le.transform([row.category])[0])

        X = np.stack(X)
        y = to_one_hot(np.array(y), len(labels))

        X -= AUDIO_MEAN
        X /= AUDIO_STD

        yield X, y


if __name__ == '__main__':
    np.random.seed(1)
    with tf.Session() as sess:
        sess = tf.Session()
        K.set_session(sess)

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Load dataset
        meta = pd.read_csv('dataset/dataset.csv')
        labels = pd.unique(meta.sort_values('category')['category'])
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)

        # Generate spectrograms
        logger.info('Generating spectrograms...')

        if not os.path.exists('dataset/tmp/'):
            os.mkdir('dataset/tmp/')

        for row in tqdm(meta.itertuples(), total=len(meta)):
            spec_file = 'dataset/tmp/' + row.filename + '.spec.npy'
            audio_file = 'dataset/audio/' + row.filename
            #audio_file = row.filename
            print(audio_file)
            print(SAMPLING_RATE)


            if os.path.exists(spec_file):
                continue

            audio = pydub.AudioSegment.from_file(audio_file).set_frame_rate(SAMPLING_RATE).set_channels(1)
            #audio = pydub.AudioSegment.from_file(audio_file).set_frame_rate(SAMPLING_RATE)
            audio = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)

            spec = librosa.feature.melspectrogram(audio, SAMPLING_RATE, n_fft=FFT_SIZE,
                                                  hop_length=CHUNK_SIZE, n_mels=MEL_BANDS)
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')  # Ignore log10 zero division
            #     spec = librosa.core.perceptual_weighting(spec, MEL_FREQS, amin=1e-5, ref_power=1e-5,
            #                                              top_db=None)

            spec = np.clip(spec, 0, 100)
            np.save(spec_file, spec.astype('float16'), allow_pickle=False)

        # Define model
        logger.info('Constructing model...')

        input_shape = 1, MEL_BANDS, SEGMENT_LENGTH
        #print(input_shape.str())

        model = keras.models.Sequential()
        #model.K.set_session(sess)

        model.add(Conv(80, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform',
                       input_shape=input_shape))
        model.add(LeakyReLU())
        model.add(Pool((3, 3), (3, 3)))

        model.add(Conv(160, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
        model.add(LeakyReLU())
        model.add(Pool((3, 3), (3, 3)))

        model.add(Conv(240, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
        model.add(LeakyReLU())
        model.add(Pool((3, 3), (3, 3)))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(len(labels), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
        model.add(Activation('softmax'))
        #changed the optimizer

        optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train model
        batch_size = 100
        EPOCH_MULTIPLIER = 1
        epochs = 1000 // EPOCH_MULTIPLIER
        epoch_size = len(meta) * EPOCH_MULTIPLIER
        bpe = epoch_size // batch_size

        logger.info('Training... (batch size of {} | {} batches per epoch)'.format(batch_size, bpe))
        #cb = ModelCheckpoint('wizzlewizzle1')
        model.fit_generator(generator=iterbatches(batch_size, meta),
                            steps_per_epoch=bpe,
                            epochs=epochs)

        with open('wizzle.json', 'w') as file:
            file.write(model.to_json())

        model.save('wizzle.h5')
        #model.save_model('final.h5')
        #model.save_weights('model.h5')

        # Save tf.keras model in HDF5 format.
    #     xport_dir = './weight/'
        saver = tf.train.Saver()


        loc = saver.save(sess,"./wizzlewizzlewizzle/model.ckpt")

        # save_graph_to_file(sess,  export_dir + "flower5.pb", "wiizle")

        #converter = tf.lite.TFLiteConverter.from_saved_model(loc)



        keras_file = "keras_wizzle.h5"
        #tf.keras.models.save_model( "keras_wizzle.h5")

        # Convert to TensorFlow Lite model.
        converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('./wizzle.h5')
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)




# def save_graph_to_file(sess,  graph_file_name, output_names):
#     output_graph_def = graph_util.convert_variables_to_constants(
#       sess,  sess.graph.as_graph_def(),  output_names)
#     with gfile.FastGFile(graph_file_name, 'wb') as f:
#         f.write(output_graph_def.SerializeToString())
