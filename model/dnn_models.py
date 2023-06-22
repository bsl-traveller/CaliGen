import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_input

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as dnet_input

from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

import ml_collections

import pickle
import numpy as np

from model.model_utils import categorical_focal_loss

class DNN:
    '''
    Deep Neural Network Class
    '''
    def __init__(self, config: ml_collections.ConfigDict, last_layer_activation="softmax"):
        '''
        Initialise resnet class
        :param config: config dictionary
        :param last_layer_activation: Softmax for training and linear for prediction
        '''
        #if last_layer_activation == "linear":
        #    self.trainable = False
        #else:
        self.trainable = True
        self.config = config
        img_shape = (config.params.CROP, config.params.CROP, 3)
        img_input = Input(shape=img_shape)
        if config.model == 'resnet':
            self.model = ResNet101(include_top=False, weights='imagenet', input_tensor=img_input,
                                   input_shape=img_shape, pooling='avg',
                                   classes=config.params.NUM_CLASSES)
        elif config.model == 'efficientnet':
            e_input = efficientnet_input(img_input)
            self.model = EfficientNetV2B0(include_top=False, weights='imagenet',  input_tensor=e_input,
                                          input_shape=img_shape, pooling='avg',
                                          classes=config.params.NUM_CLASSES)
        elif config.model == 'densenet':
            dn_input = dnet_input(img_input)
            self.model = DenseNet121(include_top=False, weights='imagenet', input_tensor=dn_input,
                                      input_shape=img_shape, pooling='avg',
                                      classes=config.params.NUM_CLASSES)

        #i = self.model.input
        i = [img_input]
        o = Dense(config.params.NUM_CLASSES, activation=last_layer_activation, name="Prediction",
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY))(self.model.layers[-1].output)
        self.model = tf.keras.models.Model(inputs=i, outputs=[o])



    def summary(self):
        print(self.model.summary())

    def compile(self, loss, optimizer, metrics='accuracy'):
        '''
        compile method for resnet class
        :param loss: Loss function
        :param optimizer: Optimizer for network
        :param metrics: Optimize on metrics
        '''
        if loss == 'focal':
            l = categorical_focal_loss
        elif loss == 'crossentropy':
            l = 'categorical_crossentropy'
        else:
            raise ValueError(
                f"Loss should be either 'crossentropy' or 'focal' "
                f"Please set through config.loss "
                f"The default is 'crossentropy'"
            )
        if self.trainable:
            self.model.compile(loss=l, optimizer=optimizer, metrics=[metrics], run_eagerly=True)
        else:
            print("Model is not trainable. It must be used for prediction or calibration")

    def fit(self, ds_train, ds_valid, weight_file, model_history=None):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_file,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True,
                                                                       save_weights_only=True)
        callbacks = [model_checkpoint_callback]

        if self.config.PATIENCE < self.config.EPOCHS:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.config.PATIENCE)
            callbacks.append(es)

        # start training
        self.hist = self.model.fit(ds_train['data'], ds_train['label'], batch_size=self.config.BATCH_SIZE,
                                    epochs=self.config.EPOCHS,
                                    callbacks=callbacks,
                                    validation_data=(ds_valid['data'], ds_valid['label']))

        if model_history is not None:
            print("Pickle model history")
            with open(model_history, 'wb') as f:
                pickle.dump(self.hist.history, f)

    def evaluate(self, ds, verbose=0):
        loss, accuracy = self.model.evaluate(ds['data'], ds['label'], verbose=verbose)
        return (loss, accuracy)

    def predict(self, x_test, batch_size=512): #, verbose=1):
        #x = tf.keras.backend.constant(x_test)
        logits_dim = self.model.layers[-1].output.shape[-1]
        representation_dim = self.model.layers[-2].output.shape[-1]
        outputs = [self.model.layers[-1].output, self.model.layers[-2].output]

        n_batches = int(np.ceil(x_test.shape[0] / float(batch_size)))
        logits = np.zeros(shape=(len(x_test), logits_dim))
        representation = np.zeros(shape=(len(x_test), representation_dim))

        layers_fn = tf.keras.backend.function([self.model.input], outputs)

        for i in range(n_batches):
            print("Predicting for batch no.: ", i)
            [logits[i * batch_size:(i + 1) * batch_size], representation[i * batch_size:(i + 1) * batch_size]] = layers_fn([x_test[i * batch_size:(i + 1) * batch_size]])
        # return self.model.predict(x_test, verbose=verbose)
        return [logits, representation]

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file) #.expect_partial()
