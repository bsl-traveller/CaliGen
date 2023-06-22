import os
from absl import logging
import ml_collections
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

from model.dnn_models import DNN
from data_utils import get_data, unpack_domains, get_filters_with_level, get_train_domains, get_calib_domains

def _data_exists_handler(data_exists: str, path: str, type: str):
    '''
    function data = {weights, logits} existence handler.
    :param data_exists: config.weights_exists or config.logits_exists
    :param path: wieghts or logits file path
    :param type: weights or logits
    :return: True if continue training for weights exists
    '''
    if data_exists == 'error':
        raise ValueError(
            f"{type} already exists at path: {path} "
            f"Either provide new path for {type} OR "
            f"To replace the {type}, set config.{type}_exists to 'replace'"
        )
    elif data_exists == 'continue' and type == 'weights':
        logging.info(f"Continue training from saved {type} at {path}")
        return True
    elif data_exists == 'replace':
        logging.info(f"Replacing saved {type} at {path}")
    else:
        if type == 'logits':
            s = f"config.logits_exists should be from ['error', 'replace']. "
        else:
            s = f"config.weights_exists should be from ['error', 'continue' ,'replace']. "
        raise ValueError(
            s + f"Please specify proper argument. Default is 'error' "
        )


def train(config: ml_collections.ConfigDict):
    '''
    train function to train the network for given dataset
    :param config: config
    '''
    if config.loss == 'focal' and config.dataset != 'OfficeHome':
        logging.info(
            f"We trained model only for OfficeHome dataset using focal loss in the paper. "
        )
    logging.info(
        f"Training model. "
        f"Dataset: {config.dataset}. "
        f"This could take some time....."
    )
    trainers = get_train_domains(config)
    for trainer in trainers:
        train_domains = []
        if config.dataset == 'cifar10c':
            sever = trainer.split('_')[1]
            train_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions, sever)
        elif config.dataset == 'OfficeHome':
            train_domains = [trainer]
        elif config.dataset == 'DomainNet':
            train_domains = trainer.split('_')

        filepath = os.path.join(config.weights_path, config.model, config.dataset, config.loss, trainer)
        Path(filepath).mkdir(parents=True, exist_ok=True)

        ds_train = get_data(config.dataset_dir, train_domains, 'train')
        ds_train = unpack_domains(ds_train)
        ds_train['label'] = tf.keras.utils.to_categorical(ds_train['label'], config.params.NUM_CLASSES)

        ds_valid = get_data(config.dataset_dir, train_domains, 'valid')
        ds_valid = unpack_domains(ds_valid)
        ds_valid['label'] = tf.keras.utils.to_categorical(ds_valid['label'], config.params.NUM_CLASSES)

        dnn_model = DNN(config)
        
        model_history = os.path.join(filepath, "model_history.p")
        weight_file = os.path.join(filepath, "model_weights.h5")

        # logging info for weights
        logging.info(f"Weights file: {weight_file} "
                     f"Model History: {model_history}")

        if os.path.exists(weight_file):
            if config.weights_exists == 'skip':
                continue
            elif _data_exists_handler(config.weights_exists, weight_file, 'weights'):
                dnn_model.load_weights(weight_file)  # .expect_partial()

        dnn_model.compile(loss=config.loss, optimizer='adam')

        dnn_model.fit(ds_train, ds_valid, weight_file=weight_file, model_history=model_history)

        # Load best weights
        dnn_model.load_weights(weight_file)

        loss, accuracy = dnn_model.evaluate(ds_valid)
        logging.info(
            f"Get accuracy on validation set:"
            f"Validation: accuracy = {accuracy};  loss = {loss}"
        )

        loss, accuracy = dnn_model.evaluate(ds_train)
        logging.info(
            f"Get accuracy on Training set:"
            f"Training: accuracy = {accuracy};  loss = {loss}"
        )


def train_limitation(config: ml_collections.ConfigDict):
    '''
    train function to train the network for given dataset with data also from calibration domains
    :param config: config
    '''
    if config.loss == 'focal':
        logging.info(
            f"We trained model only for crossentropy loss in limitation setting in the paper . "
        )
    elif config.loss == 'crossentropy' and config.dataset != 'OfficeHome':
        logging.info(
            f"We trained model only for OfficeHome dataset in limitation setting in the paper. "
        )
    logging.info(
        f"Training model in limitation setting. "
        f"Dataset: {config.dataset}. "
        f"This could take some time....."
    )
    trainers = get_train_domains(config)
    for trainer in trainers:
        calib_domains = get_calib_domains(config, trainer)
        for cd in calib_domains.keys():
            calib_d = calib_domains[cd]
            train_domains = []
            if config.dataset == 'cifar10c':
                sever = trainer.split('_')[1]
                train_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions, sever)
                filepath = os.path.join(config.calib_path, config.model, config.dataset, config.loss, 'limitation', trainer)
            elif config.dataset == 'OfficeHome':
                train_domains = [trainer]
                filepath = os.path.join(config.calib_path, config.model, config.dataset, config.loss, 'limitation', trainer, cd)
            elif config.dataset == 'DomainNet':
                train_domains = trainer.split('_')
                filepath = os.path.join(config.calib_path, config.model, config.dataset, config.loss, 'limitation', trainer, cd)

            logging.info(
                f"Training Domains: {train_domains} "
                f"Calibration Domain: {calib_d}"
            )

            Path(filepath).mkdir(parents=True, exist_ok=True)

            ds_train = get_data(config.dataset_dir, train_domains, 'train')
            ds_valid = get_data(config.dataset_dir, train_domains, 'valid')
            ds_calib = get_data(config.dataset_dir, calib_d, 'valid')

            for domain in ds_calib.keys():
                x_t, x_v, y_t, y_v = train_test_split(ds_calib[domain]['data'], ds_calib[domain]['label'],
                                                      test_size=0.5, random_state=6)
                ds_train[domain] = dict()
                ds_train[domain]['data'] = x_t
                ds_train[domain]['label'] = y_t
                ds_valid[domain] = dict()
                ds_valid[domain]['data'] = x_v
                ds_valid[domain]['label'] = y_v

            ds_train = unpack_domains(ds_train)
            ds_train['label'] = tf.keras.utils.to_categorical(ds_train['label'], config.params.NUM_CLASSES)

            ds_valid = unpack_domains(ds_valid)
            ds_valid['label'] = tf.keras.utils.to_categorical(ds_valid['label'], config.params.NUM_CLASSES)

            dnn_model = DNN(config)
            model_history = os.path.join(filepath, "model_history.p")
            weight_file = os.path.join(filepath, "model_weights.h5")

            # logging info for weights
            logging.info(f"Weights file: {weight_file} "
                         f"Model History: {model_history} "
                         f"Starting training, will take a while..."
                         )

            if os.path.exists(weight_file) and _data_exists_handler(config.weights_exists, weight_file, 'weights'):
                dnn_model.load_weights(weight_file)  # .expect_partial()

            # print(resnet.summary())
            dnn_model.compile(loss=config.loss, optimizer='adam')

            dnn_model.fit(ds_train, ds_valid, weight_file=weight_file, model_history=model_history)

            # Load best weights
            dnn_model.load_weights(weight_file)

            loss, accuracy = dnn_model.evaluate(ds_valid)
            logging.info(
                f"Get accuracy on validation set:"
                f"Validation: accuracy = {accuracy};  loss = {loss}"
            )

            loss, accuracy = dnn_model.evaluate(ds_train)
            logging.info(
                f"Get accuracy on Training set:"
                f"Training: accuracy = {accuracy};  loss = {loss}"
            )
