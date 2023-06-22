import numpy as np
import os
import ml_collections
from tensorflow import keras

from model.model_utils import softmax
from calibration.cal_methods import TemperatureScaling

def get_filters_with_level(domains: list, sever: str):
    filters_with_level = []
    for fil in domains:
        if fil != 'original':
            if sever == 'a':
                for i in range(1, 6):
                    filters_with_level.append(fil + '_' + str(i))
            else:
                filters_with_level.append(fil + '_' + sever)
    return filters_with_level


def get_train_domains(config: ml_collections.ConfigDict):
    trainers = []
    if config.dataset == 'cifar10c':
        for sever in config.severity:
            if sever not in '12345a':
                raise ValueError(
                    f"Severity should be in [1, 2, 3, 4, 5] or a (if all level of severity is considered) "
                    f"You give severity: {sever}, which is not in  level of severity")
            else:
                trainers.append(f"4filter_{sever}")
        return trainers
    elif config.dataset == 'OfficeHome':
        for domain in config.domains:
            trainers.append(domain)
        return trainers
    elif config.dataset == 'DomainNet':
        for i in range(len(config.domains)-1):
            for j in range(i + 1, len(config.domains)):
                trainers.append(f"{config.domains[i]}_{config.domains[j]}")
        return trainers
    else:
        raise ValueError(
            f"This script is configured to train model on cifar10c, OfficeHome and DomainNet datasets "
            f"Please choose model as one of the above "
            f"Or configure this function (train) in steps.py"
        )


def get_calib_domains(config: ml_collections.ConfigDict, trainer: str):
    calib_domains = dict()
    if config.dataset == 'cifar10c':
        sever = trainer.split('_')[1]
        calib_domains[trainer] = get_filters_with_level(config.domains.calib_corruptions, sever)
        return calib_domains
    elif config.dataset == 'OfficeHome':
        domains = config.domains.copy()
        domains.remove(trainer)
        for domain in domains:
            two_domains = domains.copy()
            two_domains.remove(domain)
            calib_domains['_'.join(two_domains)] = two_domains.copy()
        return calib_domains
    elif config.dataset == 'DomainNet':
        domains = config.domains.copy()
        train_domains = trainer.split('_')
        for d in train_domains:
            domains.remove(d)
        for domain in domains:
            three_domains = domains.copy()
            three_domains.remove(domain)
            calib_domains[domain] = three_domains.copy()
        return calib_domains


def get_data(dataset_dir, domains: list, folder: str):
    domains_data = dict()
    for domain in domains:
        domains_data[domain] = dict()
        domains_data[domain]['data'] = np.load(os.path.join(dataset_dir, folder, f"{domain}_X.npy"))
        domains_data[domain]['label'] = np.load(os.path.join(dataset_dir, folder, f"{domain}_y.npy"))
    return domains_data


def unpack_domains(domains_data, ts=True):
    data = []
    labels = []
    logits = []
    representation = []
    y = []

    for domain in domains_data.keys():
        data.append(domains_data[domain]['data'])
        labels.append(domains_data[domain]['label'])
        if 'logit' in domains_data[domain].keys():
            logits.append(domains_data[domain]['logit'])
            representation.append(domains_data[domain]['representation'])

            if ts:
                ts_model = TemperatureScaling()
                ts_model.fit(domains_data[domain])

                probs = softmax(domains_data[domain]['logit'] / ts_model.temp)
                y_cat = keras.utils.to_categorical(domains_data[domain]['label'], domains_data[domain]['logit'].shape[1])
                y.append(np.concatenate((probs, y_cat), axis=1))

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    if len(logits) > 0:
        logits = np.concatenate(logits)
        representation = np.concatenate(representation)
        if ts:
            y = np.concatenate(y)

    return {'data': data, 'label': labels, 'logit': logits, 'representation': representation, 'y': y}
