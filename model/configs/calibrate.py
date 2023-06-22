from absl import logging
import ml_collections
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import log_loss, brier_score_loss
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

from model.model_utils import (calibrationError, brier_multi, softmax, get_importance_weight)
from model.resnet101 import ResNet_101
from model import input_pipeline
from data_utils import get_data, get_filters_with_level, get_train_domains, get_calib_domains, unpack_domains
from calibration.cal_methods import (TemperatureScaling, HistogramBinning, IsotonicCalibration, BetaCalib,
                                     ClusterKMeans, CaliGenCalibration, SCaliGenCalibration, CaliGenKFold,
                                     CPCS, TransCal, HBTopLabel)

RHOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def calibrate_cifar10c(config: ml_collections.ConfigDict, workdir: str):
    for sever in config.severity:
        resnet = ResNet_101(config, last_layer_activation='linear')
        weight_file = os.path.join(config.weights_path, config.dataset, config.loss, f"4filter_{sever}",
                                   "model_weights.h5")

        logits_path = os.path.join(config.logits_path, config.dataset, config.loss, f"4filter_{sever}")

        for calib in ['in', 'out']:
            train_with_level = []
            for fil in config.domains.train_corruptions:
                if fil != 'original':
                    if sever == 'a':
                        for i in range(1, 6):
                            train_with_level.append(f"{fil}_{i}")
                    else:
                        train_with_level.append(f"{fil}_{sever}")
            train_with_level = ['original'] + train_with_level
            calib_with_level = []
            for fil in config.domains.calib_corruptions:
                if sever == 'a':
                    for i in range(1, 6):
                        calib_with_level.append(f"{fil}_{i}")
                else:
                    calib_with_level.append(f"{fil}_{sever}")

            if calib == 'in':
                filters = train_with_level + calib_with_level
            else:
                filters = calib_with_level

            calib_data = dict()
            if config.from_logits.lower() == 'yes':
                for d in filters:
                    logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                    with open(logit_file, 'rb') as f:
                        calib_data[d] = pickle.load(f)
            else:
                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )

                resnet.load_weights(weight_file)  # .expect_partial()
                for d in filters:
                    ds = input_pipeline.get_cifar10c_eval_data(config.dataset_dir, workdir, 'valid', d)
                    [logits, representation] = resnet.predict(ds['data'])
                    calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

            ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, f"4filter_{sever}",
                                    'cluster')
            Path(ckm_path).mkdir(parents=True, exist_ok=True)
            ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}

            ckm = ClusterKMeans(**ckm_kwargs)
            ckm.fit(calib_data)

            caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, f"4filter_{sever}",
                                        'caligen')
            Path(caligen_path).mkdir(parents=True, exist_ok=True)

            for rho in RHOS:
                caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                  'loss': config.loss, 'trainer': config.domains.train_corruptions}

                caligen = CaliGenCalibration(**caligen_kwargs)
                caligen.fit(calib_data)


def calibrate_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    for trainer in config.domains:
        resnet = ResNet_101(config, last_layer_activation='linear')
        weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                   "model_weights.h5")

        logits_path = os.path.join(config.logits_path, config.dataset, config.loss, trainer)

        three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        three_domains.remove(trainer)

        for cd in three_domains:
            two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            two_domains.remove(trainer)
            two_domains.remove(cd)

            logging.info(
                f"Trained on: {trainer} "
                f"Calibration domains: {two_domains}"
            )

            for calib in ['in', 'out']:

                if calib == 'in':
                    calib_domains = [trainer] + two_domains
                else:
                    calib_domains = two_domains

                calib_data = dict()
                if config.from_logits.lower() == 'yes':
                    for d in calib_domains:
                        logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                        with open(logit_file, 'rb') as f:
                            calib_data[d] = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    for d in calib_domains:
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                        [logits, representation] = resnet.predict(ds['data'])
                        calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                        "_".join(two_domains), 'cluster')
                Path(ckm_path).mkdir(parents=True, exist_ok=True)
                ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}

                ckm = ClusterKMeans(**ckm_kwargs)
                ckm.fit(calib_data)

                caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                        "_".join(two_domains), 'caligen')
                Path(caligen_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                      'loss': config.loss, 'trainer': [trainer]}

                    caligen = CaliGenCalibration(**caligen_kwargs)
                    caligen.fit(calib_data)

def calibrate_ablation_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    for trainer in config.domains:
        resnet = ResNet_101(config, last_layer_activation='linear')
        weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                   "model_weights.h5")

        logits_path = os.path.join(config.logits_path, config.dataset, config.loss, trainer)

        three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        three_domains.remove(trainer)

        for cd in three_domains:
            two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            two_domains.remove(trainer)
            two_domains.remove(cd)

            logging.info(
                f"Trained on: {trainer} "
                f"Calibration domains: {two_domains}"
            )

            for calib in ['in', 'out']:

                if calib == 'in':
                    calib_domains = [trainer] + two_domains
                else:
                    calib_domains = two_domains

                calib_data = dict()
                if config.from_logits.lower() == 'yes':
                    for d in calib_domains:
                        logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                        with open(logit_file, 'rb') as f:
                            calib_data[d] = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    for d in calib_domains:
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                        [logits, representation] = resnet.predict(ds['data'])
                        calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                caligen_path = os.path.join(config.calib_path, config.dataset, 'ablation', calib, trainer,
                                        "_".join(two_domains))
                Path(caligen_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': True,
                                      'loss': config.loss, 'trainer': [trainer]}

                    caligen = CaliGenCalibration(**caligen_kwargs)
                    caligen.fit(calib_data)


def calibrate_DomainNet(config: ml_collections.ConfigDict, workdir: str):
    for i in range(len(config.domains)):
        for j in range(i+1, len(config.domains)):
            trainer = [config.domains[i], config.domains[j]]
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, '_'.join(trainer),
                                       "model_weights.h5")

            logits_path = os.path.join(config.logits_path, config.dataset, config.loss, '_'.join(trainer))

            four_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            for t in trainer:
                four_domains.remove(t)

            for cd in four_domains:
                three_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                for t in trainer:
                    three_domains.remove(t)
                three_domains.remove(cd)

                logging.info(
                    f"Trained on: {trainer} "
                    f"Calibration domains: {three_domains}"
                )

                for calib in ['in', 'out']:

                    if calib == 'in':
                        calib_domains = trainer + three_domains
                    else:
                        calib_domains = three_domains

                    calib_data = dict()

                    if config.from_logits.lower() == 'yes':
                        logits_path = os.path.join(config.logits_path, config.dataset, config.loss, '_'.join(trainer))
                        for d in calib_domains:
                            logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                            with open(logit_file, 'rb') as f:
                                calib_data[d] = pickle.load(f)
                    else:
                        if not os.path.exists(weight_file):
                            raise ValueError(
                                f"weights does not exists at {weight_file} "
                                f"Please specify correct place for weights or train the model first."
                            )

                        resnet.load_weights(weight_file)
                        for d in calib_domains:
                            ds = input_pipeline.get_DomainNet_data(config, workdir, [d], 'valid')
                            [logits, representation] = resnet.predict(ds['data'])
                            calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                    ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, '_'.join(trainer),
                                            cd, 'cluster')
                    Path(ckm_path).mkdir(parents=True, exist_ok=True)
                    ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}

                    ckm = ClusterKMeans(**ckm_kwargs)
                    ckm.fit(calib_data)

                    caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, '_'.join(trainer),
                                            cd, 'caligen')
                    Path(caligen_path).mkdir(parents=True, exist_ok=True)

                    for rho in RHOS:
                        caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                          'loss': config.loss, 'trainer': trainer}

                        caligen = CaliGenCalibration(**caligen_kwargs)
                        caligen.fit(calib_data)


def evaluate(probs, y_true, verbose=False, normalize=False, bins=15):
    '''
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    :param probs: a list containing probabilities for all the classes with a shape of (samples, classes)
    :param y_true: a list containing the actual class labels
    :param verbose: (bool) are the scores printed out. (default = False)
    :param normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
    :param bins: (int) into how many bins are probabilities divided (default = 15)
    :return: (error, ece, mce, loss, brier), returns various scoring measures
    '''
    num_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

    if normalize:
        confs = np.max(probs, axis=1) / np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # calculate calibrations errors

    cal_error = calibrationError(confs, preds, y_true, bin_size=1/bins)
    ece, mce, _ = cal_error.calculate_errors()

    loss = log_loss(y_true=y_true, y_pred=probs, labels=np.arange(num_classes))

    #y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    #brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    brier = brier_multi(y_true=keras.utils.to_categorical(y_true, num_classes), y_prob=probs)  # Brier Score (MSE)

    #brier = 0
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)

    return {'error': error, 'ece': ece, 'mce': mce, 'loss': loss, 'brier': brier}

def evaluate_cifar10c(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        for sever in config.severity:
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, f"4filter_{sever}",
                                       "model_weights.h5")

            cal_models = dict()
            clmns = ['Trainer', 'Train', 'Valid', 'Domain']
            methods = [ 'Uncalibrated', 'TS Source', 'TS Oracle', 'HB', 'Isotonic', 'Beta abm', 'Beta am',
                        'Beta ab', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem']
            clmns += methods

            for rho in RHOS:
                clmns += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
                methods += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
            df_ece = pd.DataFrame(columns=clmns)
            df_error = pd.DataFrame(columns=clmns)
            df_mce = pd.DataFrame(columns=clmns)
            df_loss = pd.DataFrame(columns=clmns)
            df_brier = pd.DataFrame(columns=clmns)

            df_ece_std = pd.DataFrame(columns=clmns)
            df_error_std = pd.DataFrame(columns=clmns)
            df_mce_std = pd.DataFrame(columns=clmns)
            df_loss_std = pd.DataFrame(columns=clmns)
            df_brier_std = pd.DataFrame(columns=clmns)

            train_with_level = []
            for fil in config.domains.train_corruptions:
                if fil != 'original':
                    if sever == 'a':
                        for i in range(1, 6):
                            train_with_level.append(f"{fil}_{i}")
                    else:
                        train_with_level.append(f"{fil}_{sever}")
            train_with_level = ['original'] + train_with_level
            calib_with_level = []
            for fil in config.domains.calib_corruptions:
                if sever == 'a':
                    for i in range(1, 6):
                        calib_with_level.append(f"{fil}_{i}")
                else:
                    calib_with_level.append(f"{fil}_{sever}")

            if calib == 'in':
                filters = train_with_level + calib_with_level
            else:
                filters = calib_with_level

            calib_data = dict()
            source_data = dict()
            if config.from_logits.lower() == 'yes':
                logits_path = os.path.join(config.logits_path, config.dataset, config.loss, f"4filter_{sever}")
                for d in filters:
                    logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                    with open(logit_file, 'rb') as f:
                        calib_data[d] = pickle.load(f)
                for t in train_with_level:
                    logit_file = os.path.join(logits_path, f"valid_logits_{t}.p")
                    with open(logit_file, 'rb') as f:
                        source_data[t] = pickle.load(f)

            else:
                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )

                resnet.load_weights(weight_file)  # .expect_partial()
                for d in filters:
                    ds = input_pipeline.get_cifar10c_eval_data(config.dataset_dir, workdir, 'valid', d)
                    [logits, representation] = resnet.predict(ds['data'])
                    calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}
                for t in train_with_level:
                    ds = input_pipeline.get_cifar10c_eval_data(config.dataset_dir, workdir, 'valid', t)
                    [logits, representation] = resnet.predict(ds['data'])
                    source_data[t] = {'logit': logits, 'label': ds['label'], 'representation': representation}

            path = os.path.join(config.calib_path, config.dataset, config.loss, calib, f"4filter_{sever}")

            cal_models['TS Source'] = TemperatureScaling()
            cal_models['TS Source'].fit(source_data)

            cal_models['HB'] = HistogramBinning(**{'params': {'BINS': 15}})
            cal_models['HB'].fit(calib_data)

            cal_models['Isotonic'] = IsotonicCalibration(**{'params': {'y_min': 0, 'y_max': 1}})
            cal_models['Isotonic'].fit(calib_data)

            cal_models['Beta abm'] = BetaCalib(**{'params': {'parameters': "abm"}})
            cal_models['Beta abm'].fit(calib_data)

            cal_models['Beta am'] = BetaCalib(**{'params': {'parameters': "am"}})
            cal_models['Beta am'].fit(calib_data)

            cal_models['Beta ab'] = BetaCalib(**{'params': {'parameters': "ab"}})
            cal_models['Beta ab'].fit(calib_data)

            ckm_path = os.path.join(path, 'cluster')
            Path(ckm_path).mkdir(parents=True, exist_ok=True)
            ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}
            cal_models['Cluster'] = ClusterKMeans(**ckm_kwargs)
            cal_models['Cluster'].fit(calib_data)

            caligen_path = os.path.join(path, 'caligen')
            Path(caligen_path).mkdir(parents=True, exist_ok=True)

            for rho in RHOS:
                caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                  'loss': config.loss, 'trainer': train_with_level}

                cal_models[f'CaliGen {rho}'] = CaliGenCalibration(**caligen_kwargs)
                cal_models[f'CaliGen {rho}'].fit(calib_data)

            domains = config.domains.train_corruptions + config.domains.calib_corruptions +\
                      config.domains.test_corruptions

            filters_with_level = []
            for fil in domains:
                if fil != 'original':
                    for i in range(1, 6):
                            filters_with_level.append(f"{fil}_{i}")
            eval_domains = ['original'] + filters_with_level

            for eval_domain in eval_domains:
                if config.from_logits.lower() == 'yes':
                    logits_path = os.path.join(config.logits_path, config.dataset, config.loss, f"4filter_{sever}")
                    logit_file = os.path.join(logits_path, f"test_logits_{eval_domain}.p")
                    with open(logit_file, 'rb') as f:
                        test_data = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    ds = input_pipeline.get_cifar10c_eval_data(config.dataset_dir, workdir, 'test', eval_domain)
                    [logits, representation] = resnet.predict(ds['data'])
                    test_data = {'logit': logits, 'label': ds['label'], 'representation': representation}

                cal_models['TS Oracle'] = TemperatureScaling()
                cal_models['TS Oracle'].fit(test_data)

                ece_row = {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                error_row =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                mce_row =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                loss_row =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                brier_row = {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}

                ece_row_std =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                error_row_std =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                mce_row_std =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                loss_row_std =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}
                brier_row_std =  {'Trainer': f"4filter_{sever}", 'Train': eval_domain in train_with_level,
                           'Valid': eval_domain in calib_with_level, 'Domain': eval_domain}

                row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                             'loss': loss_row, 'brier': brier_row}
                row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                 'loss': loss_row_std, 'brier': brier_row_std}

                test_results = dict()
                for model in methods:
                    test_results[model] = dict()
                    for metric in row_dicts.keys():
                        test_results[model][metric] = []

                for seed in range(20):
                    _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                          test_data['representation'], test_size=500,
                                                                          random_state=seed)

                    ds = {'logit': logits, 'label': y, 'representation': representation}

                    uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                    for key in row_dicts.keys():
                        test_results['Uncalibrated'][key].append(uncal_res[key])

                    for model in cal_models.keys():
                        if model == 'Cluster':
                            for mode in ['TS', 'NN', 'LR', 'Ensem']:
                                cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                m = mode if mode == 'TS' else f'{model} {mode}'
                                for key in row_dicts.keys():
                                    test_results[m][key].append(cal_res[key])

                        elif 'CaliGen' in model:
                            for mode in ['CaliGen', 'TS', 'Ensem']:
                                cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                if mode == 'CaliGen':
                                    m = model
                                else:
                                    c = model.split(' ')
                                    m = f'{c[0]} {mode} {c[1]}'
                                for key in row_dicts.keys():
                                    test_results[m][key].append(cal_res[key])

                        else:
                            cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                            for key in row_dicts.keys():
                                test_results[model][key].append(cal_res[key])

                for model in methods:
                    for key in row_dicts.keys():
                        row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                        row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                df_ece = df_ece.append(ece_row, ignore_index=True)
                df_error = df_error.append(error_row, ignore_index=True)
                df_mce = df_mce.append(mce_row, ignore_index=True)
                df_loss = df_loss.append(loss_row, ignore_index=True)
                df_brier = df_brier.append(brier_row, ignore_index=True)

                df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)

            path_to_res = os.path.join(config.res_path, config.dataset, config.loss, f"4filter_{sever}")
            Path(path_to_res).mkdir(parents=True, exist_ok=True)

            df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean.csv"), index=False)
            df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean.csv"), index=False)
            df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean.csv"), index=False)
            df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean.csv"), index=False)
            df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean.csv"), index=False)

            df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std.csv"), index=False)
            df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std.csv"), index=False)
            df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std.csv"), index=False)
            df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std.csv"), index=False)
            df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std.csv"), index=False)

def evaluate_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        cal_models = dict()
        clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
        methods = ['Uncalibrated', 'TS Source', 'TS Oracle', 'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab',
                   'TS', 'Cluster NN', 'Cluster LR','Cluster Ensem']
        clmns += methods

        for rho in RHOS:
            clmns += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
            methods += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
        df_ece = pd.DataFrame(columns=clmns)
        df_error = pd.DataFrame(columns=clmns)
        df_mce = pd.DataFrame(columns=clmns)
        df_loss = pd.DataFrame(columns=clmns)
        df_brier = pd.DataFrame(columns=clmns)

        df_ece_std = pd.DataFrame(columns=clmns)
        df_error_std = pd.DataFrame(columns=clmns)
        df_mce_std = pd.DataFrame(columns=clmns)
        df_loss_std = pd.DataFrame(columns=clmns)
        df_brier_std = pd.DataFrame(columns=clmns)

        for trainer in config.domains:
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                       "model_weights.h5")
            logits_path = os.path.join(config.logits_path, config.dataset, config.loss, trainer)

            three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            three_domains.remove(trainer)

            for cd in three_domains:
                two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
                two_domains.remove(trainer)
                two_domains.remove(cd)

                logging.info(
                    f"Trained on: {trainer} "
                    f"Calibration domains: {two_domains}"
                )

                if calib == 'in':
                    calib_domains = [trainer] + two_domains
                else:
                    calib_domains = two_domains

                calib_data = dict()
                if config.from_logits.lower() == 'yes':
                    for d in calib_domains:
                        logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                        with open(logit_file, 'rb') as f:
                            calib_data[d] = pickle.load(f)
                    logit_file = os.path.join(logits_path, f"valid_logits_{trainer}.p")
                    with open(logit_file, 'rb') as f:
                        source_data = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    for d in calib_domains:
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                        [logits, representation] = resnet.predict(ds['data'])
                        calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}
                    ds = input_pipeline.get_OfficeHome_data(config, workdir, trainer, 'valid')
                    [logits, representation] = resnet.predict(ds['data'])
                    source_data = {'logit': logits, 'label': ds['label'], 'representation': representation}


                path = os.path.join(config.calib_path, config.dataset, config.loss, calib,
                                    trainer, "_".join(two_domains))

                cal_models['TS Source'] = TemperatureScaling()
                cal_models['TS Source'].fit(source_data)

                cal_models['HB'] = HistogramBinning(**{'params': {'BINS': 15}})
                cal_models['HB'].fit(calib_data)

                cal_models['Isotonic'] = IsotonicCalibration(**{'params': {'y_min': 0, 'y_max': 1}})
                cal_models['Isotonic'].fit(calib_data)

                cal_models['Beta abm'] = BetaCalib(**{'params': {'parameters': "abm"}})
                cal_models['Beta abm'].fit(calib_data)

                cal_models['Beta am'] = BetaCalib(**{'params': {'parameters': "am"}})
                cal_models['Beta am'].fit(calib_data)

                cal_models['Beta ab'] = BetaCalib(**{'params': {'parameters': "ab"}})
                cal_models['Beta ab'].fit(calib_data)


                ckm_path = os.path.join(path, 'cluster')
                Path(ckm_path).mkdir(parents=True, exist_ok=True)
                ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}
                cal_models['Cluster'] = ClusterKMeans(**ckm_kwargs)
                cal_models['Cluster'].fit(calib_data)

                caligen_path = os.path.join(path, 'caligen')
                Path(caligen_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                      'loss': config.loss, 'trainer': [trainer]}

                    cal_models[f'CaliGen {rho}'] = CaliGenCalibration(**caligen_kwargs)
                    cal_models[f'CaliGen {rho}'].fit(calib_data)

                for eval_domain in config.domains:
                    if config.from_logits.lower() == 'yes':
                        logit_file = os.path.join(logits_path, f"train_logits_{eval_domain}.p")
                        with open(logit_file, 'rb') as f:
                            test_data = pickle.load(f)
                    else:
                        if not os.path.exists(weight_file):
                            raise ValueError(
                                f"weights does not exists at {weight_file} "
                                f"Please specify correct place for weights or train the model first."
                            )

                        resnet.load_weights(weight_file)  # .expect_partial()
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, eval_domain, 'train')
                        [logits, representation] = resnet.predict(ds['data'])
                        test_data = {'logit': logits, 'label': ds['label'], 'representation': representation}

                    cal_models['TS Oracle'] = TemperatureScaling()
                    cal_models['TS Oracle'].fit(test_data)

                    ece_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    error_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    mce_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    loss_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    brier_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                    ece_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                   'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    error_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                     'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    mce_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                   'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    loss_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                    'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    brier_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                     'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                    row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                                 'loss': loss_row, 'brier': brier_row}
                    row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                     'loss': loss_row_std, 'brier': brier_row_std}


                    test_results = dict()
                    for model in methods:
                        test_results[model] = dict()
                        for metric in row_dicts.keys():
                            test_results[model][metric] = []

                    for seed in range(20):
                        _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                              test_data['representation'], test_size=500,
                                                                              random_state=seed)

                        ds = {'logit': logits, 'label': y, 'representation': representation}

                        uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                        for key in row_dicts.keys():
                            test_results['Uncalibrated'][key].append(uncal_res[key])
                            #row_dicts[key]['Uncalibrated'] = uncal_res[key]

                        for model in cal_models.keys():
                            if model == 'Cluster':
                                for mode in ['TS', 'NN', 'LR', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    m = mode if mode == 'TS' else f'{model} {mode}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            elif 'CaliGen' in model:
                                for mode in ['CaliGen', 'TS', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    if mode == 'CaliGen':
                                        m = model
                                    else:
                                        c = model.split(' ')
                                        m = f'{c[0]} {mode} {c[1]}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            else:
                                cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                                for key in row_dicts.keys():
                                    test_results[model][key].append(cal_res[key])

                    for model in methods:
                        for key in row_dicts.keys():
                            row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                            row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                    df_ece = df_ece.append(ece_row, ignore_index=True)
                    df_error = df_error.append(error_row, ignore_index=True)
                    df_mce = df_mce.append(mce_row, ignore_index=True)
                    df_loss = df_loss.append(loss_row, ignore_index=True)
                    df_brier = df_brier.append(brier_row, ignore_index=True)

                    df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                    df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                    df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                    df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                    df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)


        path_to_res = os.path.join(config.res_path, config.dataset, config.loss)
        Path(path_to_res).mkdir(parents=True, exist_ok=True)

        df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std.csv"), index=False)

def evaluate_ablation_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['out']:
        cal_models = dict()
        clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
        methods = ['Uncalibrated']
        clmns += methods

        for rho in RHOS:
            clmns += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
            methods += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
        df_ece = pd.DataFrame(columns=clmns)
        df_error = pd.DataFrame(columns=clmns)
        df_mce = pd.DataFrame(columns=clmns)
        df_loss = pd.DataFrame(columns=clmns)
        df_brier = pd.DataFrame(columns=clmns)

        df_ece_std = pd.DataFrame(columns=clmns)
        df_error_std = pd.DataFrame(columns=clmns)
        df_mce_std = pd.DataFrame(columns=clmns)
        df_loss_std = pd.DataFrame(columns=clmns)
        df_brier_std = pd.DataFrame(columns=clmns)

        for trainer in config.domains:
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, '_'.join(trainer),
                                       "model_weights.h5")
            logits_path = os.path.join(config.logits_path, config.dataset, config.loss, trainer)

            three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            three_domains.remove(trainer)

            for cd in three_domains:
                two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
                two_domains.remove(trainer)
                two_domains.remove(cd)

                logging.info(
                    f"Trained on: {trainer} "
                    f"Calibration domains: {two_domains}"
                )

                if calib == 'in':
                    calib_domains = [trainer] + two_domains
                else:
                    calib_domains = two_domains

                calib_data = dict()

                if config.from_logits.lower() == 'yes':
                    for d in calib_domains:
                        logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                        with open(logit_file, 'rb') as f:
                            calib_data[d] = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    for d in calib_domains:
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                        [logits, representation] = resnet.predict(ds['data'])
                        calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                path = os.path.join(config.calib_path, config.dataset, 'ablation', calib,
                                    trainer, "_".join(two_domains))

                caligen_path = path
                Path(caligen_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': True,
                                      'loss': config.loss, 'trainer': [trainer]}

                    cal_models[f'CaliGen {rho}'] = CaliGenCalibration(**caligen_kwargs)
                    cal_models[f'CaliGen {rho}'].fit(calib_data)

                for eval_domain in config.domains:
                    if config.from_logits.lower() == 'yes':
                        logit_file = os.path.join(logits_path, f"train_logits_{eval_domain}.p")
                        with open(logit_file, 'rb') as f:
                            test_data = pickle.load(f)
                    else:
                        if not os.path.exists(weight_file):
                            raise ValueError(
                                f"weights does not exists at {weight_file} "
                                f"Please specify correct place for weights or train the model first."
                            )

                        resnet.load_weights(weight_file)  # .expect_partial()
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, eval_domain, 'train')
                        [logits, representation] = resnet.predict(ds['data'])
                        test_data = {'logit': logits, 'label': ds['label'], 'representation': representation}

                    ece_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    error_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    mce_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    loss_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    brier_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                    ece_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                   'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    error_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                     'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    mce_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                   'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    loss_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                    'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                    brier_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                     'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                    row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                                 'loss': loss_row, 'brier': brier_row}
                    row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                     'loss': loss_row_std, 'brier': brier_row_std}


                    test_results = dict()
                    for model in methods:
                        test_results[model] = dict()
                        for metric in row_dicts.keys():
                            test_results[model][metric] = []

                    for seed in range(20):
                        _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                              test_data['representation'], test_size=500,
                                                                              random_state=seed)

                        ds = {'logit': logits, 'label': y, 'representation': representation}

                        uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                        for key in row_dicts.keys():
                            test_results['Uncalibrated'][key].append(uncal_res[key])
                            #row_dicts[key]['Uncalibrated'] = uncal_res[key]

                        for model in cal_models.keys():
                            if 'CaliGen' in model:
                                for mode in ['CaliGen', 'TS', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    if mode == 'CaliGen':
                                        m = model
                                    else:
                                        c = model.split(' ')
                                        m = f'{c[0]} {mode} {c[1]}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            else:
                                cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                                for key in row_dicts.keys():
                                    test_results[model][key].append(cal_res[key])

                    for model in methods:
                        for key in row_dicts.keys():
                            row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                            row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                    df_ece = df_ece.append(ece_row, ignore_index=True)
                    df_error = df_error.append(error_row, ignore_index=True)
                    df_mce = df_mce.append(mce_row, ignore_index=True)
                    df_loss = df_loss.append(loss_row, ignore_index=True)
                    df_brier = df_brier.append(brier_row, ignore_index=True)

                    df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                    df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                    df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                    df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                    df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)


        path_to_res = os.path.join(config.res_path, config.dataset, 'ablation')
        Path(path_to_res).mkdir(parents=True, exist_ok=True)

        df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std.csv"), index=False)

def evaluate_limitation_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    resnet = ResNet_101(config, last_layer_activation='linear')
    cal_models = dict()
    clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
    methods = ['Uncalibrated', 'TS']
    clmns += methods

    df_ece = pd.DataFrame(columns=clmns)
    df_error = pd.DataFrame(columns=clmns)
    df_mce = pd.DataFrame(columns=clmns)
    df_loss = pd.DataFrame(columns=clmns)
    df_brier = pd.DataFrame(columns=clmns)

    df_ece_std = pd.DataFrame(columns=clmns)
    df_error_std = pd.DataFrame(columns=clmns)
    df_mce_std = pd.DataFrame(columns=clmns)
    df_loss_std = pd.DataFrame(columns=clmns)
    df_brier_std = pd.DataFrame(columns=clmns)

    for trainer in config.domains:
        three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        three_domains.remove(trainer)

        for cd in three_domains:
            two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            two_domains.remove(trainer)
            two_domains.remove(cd)

            logging.info(
                f"Trained on: {trainer} "
                f"Calibration domains: {two_domains}"
            )

            filepath = os.path.join(config.calib_path, config.dataset, 'limitation', trainer, '_'.join(two_domains))
            weight_file = os.path.join(filepath, "model_weights.h5")
            resnet.load_weights(weight_file)

            calib_data = dict()
            ds = input_pipeline.get_OfficeHome_data(config, workdir, trainer, 'valid')
            [logits, representation] = resnet.predict(ds['data'])

            calib_data[trainer] = {'logit': logits, 'label': ds['label'], 'representation': representation}

            for d in two_domains:
                ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                _, x_v, _, y_v = train_test_split(ds['data'], ds['label'], test_size=0.5,
                                                      random_state=6)
                [logits, representation] = resnet.predict(x_v)
                calib_data[d] = {'logit': logits, 'label': y_v, 'representation': representation}

            cal_models['TS'] = TemperatureScaling()
            cal_models['TS'].fit(calib_data)


            for eval_domain in config.domains:
                ds = input_pipeline.get_OfficeHome_data(config, workdir, eval_domain, 'train')
                [logits, representation] = resnet.predict(ds['data'])

                test_data = {'logit': logits, 'label': ds['label'], 'representation': representation}

                ece_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                           'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                error_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                           'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                mce_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                           'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                loss_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                           'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                brier_row = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                           'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                ece_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                error_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                 'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                mce_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                               'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                loss_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                'Valid': eval_domain in two_domains, 'Domain': eval_domain}
                brier_row_std = {'Trainer': trainer, 'Calib': '_'.join(two_domains), 'Train': eval_domain == trainer,
                                 'Valid': eval_domain in two_domains, 'Domain': eval_domain}

                row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                             'loss': loss_row, 'brier': brier_row}
                row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                 'loss': loss_row_std, 'brier': brier_row_std}


                test_results = dict()
                for model in methods:
                    test_results[model] = dict()
                    for metric in row_dicts.keys():
                        test_results[model][metric] = []

                for seed in range(20):
                    _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                          test_data['representation'], test_size=500,
                                                                          random_state=seed)

                    ds = {'logit': logits, 'label': y, 'representation': representation}

                    uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                    for key in row_dicts.keys():
                        test_results['Uncalibrated'][key].append(uncal_res[key])
                        #row_dicts[key]['Uncalibrated'] = uncal_res[key]

                    cal_res = evaluate(cal_models['TS'].predict(ds), ds['label'])
                    for key in row_dicts.keys():
                        test_results['TS'][key].append(cal_res[key])

                for model in methods:
                    for key in row_dicts.keys():
                        row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                        row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                df_ece = df_ece.append(ece_row, ignore_index=True)
                df_error = df_error.append(error_row, ignore_index=True)
                df_mce = df_mce.append(mce_row, ignore_index=True)
                df_loss = df_loss.append(loss_row, ignore_index=True)
                df_brier = df_brier.append(brier_row, ignore_index=True)

                df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)


    path_to_res = os.path.join(config.res_path, config.dataset, 'limitation')
    Path(path_to_res).mkdir(parents=True, exist_ok=True)

    df_error.to_csv(os.path.join(path_to_res, "Error_mean.csv"), index=False)
    df_ece.to_csv(os.path.join(path_to_res, "ECE_mean.csv"), index=False)
    df_mce.to_csv(os.path.join(path_to_res, "MCE_mean.csv"), index=False)
    df_loss.to_csv(os.path.join(path_to_res, "Loss_mean.csv"), index=False)
    df_brier.to_csv(os.path.join(path_to_res, "Brier_mean.csv"), index=False)

    df_error_std.to_csv(os.path.join(path_to_res, "Error_std.csv"), index=False)
    df_ece_std.to_csv(os.path.join(path_to_res, "ECE_std.csv"), index=False)
    df_mce_std.to_csv(os.path.join(path_to_res, "MCE_std.csv"), index=False)
    df_loss_std.to_csv(os.path.join(path_to_res, "Loss_std.csv"), index=False)
    df_brier_std.to_csv(os.path.join(path_to_res, "Brier_std.csv"), index=False)


def evaluate_DomainNet(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        cal_models = dict()
        clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
        methods = ['Uncalibrated', 'TS Source', 'TS Oracle', 'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab',
                   'TS', 'Cluster NN', 'Cluster LR','Cluster Ensem', ]
        clmns += methods

        for rho in RHOS:
            clmns += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
            methods += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
        df_ece = pd.DataFrame(columns=clmns)
        df_error = pd.DataFrame(columns=clmns)
        df_mce = pd.DataFrame(columns=clmns)
        df_loss = pd.DataFrame(columns=clmns)
        df_brier = pd.DataFrame(columns=clmns)

        df_ece_std = pd.DataFrame(columns=clmns)
        df_error_std = pd.DataFrame(columns=clmns)
        df_mce_std = pd.DataFrame(columns=clmns)
        df_loss_std = pd.DataFrame(columns=clmns)
        df_brier_std = pd.DataFrame(columns=clmns)

        for i in range(len(config.domains)):
            for j in range(i + 1, len(config.domains)):
                trainer = [config.domains[i], config.domains[j]]
                resnet = ResNet_101(config, last_layer_activation='linear')
                weight_file = os.path.join(config.weights_path, config.dataset, config.loss, '_'.join(trainer),
                                           "model_weights.h5")
                logits_path = os.path.join(config.logits_path, config.dataset, config.loss, '_'.join(trainer))

                four_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                for t in trainer:
                    four_domains.remove(t)

                for cd in four_domains:
                    three_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                    for t in trainer:
                        three_domains.remove(t)
                    three_domains.remove(cd)

                    logging.info(
                        f"Trained on: {trainer} "
                        f"Calibration domains: {three_domains}"
                    )


                    if calib == 'in':
                        calib_domains = trainer + three_domains
                    else:
                        calib_domains = three_domains

                    calib_data = dict()
                    source_data = dict()

                    if config.from_logits.lower() == 'yes':
                        for d in calib_domains:
                            logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                            with open(logit_file, 'rb') as f:
                                calib_data[d] = pickle.load(f)
                        for t in trainer:
                            logit_file = os.path.join(logits_path, f"valid_logits_{t}.p")
                            with open(logit_file, 'rb') as f:
                                source_data[t] = pickle.load(f)
                    else:
                        if not os.path.exists(weight_file):
                            raise ValueError(
                                f"weights does not exists at {weight_file} "
                                f"Please specify correct place for weights or train the model first."
                            )

                        resnet.load_weights(weight_file)
                        for d in calib_domains:
                            ds = input_pipeline.get_DomainNet_data(config, workdir, [d], 'valid')
                            [logits, representation] = resnet.predict(ds['data'])
                            calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}
                        for t in trainer:
                            ds = input_pipeline.get_DomainNet_data(config, workdir, [t], 'valid')
                            [logits, representation] = resnet.predict(ds['data'])
                            source_data[t] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                    path = os.path.join(config.calib_path, config.dataset, config.loss, calib,
                                        '_'.join(trainer), cd)

                    cal_models['TS Source'] = TemperatureScaling()
                    cal_models['TS Source'].fit(source_data)

                    cal_models['HB'] = HistogramBinning(**{'params': {'BINS': 15}})
                    cal_models['HB'].fit(calib_data)

                    cal_models['Isotonic'] = IsotonicCalibration(**{'params': {'y_min': 0, 'y_max': 1}})
                    cal_models['Isotonic'].fit(calib_data)

                    cal_models['Beta abm'] = BetaCalib(**{'params': {'parameters': "abm"}})
                    cal_models['Beta abm'].fit(calib_data)

                    cal_models['Beta am'] = BetaCalib(**{'params': {'parameters': "am"}})
                    cal_models['Beta am'].fit(calib_data)

                    cal_models['Beta ab'] = BetaCalib(**{'params': {'parameters': "ab"}})
                    cal_models['Beta ab'].fit(calib_data)


                    ckm_path = os.path.join(path, 'cluster')
                    Path(ckm_path).mkdir(parents=True, exist_ok=True)
                    ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}
                    cal_models['Cluster'] = ClusterKMeans(**ckm_kwargs)
                    cal_models['Cluster'].fit(calib_data)

                    caligen_path = os.path.join(path, 'caligen')
                    Path(caligen_path).mkdir(parents=True, exist_ok=True)

                    for rho in RHOS:
                        caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                          'loss': config.loss, 'trainer': [trainer]}

                        cal_models[f'CaliGen {rho}'] = CaliGenCalibration(**caligen_kwargs)
                        cal_models[f'CaliGen {rho}'].fit(calib_data)

                    for eval_domain in config.domains:
                        if config.from_logits.lower() == 'yes':
                            logit_file = os.path.join(logits_path, f"train_logits_{eval_domain}.p")
                            with open(logit_file, 'rb') as f:
                                test_data = pickle.load(f)
                        else:
                            if not os.path.exists(weight_file):
                                raise ValueError(
                                    f"weights does not exists at {weight_file} "
                                    f"Please specify correct place for weights or train the model first."
                                )

                            resnet.load_weights(weight_file)
                            ds = input_pipeline.get_DomainNet_data(config, workdir, [eval_domain], 'train')
                            [logits, representation] = resnet.predict(ds['data'])
                            test_data = {'logit': logits, 'label': ds['label'], 'representation': representation}


                        cal_models['TS Oracle'] = TemperatureScaling()
                        cal_models['TS Oracle'].fit(test_data)

                        ece_row = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                   'Train': eval_domain in trainer, 'Valid': eval_domain in three_domains,
                                   'Domain': eval_domain}
                        error_row = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                     'Train': eval_domain in trainer,
                                     'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        mce_row = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                   'Train': eval_domain in trainer,
                                   'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        loss_row = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                    'Train': eval_domain in trainer,
                                    'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        brier_row = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                     'Train': eval_domain in trainer,
                                     'Valid': eval_domain in three_domains, 'Domain': eval_domain}

                        ece_row_std = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                       'Train': eval_domain in trainer,
                                       'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        error_row_std = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                         'Train': eval_domain in trainer,
                                         'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        mce_row_std = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                       'Train': eval_domain in trainer,
                                       'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        loss_row_std = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                        'Train': eval_domain in trainer,
                                        'Valid': eval_domain in three_domains, 'Domain': eval_domain}
                        brier_row_std = {'Trainer': '_'.join(trainer), 'Calib': '_'.join(three_domains),
                                         'Train': eval_domain in trainer,
                                         'Valid': eval_domain in three_domains, 'Domain': eval_domain}

                        row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                                     'loss': loss_row, 'brier': brier_row}
                        row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                         'loss': loss_row_std, 'brier': brier_row_std}

                        test_results = dict()
                        for model in methods:
                            test_results[model] = dict()
                            for metric in row_dicts.keys():
                                test_results[model][metric] = []

                        for seed in range(20):
                            _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                                  test_data['representation'], test_size=500,
                                                                                  random_state=seed)

                            ds = {'logit': logits, 'label': y, 'representation': representation}

                            uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                            for key in row_dicts.keys():
                                test_results['Uncalibrated'][key].append(uncal_res[key])
                                #row_dicts[key]['Uncalibrated'] = uncal_res[key]

                            for model in cal_models.keys():
                                if model == 'Cluster':
                                    for mode in ['TS', 'NN', 'LR', 'Ensem']:
                                        cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                        m = mode if mode == 'TS' else f'{model} {mode}'
                                        for key in row_dicts.keys():
                                            test_results[m][key].append(cal_res[key])

                                elif 'CaliGen' in model:
                                    for mode in ['CaliGen', 'TS', 'Ensem']:
                                        cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                        if mode == 'CaliGen':
                                            m = model
                                        else:
                                            c = model.split(' ')
                                            m = f'{c[0]} {mode} {c[1]}'
                                        for key in row_dicts.keys():
                                            test_results[m][key].append(cal_res[key])

                                else:
                                    cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                                    for key in row_dicts.keys():
                                        test_results[model][key].append(cal_res[key])

                        for model in test_results.keys():
                            for key in row_dicts.keys():
                                row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                                row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                        df_ece = df_ece.append(ece_row, ignore_index=True)
                        df_error = df_error.append(error_row, ignore_index=True)
                        df_mce = df_mce.append(mce_row, ignore_index=True)
                        df_loss = df_loss.append(loss_row, ignore_index=True)
                        df_brier = df_brier.append(brier_row, ignore_index=True)

                        df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                        df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                        df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                        df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                        df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)


        path_to_res = os.path.join(config.res_path, config.dataset, config.loss)
        Path(path_to_res).mkdir(parents=True, exist_ok=True)

        df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std.csv"), index=False)

def kfold_cifar10c(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        clmns = ["Rho", "Error", "ECE", "MCE", "Loss", "Brier"]

        for sever in config.severity:
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, f"4filter_{sever}",
                                       "model_weights.h5")
            logits_path = os.path.join(config.logits_path, config.dataset, config.loss, f"4filter_{sever}")

            train_with_level = []
            for fil in config.domains.train_corruptions:
                if fil != 'original':
                    if sever == 'a':
                        for i in range(1, 6):
                            train_with_level.append(f"{fil}_{i}")
                    else:
                        train_with_level.append(f"{fil}_{sever}")
            train_with_level = ['original'] + train_with_level
            calib_with_level = []
            for fil in config.domains.calib_corruptions:
                if sever == 'a':
                    for i in range(1, 6):
                        calib_with_level.append(f"{fil}_{i}")
                else:
                    calib_with_level.append(f"{fil}_{sever}")

            if calib == 'in':
                filters = train_with_level + calib_with_level
            else:
                filters = calib_with_level

            df_kfold = pd.DataFrame(columns=clmns)

            calib_data = dict()
            if config.from_logits.lower() == 'yes':
                logits_path = os.path.join(config.logits_path, config.dataset, config.loss, f"4filter_{sever}")
                for d in filters:
                    logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                    with open(logit_file, 'rb') as f:
                        calib_data[d] = pickle.load(f)
            else:
                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )

                resnet.load_weights(weight_file)  # .expect_partial()
                for d in filters:
                    ds = input_pipeline.get_cifar10c_eval_data(config.dataset_dir, workdir, 'valid', d)
                    [logits, representation] = resnet.predict(ds['data'])
                    calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

            logits = []
            labels = []
            representations = []
            train_probs = []
            ts_model = TemperatureScaling()

            for d in calib_data.keys():
                logits.append(calib_data[d]['logit'])
                labels.append(calib_data[d]['label'])
                representations.append(calib_data[d]['representation'])
                ts_model.fit(calib_data[d])
                train_probs.append(softmax(calib_data[d]['logit'] / ts_model.temp))

            logits = np.concatenate(logits)
            representations = np.concatenate(representations)
            train_probs = np.concatenate(train_probs)
            labels = np.concatenate(labels).flatten()

            y_cat = keras.utils.to_categorical(labels, config.params.NUM_CLASSES)
            y_train = np.concatenate((train_probs, y_cat), axis=1)
            representations = representations.reshape(-1, 2048)

            n_folds = 3
            kfold = KFold(n_splits=n_folds, shuffle=True)

            path = os.path.join(config.calib_path, config.dataset, config.loss, calib,
                                f"4filter_{sever}")
            caligen_res_path = os.path.join(config.res_path, config.dataset, config.loss, 'kfold', 'caligen', calib,
                                            f"4filter_{sever}")
            Path(caligen_res_path).mkdir(parents=True, exist_ok=True)

            for rho in RHOS:
                error, ece, mce, loss, brier = 0, 0, 0, 0, 0
                i = 0
                for train, test in kfold.split(representations, y_train):
                    train_data = {'representation': representations[train], 'label': y_train[train]}
                    valid_data = {'representation': representations[test], 'label': y_train[test]}
                    caligen_path = os.path.join(path, 'caligen', f'fold_{i}')
                    Path(caligen_path).mkdir(parents=True, exist_ok=True)

                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                      'loss': config.loss, 'trainer': f"4filter_{sever}"}

                    caligen_model = CaliGenKFold(**caligen_kwargs)
                    caligen_model.fit(train_data, valid_data)

                    valid_probs = caligen_model.predict(valid_data)
                    valid_y = np.argmax(valid_data['label'][:, config.params.NUM_CLASSES:], axis=1)
                    eval_metric = evaluate(valid_probs, valid_y)

                    error += eval_metric['error']
                    ece += eval_metric['ece']
                    mce += eval_metric['mce']
                    loss += eval_metric['loss']
                    brier += eval_metric['brier']
                    i += 1

                row_dict = {"Rho": rho, "Error": error / n_folds, "ECE": ece / n_folds, "MCE": mce / n_folds,
                            "Loss": loss / n_folds, "Brier": brier / n_folds}

                df_kfold = df_kfold.append(row_dict, ignore_index=True)

            df_kfold.to_csv(os.path.join(caligen_res_path, "kfold_results.csv"))


def kfold_OfficeHome(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        clmns = ["Rho", "Error", "ECE", "MCE", "Loss", "Brier"]

        for trainer in config.domains:
            resnet = ResNet_101(config, last_layer_activation='linear')
            weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                       "model_weights.h5")
            logits_path = os.path.join(config.logits_path, config.dataset, config.loss, trainer)

            three_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
            three_domains.remove(trainer)

            for cd in three_domains:
                two_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
                two_domains.remove(trainer)
                two_domains.remove(cd)

                logging.info(
                    f"Trained on: {trainer} "
                    f"Calibration domains: {two_domains}"
                )

                df_kfold = pd.DataFrame(columns=clmns)

                if calib == 'in':
                    calib_domains = [trainer] + two_domains
                else:
                    calib_domains = two_domains

                logits = []
                labels = []
                representations = []
                train_probs = []
                ts_model = TemperatureScaling()

                calib_data = dict()
                if config.from_logits.lower() == 'yes':
                    for d in calib_domains:
                        logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                        with open(logit_file, 'rb') as f:
                            calib_data[d] = pickle.load(f)
                else:
                    if not os.path.exists(weight_file):
                        raise ValueError(
                            f"weights does not exists at {weight_file} "
                            f"Please specify correct place for weights or train the model first."
                        )

                    resnet.load_weights(weight_file)  # .expect_partial()
                    for d in calib_domains:
                        ds = input_pipeline.get_OfficeHome_data(config, workdir, d, 'valid')
                        [logits, representation] = resnet.predict(ds['data'])
                        calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                for d in calib_data.keys():
                    logits.append(calib_data[d]['logit'])
                    labels.append(calib_data[d]['label'])
                    representations.append(calib_data[d]['representation'])
                    ts_model.fit(calib_data[d])
                    train_probs.append(softmax(calib_data[d]['logit'] / ts_model.temp))

                logits = np.concatenate(logits)
                representations = np.concatenate(representations)
                train_probs = np.concatenate(train_probs)
                labels = np.concatenate(labels).flatten()

                y_cat = keras.utils.to_categorical(labels, config.params.NUM_CLASSES)
                y_train = np.concatenate((train_probs, y_cat), axis=1)
                representations = representations.reshape(-1, 2048)

                n_folds = 3
                kfold = KFold(n_splits=n_folds, shuffle=True)

                path = os.path.join(config.calib_path, config.dataset, config.loss, calib,
                                    trainer, "_".join(two_domains))
                caligen_res_path = os.path.join(config.res_path, config.dataset, config.loss, 'kfold', 'caligen', calib,
                                    trainer, "_".join(two_domains))
                Path(caligen_res_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    error, ece, mce, loss, brier = 0, 0, 0, 0, 0
                    i = 0
                    for train, test in kfold.split(representations, y_train):
                        train_data = {'representation': representations[train], 'label': y_train[train]}
                        valid_data = {'representation': representations[test], 'label': y_train[test]}
                        caligen_path = os.path.join(path, 'caligen', f'fold_{i}')
                        Path(caligen_path).mkdir(parents=True, exist_ok=True)

                        caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                          'loss': config.loss, 'trainer': [trainer]}

                        caligen_model = CaliGenKFold(**caligen_kwargs)
                        caligen_model.fit(train_data, valid_data)

                        valid_probs = caligen_model.predict(valid_data)
                        valid_y = np.argmax(valid_data['label'][:, config.params.NUM_CLASSES:], axis=1)
                        eval_metric = evaluate(valid_probs, valid_y)

                        error += eval_metric['error']
                        ece += eval_metric['ece']
                        mce += eval_metric['mce']
                        loss += eval_metric['loss']
                        brier += eval_metric['brier']
                        i += 1

                    row_dict = {"Rho": rho, "Error": error/n_folds, "ECE": ece/n_folds, "MCE": mce/n_folds,
                                "Loss": loss/n_folds, "Brier": brier/n_folds}

                    df_kfold = df_kfold.append(row_dict, ignore_index=True)

                df_kfold.to_csv(os.path.join(caligen_res_path, "kfold_results.csv"))

def kfold_DomainNet(config: ml_collections.ConfigDict, workdir: str):
    for calib in ['in', 'out']:
        clmns = ["Rho", "Error", "ECE", "MCE", "Loss", "Brier"]

        for i in range(len(config.domains)):
            for j in range(i + 1, len(config.domains)):
                trainer = [config.domains[i], config.domains[j]]
                resnet = ResNet_101(config, last_layer_activation='linear')
                weight_file = os.path.join(config.weights_path, config.dataset, config.loss, '_'.join(trainer),
                                           "model_weights.h5")
                logits_path = os.path.join(config.logits_path, config.dataset, config.loss, '_'.join(trainer))

                four_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                for t in trainer:
                    four_domains.remove(t)

                for cd in four_domains:
                    three_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                    for t in trainer:
                        three_domains.remove(t)
                    three_domains.remove(cd)

                    logging.info(
                        f"Trained on: {trainer} "
                        f"Calibration domains: {three_domains}"
                    )

                    df_kfold = pd.DataFrame(columns=clmns)

                    if calib == 'in':
                        calib_domains = trainer + three_domains
                    else:
                        calib_domains = three_domains

                    calib_data = dict()
                    source_data = dict()

                    if config.from_logits.lower() == 'yes':
                        for d in calib_domains:
                            logit_file = os.path.join(logits_path, f"valid_logits_{d}.p")
                            with open(logit_file, 'rb') as f:
                                calib_data[d] = pickle.load(f)
                    else:
                        if not os.path.exists(weight_file):
                            raise ValueError(
                                f"weights does not exists at {weight_file} "
                                f"Please specify correct place for weights or train the model first."
                            )

                        resnet.load_weights(weight_file)
                        for d in calib_domains:
                            ds = input_pipeline.get_DomainNet_data(config, workdir, [d], 'valid')
                            [logits, representation] = resnet.predict(ds['data'])
                            calib_data[d] = {'logit': logits, 'label': ds['label'], 'representation': representation}

                    logits = []
                    labels = []
                    representations = []
                    train_probs = []
                    ts_model = TemperatureScaling()

                    for d in calib_data.keys():
                        logits.append(calib_data[d]['logit'])
                        labels.append(calib_data[d]['label'])
                        representations.append(calib_data[d]['representation'])
                        ts_model.fit(calib_data[d])
                        train_probs.append(softmax(calib_data[d]['logit'] / ts_model.temp))

                    logits = np.concatenate(logits)
                    representations = np.concatenate(representations)
                    train_probs = np.concatenate(train_probs)
                    labels = np.concatenate(labels).flatten()

                    y_cat = keras.utils.to_categorical(labels, config.params.NUM_CLASSES)
                    y_train = np.concatenate((train_probs, y_cat), axis=1)
                    representations = representations.reshape(-1, 2048)

                    n_folds = 3
                    kfold = KFold(n_splits=n_folds, shuffle=True)

                    path = os.path.join(config.calib_path, config.dataset, config.loss, calib,
                                        '_'.join(trainer), cd)
                    caligen_res_path = os.path.join(config.res_path, config.dataset, config.loss, 'kfold', 'caligen',
                                                    calib, '_'.join(trainer), cd)
                    Path(caligen_res_path).mkdir(parents=True, exist_ok=True)

                    for rho in RHOS:
                        error, ece, mce, loss, brier = 0, 0, 0, 0, 0
                        i = 0
                        for train, test in kfold.split(representations, y_train):
                            train_data = {'representation': representations[train], 'label': y_train[train]}
                            valid_data = {'representation': representations[test], 'label': y_train[test]}
                            caligen_path = os.path.join(path, 'caligen', f'fold_{i}')
                            Path(caligen_path).mkdir(parents=True, exist_ok=True)

                            caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': False,
                                              'loss': config.loss, 'trainer': [trainer]}

                            caligen_model = CaliGenKFold(**caligen_kwargs)
                            caligen_model.fit(train_data, valid_data)

                            valid_probs = caligen_model.predict(valid_data)
                            valid_y = np.argmax(valid_data['label'][:, config.params.NUM_CLASSES:], axis=1)
                            eval_metric = evaluate(valid_probs, valid_y)

                            error += eval_metric['error']
                            ece += eval_metric['ece']
                            mce += eval_metric['mce']
                            loss += eval_metric['loss']
                            brier += eval_metric['brier']
                            i += 1

                        row_dict = {"Rho": rho, "Error": error / n_folds, "ECE": ece / n_folds, "MCE": mce / n_folds,
                                    "Loss": loss / n_folds, "Brier": brier / n_folds}

                        df_kfold = df_kfold.append(row_dict, ignore_index=True)

                    df_kfold.to_csv(os.path.join(caligen_res_path, "kfold_results.csv"))


def calibrate(config: ml_collections.ConfigDict, ablation=False):
    if ablation and config.dataset != 'OfficeHome':
        logging.info(
            f"We have done ablation study only on OfficeHome dataset. "
            f"This is fresh ablation study on {config.dataset}."
        )
    elif config.loss == 'focal' and config.dataset != 'OfficeHome':
        logging.info(
            f"We have done calibration evaluation with focal loss only on OfficeHome dataset. "
            f"This is fresh calibration learning on {config.dataset}."
        )
    resnet = ResNet_101(config, last_layer_activation='linear')
    trainers = get_train_domains(config)
    for calib in ['in', 'out']:
        for trainer in trainers:
            calib_domains = get_calib_domains(config, trainer)
            for cd in calib_domains.keys():
                ckm_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib, trainer,
                                        cd, 'cluster')
                caligen_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                            trainer, cd, 'caligen')
                scaligen_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                            trainer, cd, 'scaligen')
                imp_weight_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                            trainer, cd, 'imp_weight')
                cpcs_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                             trainer, cd, 'cpcs')
                transcal_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                             trainer, cd, 'transcal')
                if config.model == 'resnet':
                    ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                            'cluster')
                    caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                'caligen')
                    scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                'scaligen')
                    imp_weight_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                 'imp_weight')
                    cpcs_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                             'cpcs')
                    transcal_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                 'transcal')
                    if ablation:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                    'ablation', 'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                    'ablation', 'scaligen')
                calib_d = calib_domains[cd]
                train_domains = []
                if config.dataset == 'cifar10c':
                    sever = trainer.split('_')[1]
                    train_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions, sever)
                    ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                            'cluster')
                    imp_weight_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                   'imp_weight')
                    cpcs_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                             'cpcs')
                    transcal_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                 'transcal')
                    if ablation is False:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'scaligen')

                    else:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'ablation', 'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'ablation', 'scaligen')
                elif config.dataset == 'OfficeHome':
                    train_domains = [trainer]

                elif config.dataset == 'DomainNet':
                    train_domains = trainer.split('_')


                logging.info(
                    f"Training Domains: {train_domains} "
                    f"Calibration Domain: {calib_d}"
                )

                if calib == 'in':
                    calib_d = train_domains + calib_d

                weight_file = os.path.join(config.weights_path, config.model, config.dataset, config.loss, trainer,
                                           'model_weights.h5')
                if config.model == 'resnet':
                    weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                           'model_weights.h5')

                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )
                resnet.load_weights(weight_file)
                calib_data = get_data(config.dataset_dir, calib_d, 'valid')
                for d in calib_data:
                    [logits, representation] = resnet.predict(calib_data[d]['data'])
                    calib_data[d]['logit'] = logits
                    calib_data[d]['representation'] = representation

                calib_data = unpack_domains(calib_data)
                del calib_data["data"]

                source_data = get_data(config.dataset_dir, train_domains, 'valid')
                for d in source_data:
                    [logits, representation] = resnet.predict(source_data[d]['data'])
                    source_data[d]['logit'] = logits
                    source_data[d]['representation'] = representation

                source_data = unpack_domains(source_data)
                del source_data["data"]
                del source_data["y"]

                source_data_train = get_data(config.dataset_dir, train_domains, 'train')
                for d in source_data_train:
                    [logits, representation] = resnet.predict(source_data_train[d]['data'])
                    source_data_train[d]['logit'] = logits
                    source_data_train[d]['representation'] = representation

                source_data_train = unpack_domains(source_data_train)
                del source_data_train["data"]
                del source_data_train["y"]

                if ablation is False:
                    Path(ckm_path).mkdir(parents=True, exist_ok=True)

                    ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}

                    ckm = ClusterKMeans(**ckm_kwargs)
                    ckm.fit(calib_data)

                    Path(imp_weight_path).mkdir(parents=True, exist_ok=True)
                    imp_weight = get_importance_weight(source_data_train['representation'],
                                                       calib_data['representation'],
                                                       source_data['representation'],
                                                       imp_weight_path)
                    del source_data_train

                    Path(cpcs_path).mkdir(parents=True, exist_ok=True)
                    cpcs_kwargs = {'path': cpcs_path, 'weight': imp_weight}
                    cpcs = CPCS(**cpcs_kwargs)
                    cpcs.fit(calib_data)

                    source_probs = ckm.predict(source_data, mode='TS')
                    source_confs = np.mean(np.max(source_probs, axis=1))
                    source_preds = np.argmax(source_probs, axis=1)
                    accuracies = np.array([source_preds==source_data['label']], dtype=np.float16)
                    source_error = 1 - accuracies
                    source_error = np.reshape(source_error, (source_error.shape[1], 1))

                    Path(transcal_path).mkdir(parents=True, exist_ok=True)
                    transcal_kwargs = {'path': transcal_path, 'weight': imp_weight, 'bias': True, 'variance': True,
                                       'source_confidence': source_confs, 'error': source_error}
                    transcal = TransCal(**transcal_kwargs)
                    transcal.fit(calib_data)

                '''
                Path(caligen_path).mkdir(parents=True, exist_ok=True)
                Path(scaligen_path).mkdir(parents=True, exist_ok=True)

                scaligen_kwargs = {'config': config, 'path': scaligen_path, 'ablation': ablation,
                                  'loss': config.loss, 'trainer': trainer, 'repr': config.model_repr}

                scaligen = SCaliGenCalibration(**scaligen_kwargs)
                scaligen.fit(calib_data)
                
                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': ablation,
                                      'loss': config.loss, 'trainer': trainer, 'repr': config.model_repr}

                    caligen = CaliGenCalibration(**caligen_kwargs)
                    caligen.fit(calib_data)
                '''

def eval_performance(config: ml_collections.ConfigDict, workdir: str):
    performer = {'cifar10c': evaluate_cifar10c, 'OfficeHome': evaluate_OfficeHome, 'DomainNet': evaluate_DomainNet}

    performer[config.dataset](config, workdir)

    if config.dataset == 'OfficeHome':
        evaluate_ablation_OfficeHome(config, workdir)
        evaluate_limitation_OfficeHome(config, workdir)


def evaluate_performance(config: ml_collections.ConfigDict, ablation=False):

    if ablation and config.dataset != 'OfficeHome':
        logging.info(
            f"We have done ablation study only on OfficeHome dataset. "
            f"This is fresh ablation study on {config.dataset}."
        )
    elif config.loss == 'focal' and config.dataset != 'OfficeHome':
        logging.info(
            f"We have done calibration evaluation with focal loss only on OfficeHome dataset. "
            f"This is fresh calibration learning on {config.dataset}."
        )
    resnet = ResNet_101(config, last_layer_activation='linear')
    trainers = get_train_domains(config)
    for calib in ['in', 'out']:
        cal_models = dict()
        clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
        #methods = ['Uncalibrated', 'TS Source', 'TS Oracle', 'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab',
        #           'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem', 'SCaliGen', 'SCaliGen TS', 'SCaliGen TSA',
        #           'SCaliGen Ensem']
        methods = ['Uncalibrated', 'TS Source', 'TS Oracle', 'TS', 'Cluster NN', 'Cluster LR', 'Cluster Ensem',
                   'CPCS', 'TransCal', 'HB TopLabel']
        clmns += methods

        #for rho in RHOS:
        #    clmns += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
        #    methods += [f'CaliGen {rho}', f'CaliGen TS {rho}', f'CaliGen Ensem {rho}']
        df_ece = pd.DataFrame(columns=clmns)
        df_error = pd.DataFrame(columns=clmns)
        df_mce = pd.DataFrame(columns=clmns)
        df_loss = pd.DataFrame(columns=clmns)
        df_brier = pd.DataFrame(columns=clmns)

        df_ece_std = pd.DataFrame(columns=clmns)
        df_error_std = pd.DataFrame(columns=clmns)
        df_mce_std = pd.DataFrame(columns=clmns)
        df_loss_std = pd.DataFrame(columns=clmns)
        df_brier_std = pd.DataFrame(columns=clmns)

        for trainer in trainers:
            calib_domains = get_calib_domains(config, trainer)

            for cd in calib_domains.keys():
                ckm_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib, trainer,
                                        cd, 'cluster')
                caligen_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                            trainer, cd, 'caligen')
                scaligen_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                             trainer, cd, 'scaligen')
                imp_weight_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                               trainer, cd, 'imp_weight')
                cpcs_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                         trainer, cd, 'cpcs')
                transcal_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                             trainer, cd, 'transcal')
                if config.model == 'resnet':
                    ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                            'cluster')
                    caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                'caligen')
                    scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                 'scaligen')
                    imp_weight_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                   'imp_weight')
                    cpcs_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                             'cpcs')
                    transcal_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                 'transcal')
                    if ablation:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                    'ablation', 'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                     'ablation', 'scaligen')
                calib_d = calib_domains[cd].copy()
                train_domains = []
                if config.dataset == 'cifar10c':
                    sever = trainer.split('_')[1]
                    train_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions, sever)
                    ckm_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                            'cluster')
                    imp_weight_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                   'imp_weight')
                    cpcs_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                             'cpcs')
                    transcal_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                 'transcal')
                    if ablation is False:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                     'scaligen')

                    else:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'ablation', 'caligen')
                        scaligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                     'ablation', 'scaligen')
                elif config.dataset == 'OfficeHome':
                    train_domains = [trainer]

                elif config.dataset == 'DomainNet':
                    train_domains = trainer.split('_')

                logging.info(
                    f"Training Domains: {train_domains} "
                    f"Calibration Domain: {calib_d}"
                )

                if calib == 'in':
                    calib_d = train_domains + calib_d

                weight_file = os.path.join(config.weights_path, config.model, config.dataset, config.loss, trainer,
                                           'model_weights.h5')
                if config.model == 'resnet':
                    weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                               'model_weights.h5')

                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )
                resnet.load_weights(weight_file)
                calib_data = get_data(config.dataset_dir, calib_d, 'valid')
                for d in calib_data:
                    [logits, representation] = resnet.predict(calib_data[d]['data'])
                    calib_data[d]['logit'] = logits
                    calib_data[d]['representation'] = representation

                calib_data = unpack_domains(calib_data)
                del calib_data["data"]

                source_data = get_data(config.dataset_dir, train_domains, 'valid')
                for d in source_data:
                    [logits, representation] = resnet.predict(source_data[d]['data'])
                    source_data[d]['logit'] = logits
                    source_data[d]['representation'] = representation

                source_data = unpack_domains(source_data)
                del source_data["data"]
                del source_data["y"]

                source_data_train = get_data(config.dataset_dir, train_domains, 'train')
                for d in source_data_train:
                    [logits, representation] = resnet.predict(source_data_train[d]['data'])
                    source_data_train[d]['logit'] = logits
                    source_data_train[d]['representation'] = representation

                source_data_train = unpack_domains(source_data_train)
                del source_data_train["data"]
                del source_data_train["y"]

                if ablation is False:
                    Path(ckm_path).mkdir(parents=True, exist_ok=True)

                    ckm_kwargs = {'path': ckm_path, 'rho': config.CLUSTERS}

                    cal_models['Cluster'] = ClusterKMeans(**ckm_kwargs)
                    cal_models['Cluster'].fit(calib_data)

                    Path(imp_weight_path).mkdir(parents=True, exist_ok=True)
                    imp_weight = get_importance_weight(source_data_train['representation'],
                                                       calib_data['representation'],
                                                       source_data['representation'],
                                                       imp_weight_path)
                    del source_data_train
                    del source_data['representation']

                    Path(cpcs_path).mkdir(parents=True, exist_ok=True)
                    cpcs_kwargs = {'path': cpcs_path, 'weight': imp_weight}
                    cal_models['CPCS'] = CPCS(**cpcs_kwargs)
                    cal_models['CPCS'].fit(calib_data)

                    calib_probs = cal_models['Cluster'].predict(calib_data, mode='TS')
                    calib_confs = np.mean(np.max(calib_probs, axis=1))
                    calib_preds = np.argmax(calib_probs, axis=1)
                    accuracies = np.array([calib_preds == calib_data['label']], dtype=np.float16)
                    calib_error = 1 - accuracies
                    calib_error = np.reshape(calib_error, (calib_error.shape[1], 1))

                    Path(transcal_path).mkdir(parents=True, exist_ok=True)
                    transcal_kwargs = {'path': transcal_path, 'weight': imp_weight, 'bias': True, 'variance': True,
                                       'source_confidence': calib_confs, 'error': calib_error}
                    cal_models['TransCal'] = TransCal(**transcal_kwargs)
                    cal_models['TransCal'].fit(calib_data)


                '''
                Path(caligen_path).mkdir(parents=True, exist_ok=True)
                Path(scaligen_path).mkdir(parents=True, exist_ok=True)
                scaligen_kwargs = {'config': config, 'path': scaligen_path, 'ablation': ablation,
                                  'loss': config.loss, 'trainer': trainer, 'repr': config.model_repr}

                cal_models[f'SCaliGen'] = SCaliGenCalibration(**scaligen_kwargs)
                cal_models[f'SCaliGen'].fit(calib_data)
                
                for rho in RHOS:
                    caligen_kwargs = {'config': config, 'path': caligen_path, 'rho': rho, 'ablation': ablation,
                                      'loss': config.loss, 'trainer': trainer, 'repr': config.model_repr}

                    cal_models[f'CaliGen {rho}'] = CaliGenCalibration(**caligen_kwargs)
                    cal_models[f'CaliGen {rho}'].fit(calib_data)
                '''



                cal_models['TS Source'] = TemperatureScaling()
                cal_models['TS Source'].fit(source_data)

                '''
                cal_models['HB'] = HistogramBinning(**{'params': {'BINS': 15}})
                cal_models['HB'].fit(calib_data)

                cal_models['Isotonic'] = IsotonicCalibration(**{'params': {'y_min': 0, 'y_max': 1}})
                cal_models['Isotonic'].fit(calib_data)

                cal_models['Beta abm'] = BetaCalib(**{'params': {'parameters': "abm"}})
                cal_models['Beta abm'].fit(calib_data)

                cal_models['Beta am'] = BetaCalib(**{'params': {'parameters': "am"}})
                cal_models['Beta am'].fit(calib_data)

                cal_models['Beta ab'] = BetaCalib(**{'params': {'parameters': "ab"}})
                cal_models['Beta ab'].fit(calib_data)
                '''

                cal_models['HB TopLabel'] = HBTopLabel(**{'params': {'points_per_bin': 50}})
                cal_models['HB TopLabel'].fit(calib_data)

                eval_domains = config.domains
                if config.dataset == 'cifar10c':
                    eval_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions +
                                                                         config.domains.calib_corruptions +
                                                                         config.domains.test_corruptions, 'a')

                for eval_domain in eval_domains:
                    test_data = get_data(config.dataset_dir, [eval_domain], config.params.test)
                    for d in test_data:
                        [logits, representation] = resnet.predict(test_data[d]['data'])
                        test_data[d]['logit'] = logits
                        test_data[d]['representation'] = representation

                    test_data = unpack_domains(test_data, ts=False)
                    del test_data["data"]
                    del test_data["y"]

                    cal_models['TS Oracle'] = TemperatureScaling()
                    cal_models['TS Oracle'].fit(test_data)

                    calib_contexts = "" if config.dataset == 'cifar10c' else '_'.join(calib_domains[cd])


                    ece_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    error_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    mce_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    loss_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    brier_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}

                    ece_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                   'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    error_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                     'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    mce_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                   'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    loss_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                    'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                    brier_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                     'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}

                    row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                                 'loss': loss_row, 'brier': brier_row}
                    row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                     'loss': loss_row_std, 'brier': brier_row_std}


                    test_results = dict()
                    for model in methods:
                        test_results[model] = dict()
                        for metric in row_dicts.keys():
                            test_results[model][metric] = []

                    for seed in range(20):
                        _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                              test_data['representation'], test_size=500,
                                                                              random_state=seed)

                        ds = {'logit': logits, 'label': y, 'representation': representation}

                        uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                        for key in row_dicts.keys():
                            test_results['Uncalibrated'][key].append(uncal_res[key])
                            #row_dicts[key]['Uncalibrated'] = uncal_res[key]

                        for model in cal_models.keys():
                            if model == 'Cluster':
                                for mode in ['TS', 'NN', 'LR', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    m = mode if mode == 'TS' else f'{model} {mode}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            elif model == 'SCaliGen':
                                for mode in ['SCaliGen', 'TS', 'TSA', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    m = model if mode == 'SCaliGen' else f'{model} {mode}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            elif 'CaliGen' in model:
                                for mode in ['CaliGen', 'TS', 'Ensem']:
                                    cal_res = evaluate(cal_models[model].predict(ds, mode), ds['label'])
                                    if mode == 'CaliGen':
                                        m = model
                                    else:
                                        c = model.split(' ')
                                        m = f'{c[0]} {mode} {c[1]}'
                                    for key in row_dicts.keys():
                                        test_results[m][key].append(cal_res[key])

                            else:
                                cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                                for key in row_dicts.keys():
                                    test_results[model][key].append(cal_res[key])

                    for model in methods:
                        for key in row_dicts.keys():
                            row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                            row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                    df_ece = df_ece.append(ece_row, ignore_index=True)
                    df_error = df_error.append(error_row, ignore_index=True)
                    df_mce = df_mce.append(mce_row, ignore_index=True)
                    df_loss = df_loss.append(loss_row, ignore_index=True)
                    df_brier = df_brier.append(brier_row, ignore_index=True)

                    df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                    df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                    df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                    df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                    df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)


        path_to_res = os.path.join(config.res_path, config.model, config.dataset, config.loss)
        if ablation:
            path_to_res = os.path.join(config.res_path, config.model, config.dataset, config.loss, 'ablation')
        Path(path_to_res).mkdir(parents=True, exist_ok=True)

        df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean_more.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean_more.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean_more.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean_more.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean_more.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std_more.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std_more.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std_more.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std_more.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std_more.csv"), index=False)
        '''
        df_error.to_csv(os.path.join(path_to_res, f"{calib}_Error_mean.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"{calib}_ECE_mean.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"{calib}_MCE_mean.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"{calib}_Loss_mean.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"{calib}_Brier_mean.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"{calib}_Error_std.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"{calib}_ECE_std.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"{calib}_MCE_std.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"{calib}_Loss_std.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"{calib}_Brier_std.csv"), index=False)
        '''

def evaluate_limitation(config: ml_collections.ConfigDict):
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

    resnet = ResNet_101(config, last_layer_activation='linear')
    trainers = get_train_domains(config)
    cal_models = dict()
    clmns = ['Trainer', 'Calib', 'Train', 'Valid', 'Domain']
    methods = ['Uncalibrated', 'HB', 'Isotonic', 'Beta abm', 'Beta am', 'Beta ab', 'TS']
    clmns += methods

    df_ece = pd.DataFrame(columns=clmns)
    df_error = pd.DataFrame(columns=clmns)
    df_mce = pd.DataFrame(columns=clmns)
    df_loss = pd.DataFrame(columns=clmns)
    df_brier = pd.DataFrame(columns=clmns)

    df_ece_std = pd.DataFrame(columns=clmns)
    df_error_std = pd.DataFrame(columns=clmns)
    df_mce_std = pd.DataFrame(columns=clmns)
    df_loss_std = pd.DataFrame(columns=clmns)
    df_brier_std = pd.DataFrame(columns=clmns)

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

            ds_valid = get_data(config.dataset_dir, train_domains, 'valid')
            ds_calib = get_data(config.dataset_dir, calib_d, 'valid')

            for domain in ds_calib.keys():
                _, x_v, _, y_v = train_test_split(ds_calib[domain]['data'], ds_calib[domain]['label'],
                                                      test_size=0.5, random_state=6)
                ds_valid[domain] = dict()
                ds_valid[domain]['data'] = x_v
                ds_valid[domain]['label'] = y_v

            weight_file = os.path.join(filepath, "model_weights.h5")

            # logging info for weights
            logging.info(f"Weights file: {weight_file} "
                         f"Starting training, will take a while..."
                         )

            if os.path.exists(weight_file):
                resnet.load_weights(weight_file)  # .expect_partial()
            else:
                raise ValueError(
                    f"weights does not exists at {weight_file} "
                    f"Please specify correct place for weights or train the model first."
                )
            for d in ds_valid.keys():
                [logits, representation] = resnet.predict(ds_valid[d]['data'])
                ds_valid[d]['logit'] = logits
                ds_valid[d]['representation'] = representation

            ds_valid = unpack_domains(ds_valid)

            cal_models['TS'] = TemperatureScaling()
            cal_models['TS'].fit(ds_valid)

            cal_models['HB'] = HistogramBinning(**{'params': {'BINS': 15}})
            cal_models['HB'].fit(ds_valid)

            cal_models['Isotonic'] = IsotonicCalibration(**{'params': {'y_min': 0, 'y_max': 1}})
            cal_models['Isotonic'].fit(ds_valid)

            cal_models['Beta abm'] = BetaCalib(**{'params': {'parameters': "abm"}})
            cal_models['Beta abm'].fit(ds_valid)

            cal_models['Beta am'] = BetaCalib(**{'params': {'parameters': "am"}})
            cal_models['Beta am'].fit(ds_valid)

            cal_models['Beta ab'] = BetaCalib(**{'params': {'parameters': "ab"}})
            cal_models['Beta ab'].fit(ds_valid)

            eval_domains = config.domains
            if config.dataset == 'cifar10c':
                eval_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions +
                                                                     config.domains.calib_corruptions +
                                                                     config.domains.test_corruptions, 'a')

            for eval_domain in eval_domains:
                test_data = get_data(config.dataset_dir, [eval_domain], config.params.test)
                for d in test_data:
                    [logits, representation] = resnet.predict(test_data[d]['data'])
                    test_data[d]['logit'] = logits
                    test_data[d]['representation'] = representation

                test_data = unpack_domains(test_data, ts=False)
                del test_data["data"]
                del test_data["y"]

                calib_contexts = "" if config.dataset == 'cifar10c' else '_'.join(calib_domains[cd])

                ece_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                           'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                error_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                             'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                mce_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                           'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                loss_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                            'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                brier_row = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                             'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}

                ece_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                error_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                 'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                mce_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                               'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                loss_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}
                brier_row_std = {'Trainer': trainer, 'Calib': calib_contexts, 'Train': eval_domain in train_domains,
                                 'Valid': eval_domain in calib_domains[cd], 'Domain': eval_domain}

                row_dicts = {'error': error_row, 'ece': ece_row, 'mce': mce_row,
                             'loss': loss_row, 'brier': brier_row}
                row_dicts_std = {'error': error_row_std, 'ece': ece_row_std, 'mce': mce_row_std,
                                 'loss': loss_row_std, 'brier': brier_row_std}

                test_results = dict()
                for model in methods:
                    test_results[model] = dict()
                    for metric in row_dicts.keys():
                        test_results[model][metric] = []

                for seed in range(20):
                    _, logits, _, y, _, representation = train_test_split(test_data['logit'], test_data['label'],
                                                                          test_data['representation'], test_size=500,
                                                                          random_state=seed)

                    ds = {'logit': logits, 'label': y, 'representation': representation}

                    uncal_res = evaluate(softmax(ds['logit']), ds['label'])
                    for key in row_dicts.keys():
                        test_results['Uncalibrated'][key].append(uncal_res[key])
                        # row_dicts[key]['Uncalibrated'] = uncal_res[key]

                    for model in cal_models.keys():
                        cal_res = evaluate(cal_models[model].predict(ds), ds['label'])
                        for key in row_dicts.keys():
                            test_results[model][key].append(cal_res[key])

                for model in methods:
                    for key in row_dicts.keys():
                        row_dicts[key][model] = np.mean(np.array(test_results[model][key]))
                        row_dicts_std[key][model] = np.std(np.array(test_results[model][key]))

                df_ece = df_ece.append(ece_row, ignore_index=True)
                df_error = df_error.append(error_row, ignore_index=True)
                df_mce = df_mce.append(mce_row, ignore_index=True)
                df_loss = df_loss.append(loss_row, ignore_index=True)
                df_brier = df_brier.append(brier_row, ignore_index=True)

                df_ece_std = df_ece_std.append(ece_row_std, ignore_index=True)
                df_error_std = df_error_std.append(error_row_std, ignore_index=True)
                df_mce_std = df_mce_std.append(mce_row_std, ignore_index=True)
                df_loss_std = df_loss_std.append(loss_row_std, ignore_index=True)
                df_brier_std = df_brier_std.append(brier_row_std, ignore_index=True)

        path_to_res = os.path.join(config.res_path, config.model, config.dataset, config.loss, 'limitation')
        Path(path_to_res).mkdir(parents=True, exist_ok=True)

        df_error.to_csv(os.path.join(path_to_res, f"Error_mean.csv"), index=False)
        df_ece.to_csv(os.path.join(path_to_res, f"ECE_mean.csv"), index=False)
        df_mce.to_csv(os.path.join(path_to_res, f"MCE_mean.csv"), index=False)
        df_loss.to_csv(os.path.join(path_to_res, f"Loss_mean.csv"), index=False)
        df_brier.to_csv(os.path.join(path_to_res, f"Brier_mean.csv"), index=False)

        df_error_std.to_csv(os.path.join(path_to_res, f"Error_std.csv"), index=False)
        df_ece_std.to_csv(os.path.join(path_to_res, f"ECE_std.csv"), index=False)
        df_mce_std.to_csv(os.path.join(path_to_res, f"MCE_std.csv"), index=False)
        df_loss_std.to_csv(os.path.join(path_to_res, f"Loss_std.csv"), index=False)
        df_brier_std.to_csv(os.path.join(path_to_res, f"Brier_std.csv"), index=False)


def calibrateKFold(config: ml_collections.ConfigDict, workdir: str):
    kFold = {'cifar10c': kfold_cifar10c, 'OfficeHome': kfold_OfficeHome, 'DomainNet': kfold_DomainNet}

    kFold[config.dataset](config, workdir)

def calibrate_KFold(config: ml_collections.ConfigDict, ablation=False):
    if ablation and config.dataset != 'OfficeHome':
        logging.info(
        f"We have done ablation study only on OfficeHome dataset. "
        f"This is fresh ablation study on {config.dataset}."
        )
    elif config.loss == 'focal' and config.dataset != 'OfficeHome':
        logging.info(
        f"We have done calibration evaluation with focal loss only on OfficeHome dataset. "
        f"This is fresh calibration learning on {config.dataset}."
        )
    resnet = ResNet_101(config, last_layer_activation='linear')
    trainers = get_train_domains(config)
    for calib in ['in', 'out']:
        clmns = ["Rho", "Error", "ECE", "MCE", "Loss", "Brier"]
        for trainer in trainers:
            calib_domains = get_calib_domains(config, trainer)
            for cd in calib_domains.keys():

                df_kfold = pd.DataFrame(columns=clmns)
                caligen_path = os.path.join(config.calib_path, config.model, config.dataset, config.loss, calib,
                                            trainer, cd, 'caligen')
                if config.model == 'resnet':
                    caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                'caligen')
                    if ablation:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer, cd,
                                                    'ablation')
                calib_d = calib_domains[cd]
                train_domains = []
                if config.dataset == 'cifar10c':
                    sever = trainer.split('_')[1]
                    train_domains = ['original'] + get_filters_with_level(config.domains.train_corruptions, sever)
                    if ablation is False:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                'caligen')
                    else:
                        caligen_path = os.path.join(config.calib_path, config.dataset, config.loss, calib, trainer,
                                                    'ablation')
                elif config.dataset == 'OfficeHome':
                    train_domains = [trainer]

                elif config.dataset == 'DomainNet':
                    train_domains = trainer.split('_')


                logging.info(
                    f"Training Domains: {train_domains} "
                    f"Calibration Domain: {calib_d}"
                )

                if calib == 'in':
                    calib_d = train_domains + calib_d

                weight_file = os.path.join(config.weights_path, config.model, config.dataset, config.loss, trainer,
                                           'model_weights.h5')
                if config.model == 'resnet':
                    weight_file = os.path.join(config.weights_path, config.dataset, config.loss, trainer,
                                           'model_weights.h5')

                if not os.path.exists(weight_file):
                    raise ValueError(
                        f"weights does not exists at {weight_file} "
                        f"Please specify correct place for weights or train the model first."
                    )
                resnet.load_weights(weight_file)
                calib_data = get_data(config.dataset_dir, calib_d, 'valid')
                for d in calib_data:
                    [logits, representation] = resnet.predict(calib_data[d]['data'])
                    calib_data[d]['logit'] = logits
                    calib_data[d]['representation'] = representation

                calib_data = unpack_domains(calib_data)
                del calib_data["data"]

                n_folds = 3
                kfold = KFold(n_splits=n_folds, shuffle=True)

                caligen_res_path = os.path.join(config.res_path, config.model, config.dataset, config.loss, 'kfold',
                                                'caligen', calib, trainer, cd)
                if config.model == 'resnet':
                    caligen_res_path = os.path.join(config.res_path, config.dataset, config.loss, 'kfold',
                                                'caligen', calib, trainer, cd)

                Path(caligen_res_path).mkdir(parents=True, exist_ok=True)

                for rho in RHOS:
                    error, ece, mce, loss, brier = 0, 0, 0, 0, 0
                    i = 0
                    for train, test in kfold.split(calib_data['representation'], calib_data['y']):
                        train_data = {'label': calib_data['label'][train], 'logit': calib_data['logit'][train],
                                      'representation': calib_data['representation'][train], 'y': calib_data['y'][train]}
                        valid_data = {'label': calib_data['label'][test], 'logit': calib_data['logit'][test],
                                      'representation': calib_data['representation'][test], 'y': calib_data['y'][test]}

                        weight_path = os.path.join(caligen_path, f'fold_{i}')
                        Path(weight_path).mkdir(parents=True, exist_ok=True)

                        caligen_kwargs = {'config': config, 'path': weight_path, 'rho': rho, 'ablation': ablation,
                                          'loss': config.loss, 'trainer': trainer, 'repr': config.model_repr}

                        caligen_model = CaliGenCalibration(**caligen_kwargs)
                        caligen_model.fit(train_data, valid_data)

                        valid_probs = caligen_model.predict(valid_data, mode='CaliGen')
                        eval_metric = evaluate(valid_probs, valid_data['label'])

                        error += eval_metric['error']
                        ece += eval_metric['ece']
                        mce += eval_metric['mce']
                        loss += eval_metric['loss']
                        brier += eval_metric['brier']
                        i += 1

                    row_dict = {"Rho": rho, "Error": error/n_folds, "ECE": ece/n_folds, "MCE": mce/n_folds,
                                "Loss": loss/n_folds, "Brier": brier/n_folds}

                    df_kfold = df_kfold.append(row_dict, ignore_index=True)

                df_kfold.to_csv(os.path.join(caligen_res_path, "kfold_results.csv"))


def calibrate_ablation(config: ml_collections.ConfigDict, workdir: str):
    calibrate_ablation_OfficeHome(config, workdir)

def eval_performance_ablation(config: ml_collections.ConfigDict, workdir: str):
    #evaluate_ablation_OfficeHome(config, workdir)
    evaluate_limitation_OfficeHome(config, workdir)



