import numpy as np
from scipy.optimize import minimize, fmin

from sklearn.metrics import log_loss, brier_score_loss
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import ml_collections
import os

from betacal import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from absl import logging

from model.model_utils import softmax, brier_multi, CaliGenCELoss, CaliGenFocalLoss

def get_class_conf(probs, labels):
    res = []
    for (p, l) in zip(probs, labels):
        res.append(p[l])
    return np.array(res)

def get_th_conf(probs, labels, th):
    max_conf = np.max(probs, axis=1)
    th_ind = max_conf >= th
    th_conf = probs[th_ind]
    th_labels = labels[th_ind]
    rej = (len(probs) - len(th_conf)) * 100 / len(probs)
    return {'probs': th_conf, 'labels': th_labels, 'reject': rej}


class CalibrationMethod:
    def __init__(self, logits=True):
        self.model = dict()
        self.logits = logits

    def fit(self, ds):
        '''
        fit method to fit the calibration curve
        :param data: dictionary of logit and groundtruth
        '''
        if self.logits:
            probs = softmax(ds['logit'])
        else:
            probs = ds['logit']
        K = probs.shape[1]
        # Go through all the classes
        for k in range(K):
            # Prep class labels (1 fixed true class, 0 other classes)
            y_one = np.array(ds['label'] == k, dtype="int")  # [:, 0]

            self._fit_single(probs[:, k], y_one, k)

    def _fit_single(self, probs, true, cl):
        return NotImplementedError("method _fit_single is not implemented in derived class.")

    def predict(self, ds):
        '''
        :param dictionay with key logit containing logits
        :return: calibrated probabilities
        '''
        if self.logits:
            probs = softmax(ds['logit'])
        else:
            probs = ds['logit']

        K = probs.shape[1]

        # Go through all the classes
        for k in range(K):
            # Go through all the probs and check what confidence is suitable for it.
            probs[:, k] = self._predict_single(probs[:, k], k)

        # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
        idx_nan = np.where(np.isnan(probs))
        probs[idx_nan] = 0

        return probs

    def _predict_single(self, probs, cl):
        return NotImplementedError("method _predict_single is not implemented in derived class.")


class HistogramBinning(CalibrationMethod):
    '''
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    The class is inherited from class CalibrationMethod.
    It overrides methods _fit_single and _predict_single
    '''

    def __init__(self, logits=True, **kwargs):
        '''
        :param kwargs:
        kwargs['config']:  config dictionary
        '''
        super().__init__()
        self.logits = logits
        self.params = kwargs['params']
        self.bin_size = 1. / self.params['BINS']  # Calculate bin size
        self.conf = dict()  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1 + self.bin_size, self.bin_size)  # Set bin bounds for intervals
        self.upper_bounds[-1] = 1.0

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        '''
        :param conf_thresh_lower: (float) start of the interval (not included)
        :param conf_thresh_upper: (float): end of the interval (included)
        :param probs: list of probabilities.
        :param true: list with true labels, where 1 is positive class and 0 is negative).
        :return: confidence
        '''

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered) / nr_elems  # Sums positive classes
            return conf

    def _fit_single(self, probs, true, cl):
        conf = []
        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh,
                                       probs=probs, true=true)
            conf.append(temp_conf)

        self.model[cl] = np.array(conf)

    def _predict_single(self, probs, cl):
        return self.model[cl][np.searchsorted(self.upper_bounds, probs)]


class HBTopLabel:
    '''
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    The class is inherited from class CalibrationMethod.
    It overrides methods _fit_single and _predict_single
    '''

    def __init__(self, **kwargs):
        '''
        :param kwargs:
        kwargs['config']:  config dictionary
        '''
        super().__init__()
        self.model = dict()
        self.params = kwargs['params']
        self.points_per_bin = self.params['points_per_bin']

    def fit(self, ds):
        '''
        fit method to fit the calibration curve
        :param data: dictionary of logit and groundtruth
        '''
        probs = softmax(ds['logit'])  # Softmax logits
        top_score = np.max(probs, axis=1).squeeze()
        pred_class = (np.argmax(probs, axis=1)).squeeze()

        K = probs.shape[1]

        for k in range(K):
            pred_k_indices = np.where(pred_class == k)
            n_k = np.size(pred_k_indices)

            bins_k = np.floor(n_k / self.points_per_bin).astype('int')
            if (bins_k != 0):
                kwargs = {'params': {'BINS': bins_k}}
                probs_k = np.zeros((n_k, 2))
                probs_k[:, 0] = 1 - top_score[pred_k_indices]
                probs_k[:, 1] = top_score[pred_k_indices]
                hb = HistogramBinning(logits=False, **kwargs)
                hb.fit({'logit': probs_k, 'label': ds['label'][pred_k_indices] == k})
                self.model[k] = hb

    def predict(self, ds):
        '''
        :param dictionay with key logit containing logits
        :return: calibrated probabilities
        '''
        probs = softmax(ds['logit'])
        top_score = np.max(probs, axis=1).squeeze()
        pred_class = (np.argmax(probs, axis=1)).squeeze()

        K = probs.shape[1]
        n = probs.shape[0]
        probs_top = np.zeros((n, 2))
        for k in range(K):
            pred_k_indices = np.where(pred_class == k)
            n_k = np.size(pred_k_indices)

            probs_k = np.zeros((n_k, 2))
            probs_k[:, 0] = 1 - top_score[pred_k_indices]
            probs_k[:, 1] = top_score[pred_k_indices]
            probs_top[pred_k_indices] = probs_k
            if (k in self.model.keys()):
                probs_top[pred_k_indices] = self.model[k].predict({'logit': probs_k})

        for (i, j) in zip(range(n), pred_class):
            c = list(range(j)) + list(range(j + 1, K))
            probs[i, c] += (probs[i, j] - probs_top[i, 1]) / (K - 1)
            probs[i, j] = probs_top[i, 1]

        return probs


class IsotonicCalibration(CalibrationMethod):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs['params']

    def _fit_single(self, probs, true, cl):
        self.model[cl] = IsotonicRegression(**self.params)
        self.model[cl].fit(probs, true)

    def _predict_single(self, probs, cl):
        return self.model[cl].predict(probs)


class BetaCalib(CalibrationMethod):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs['params']

    def _fit_single(self, probs, true, cl):
        self.model[cl] = BetaCalibration(**self.params)
        self.model[cl].fit(probs, true)

    def _predict_single(self, probs, cl):
        return self.model[cl].predict(probs)


class TemperatureScaling:
    def __init__(self):
        self.temp = 1
        self.maxiter = 50
        self.solver = 'BFGS'

    def _loss_fun(self, x, logits, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        return log_loss(y_true=true, y_pred=softmax(logits / x), labels=np.arange(logits.shape[1]))

    # Find the temperature
    def fit(self, ds):
        '''
        fit method to fit the calibration curve
        :param ds: dictionary of logit, representations and groundtruth
                '''
        opt = minimize(self._loss_fun, x0=1, args=(ds['logit'], ds['label']), options={'maxiter': self.maxiter},
                       method=self.solver)
        self.temp = opt.x[0]

    def predict(self, ds):
        '''
        Scales logits based on the temperature and returns calibrated probabilities
        :param dictionay with key logit
        :return: calibrated probabilities (nd.array with shape [samples, classes])
        '''

        return softmax(ds['logit'] / self.temp)

class CPCS:
    '''
    This code is based on https://github.com/thuml/TransCal
    '''
    def __init__(self, **kwargs):
        super(CPCS, self).__init__()
        self.weight = kwargs['weight']
        self.path = kwargs['path']
        self.temp = None

    def _loss_fun(self, x, logits, labels):
        "x ==> temperature T"
        scaled_logits = logits / x
        softmaxes = softmax(scaled_logits)

        ## Transform to onehot encoded labels
        labels_onehot = keras.utils.to_categorical(labels, logits.shape[1])
        return brier_multi(softmaxes, labels_onehot, weight=self.weight)

    def fit(self, ds):
        temp_file = os.path.join(self.path, 'temperature.p')

        if os.path.exists(temp_file):
            logging.info(
                f"Loading TransCal Temperature from file: {temp_file}"
            )
            with open(temp_file, 'rb') as f:
                self.temp = pickle.load(f)
        else:
            optimal_parameter = fmin(self._loss_fun, 2.0, args=(ds['logit'], ds['label']), disp=False)
            self.temp = optimal_parameter[0]

            with open(temp_file, 'wb') as f:
                pickle.dump(self.temp, f)

    def predict(self, ds):
        return softmax(ds['logit'] / self.temp)

class TransCal:
    '''
    This code is based on https://github.com/thuml/TransCal
    '''
    def __init__(self, **kwargs):
        super(TransCal, self).__init__()
        self.path = kwargs['path']
        self.bias_term = kwargs['bias']
        self.variance_term = kwargs['variance']
        self.source_confidence = kwargs['source_confidence']
        self.weight = kwargs['weight']
        self.error = kwargs['error']
        self.temp = None

    def _loss_fun(self, x, logits, error):
        "x[0] ==> temperature T"
        scaled_logits = logits / x[0]

        "x[1] ==> learnable meta parameter \lambda"
        if self.bias_term:
            controled_weight = self.weight ** x[1]
        else:
            controled_weight = self.weight

        ## 1. confidence
        max_L = np.max(scaled_logits, axis=1, keepdims=True)
        exp_L = np.exp(scaled_logits - max_L)
        softmaxes = exp_L / np.sum(exp_L, axis=1, keepdims=True)
        confidences = np.max(softmaxes, axis=1)
        confidence = np.mean(confidences)

        ## 2. accuracy
        if self.variance_term:
            weighted_error = controled_weight * error
            cov_1 = np.cov(np.concatenate((weighted_error, controled_weight), axis=1), rowvar=False)[0][1]
            var_w = np.var(controled_weight, ddof=1)
            eta_1 = - cov_1 / (var_w)

            cv_weighted_error = weighted_error + eta_1 * (controled_weight - 1)
            correctness = 1 - error
            cov_2 = np.cov(np.concatenate((cv_weighted_error, correctness), axis=1), rowvar=False)[0][1]
            var_r = np.var(correctness, ddof=1)
            eta_2 = - cov_2 / (var_r)

            target_risk = np.mean(weighted_error) + eta_1 * np.mean(controled_weight) - eta_1 \
                          + eta_2 * np.mean(correctness) - eta_2 * self.source_confidence
            estimated_acc = 1.0 - target_risk
        else:
            weighted_error = controled_weight * error
            target_risk = np.mean(weighted_error)
            estimated_acc = 1.0 - target_risk

        # return loss
        return np.abs(confidence - estimated_acc)

    def fit(self, ds):
        pkl_file = ''
        if self.bias_term:
            pkl_file += 'bias_'
        if self.variance_term:
            pkl_file += 'variance_'
        pkl_file += 'temperature.p'
        temp_file = os.path.join(self.path, pkl_file)

        if os.path.exists(temp_file):
            logging.info(
                f"Loading TransCal Temperature from file: {temp_file}"
            )
            with open(temp_file, 'rb') as f:
                self.temp = pickle.load(f)
        else:
            logits = ds['logit']
            error = self.error
            bnds = ((1.0, None), (0.0, 1.0))
            optimal_parameter = minimize(self._loss_fun, np.array([2.0, 0.5]), args=(logits, error), method='SLSQP', bounds=bnds)
            self.temp = optimal_parameter.x[0]

            with open(temp_file, 'wb') as f:
                pickle.dump(self.temp, f)

    def predict(self, ds):
        return softmax(ds['logit'] / self.temp)


class ClusterKMeans:
    def __init__(self, **kwargs):
        self.path = kwargs['path']
        self.clusters = kwargs['rho']
        self.cluster_temps = None
        self.kmeans = None
        self.ts_temp = None
        self.lr = None

    def fit(self, ds):
        '''
        fit method to fit the calibration curve
        :param ds: dictionary of logit, representations and groundtruth
                '''
        kmeans_file = os.path.join(self.path, f'kmeans_{self.clusters}.p')
        lr_file = os.path.join(self.path, f'lr_model_{self.clusters}.p')
        cluster_temp_file = os.path.join(self.path, f'cluster_temps_{self.clusters}.p')
        ts_temp_file = os.path.join(self.path, f'temperature_{self.clusters}.p')

        if os.path.exists(ts_temp_file):
            logging.info(
                f"Loading temperature from path: {ts_temp_file}"
            )
            with open(ts_temp_file, 'rb') as f:
                self.ts_temp = pickle.load(f)
        else:
            ts_model = TemperatureScaling()
            ts_model.fit(ds)
            self.ts_temp = ts_model.temp

        if os.path.exists(kmeans_file) and os.path.exists(lr_file) and os.path.exists(cluster_temp_file):
            logging.info(
                f"Loading kMeans model from path: {self.path}"
            )
            with open(kmeans_file, 'rb') as f:
                self.kmeans = pickle.load(f)
            with open(cluster_temp_file, 'rb') as f:
                self.cluster_temps = pickle.load(f)
            with open(lr_file, 'rb') as f:
                self.lr = pickle.load(f)
        else:
            logging.info(
                f"Training Cluster Model"
            )

            self.kmeans = KMeans(n_clusters=self.clusters, verbose=1, random_state=0).fit(ds['representation'])
            with open(kmeans_file, 'wb') as f:
                pickle.dump(self.kmeans, f)

            temperatures = []
            for i in range(self.clusters):
                temp_logits = ds['logit'][self.kmeans.labels_ == i]
                temp_true = ds['label'][self.kmeans.labels_ == i]
                temp_repr = ds['representation'][self.kmeans.labels_ == i]

                temp_model = TemperatureScaling()
                temp_model.fit({'logit': temp_logits, 'label': temp_true, 'representation': temp_repr})
                temperatures.append(temp_model.temp)

            representation_train = self.kmeans.cluster_centers_
            self.cluster_temps = np.array(temperatures)
            with open(cluster_temp_file, 'wb') as f:
                pickle.dump(self.cluster_temps, f)

            self.lr = LinearRegression().fit(representation_train, self.cluster_temps)
            with open(lr_file, 'wb') as f:
                pickle.dump(self.lr, f)

    def get_temp(self, ds, mode='TS'):
        if mode == 'TS':
            return self.ts_temp

        if mode == 'NN':
            labels_nn = self.kmeans.predict(ds['representation'])
            temps_nn = np.array([self.cluster_temps[i] for i in labels_nn])
            return np.mean(temps_nn)

        if mode == 'LR':
            temp_lr = self.lr.predict(ds['representation'])
            return np.mean(temp_lr)

    def predict(self, ds, mode='TS'):
        logits_temp = ds['logit'] / self.ts_temp
        if mode == 'TS':
            return softmax(logits_temp)

        labels_nn = self.kmeans.predict(ds['representation'])
        temps_nn = np.array([self.cluster_temps[i] for i in labels_nn])
        logits_nn = ds['logit'] / temps_nn[:, None]
        if mode == 'NN':
            return softmax(logits_nn)

        temp_lr = self.lr.predict(ds['representation'])
        logits_lr = ds['logit'] / temp_lr[:, None]
        if mode == 'LR':
            return softmax(logits_lr)

        logits_mean = np.mean(np.array([logits_temp, logits_nn, logits_lr]), axis=0)
        if mode == 'Ensem':
            return softmax(logits_mean)

        raise ValueError(
                f"Please provide mode argument in the predict method "
                f"mode should be in ['TS', 'NN', 'LR', 'Ensem'] "
            )


class CaliGenCalibration:
    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.path = kwargs['path']
        self.rho = kwargs['rho']
        self.loss = kwargs['loss']
        self.trainer = kwargs['trainer']
        self.ablation = kwargs['ablation']
        self.repr = kwargs['repr']
        self.num_classes = self.config.params.NUM_CLASSES
        self.model = None
        self.ts_temp = None
        self.caligen_temp = None
        self.caligen_temp_all = None

    def _build_and_compile_model(self, num_classes, activation='linear'):
        i = layers.Input(shape=(self.repr,))
        if self.ablation is False:
            h = layers.Dropout(0.5)(i)
            h = layers.Dense(self.config.caligen_first_layer, kernel_regularizer=l2(0.01), activation='relu')(h)
            h = layers.Dropout(0.5)(h)
            h = layers.Dense(self.config.caligen_second_layer, kernel_regularizer=l2(0.01), activation='relu')(h)
            h = layers.Dropout(0.5)(h)
            p = layers.Dense(num_classes, activation=activation)(h)
        else:
            p = layers.Dense(num_classes, activation=activation)(i)

        model = tf.keras.models.Model(inputs=i, outputs=p)

        if activation == 'softmax':
            if self.loss == 'crossentropy':
                model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                              loss=CaliGenCELoss(num_classes=num_classes, rho=self.rho))
            elif self.loss == 'focal':
                model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                              loss=CaliGenFocalLoss(num_classes=num_classes, rho=self.rho))
            else:
                raise ValueError(
                    f"loss should be either 'crossentropy' or 'focal' "
                    f"Provide correct value during initialization "
                )
        return model

    def fit(self, ds, ds_valid=None):
        weight_file = os.path.join(self.path, f'model_weights_{self.rho}.h5')
        ts_file = os.path.join(self.path, f'ts_temp_{self.rho}.p')
        caligen_temp_file = os.path.join(self.path, f'caligen_temp_{self.rho}.p')

        ts_model = TemperatureScaling()
        ts_model.fit(ds)
        self.ts_temp = ts_model.temp
        with open(ts_file, 'wb') as f:
            pickle.dump(self.ts_temp, f)

        if os.path.exists(weight_file):
            logging.info(
                f"Loading CaliGen model from path: {self.path}"
            )
            self.model = self._build_and_compile_model(self.num_classes, "linear")
            self.model.load_weights(weight_file)

        else:
            logging.info(
                f"Training CaliGen model "
                f"Weights to be saved at {weight_file}"
            )

            if ds_valid is None:
                representation_train, representation_val, y_train, y_val = train_test_split(ds['representation'],
                                                                                            ds['y'], test_size=0.2,
                                                                                            random_state=22)
            else:
                representation_train, y_train =  ds['representation'], ds['y']
                representation_val, y_val = ds_valid['representation'], ds_valid['y']

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_file,
                                                                           monitor='val_loss',
                                                                           mode='min',
                                                                           save_best_only=True,
                                                                           save_weights_only=True)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
            callbacks = [model_checkpoint_callback, es]

            self.model = self._build_and_compile_model(self.num_classes, "softmax")

            hist = self.model.fit(representation_train, y_train, batch_size=64, verbose=1,
                                  validation_data=(representation_val, y_val), callbacks=callbacks, epochs=600)

            self.model = self._build_and_compile_model(num_classes=self.num_classes, activation='linear')
            self.model.load_weights(weight_file)

        logits = self.model.predict(ds['representation'])
        ts_model = TemperatureScaling()
        ts_model.fit({'logit': logits, 'label': ds['label']})
        self.caligen_temp = ts_model.temp
        with open(caligen_temp_file, 'wb') as f:
            pickle.dump(self.caligen_temp, f)


    def predict(self, ds, mode='CaliGen'):
        logits_cg = self.model.predict(ds['representation'])
        if mode == 'CaliGen':
            return softmax(logits_cg)
        elif mode == 'TS':
            return softmax(logits_cg / self.caligen_temp)

        logits_temp = ds['logit'] / self.ts_temp
        logits_mean = np.mean(np.array([logits_temp, logits_cg]), axis=0)
        if mode == 'Ensem':
            return softmax(logits_mean)

        raise ValueError(
                f"Please provide mode argument in the predict method "
                f"mode should be in ['TS', 'CaliGen', 'Ensem'] "
            )
