import logging as python_logging
import os, pickle
import threading
import numpy as np
from absl import logging
import tensorflow as tf
import warnings
from sklearn.linear_model import LogisticRegression

_EPSILON = 1e-7

def _get_logits(output, from_logits, op_type, fn_name):
    '''
    Function to return logits
    :param output: output of model
    :param from_logits: True or False
    :param op_type: Softmax
    :param fn_name: loss
    :return: logits and from_logits
    '''
    output_ = output
    from_logits_ = from_logits

    has_keras_logits = hasattr(output, "_keras_logits")
    if has_keras_logits:
        output_ = output._keras_logits
        from_logits_ = True

    from_expected_op_type = (
        not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable))
        and output.op.type == op_type
    ) and not has_keras_logits

    if from_expected_op_type:
        # When softmax activation function is used for output operation, we
        # use logits from the softmax function directly to compute loss in order
        # to prevent collapsing zero when training.
        # See b/117284466
        assert len(output.op.inputs) == 1
        output_ = output.op.inputs[0]
        from_logits_ = True

    if from_logits and (has_keras_logits or from_expected_op_type):
        warnings.warn(
            f'"`{fn_name}` received `from_logits=True`, but '
            f"the `output` argument was produced by a {op_type} "
            "activation and thus does not represent logits. "
            "Was this intended?",
            stacklevel=2,
        )

    return output_, from_logits_

def categorical_focal_loss_back(target, output, from_logits=False, axis=-1):
    '''
    Focal loss function backend for training
    :param target: Groundtruth
    :param output: Output
    :param from_logits: True or False
    :param axis: Axis along the batch = 1
    :return: loss for the batch
    '''
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)

    output, from_logits = _get_logits(
        output, from_logits, "Softmax", "categorical_focal_loss"
    )
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=output, axis=axis
        )

    # scale preds so that the class probas of each sample sum to 1
    output = output / tf.reduce_sum(output, axis, True)
    # Compute cross entropy from probabilities.

    gamma = np.full((output.shape[0], output.shape[1]), 5.0)
    gamma[output >= 0.5] = 3.0
    gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
    epsilon_ = tf.constant(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target * tf.pow(1-output, gamma) * tf.math.log(output), axis)

def categorical_focal_loss(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):
    '''
    Focal loss function for training
    :param y_true: Groundtruth
    :param y_pred: Predictions
    :param from_logits: True or False
    :param label_smoothing: Level of lable smoothing
    :param axis: For the batch. 1
    :return: Focal loss
    '''
    if isinstance(axis, bool):
        raise ValueError(
            f"`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    def _smooth_labels():
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )

    return categorical_focal_loss_back(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )


def softmax(x, axis = 1):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    m = np.max(x, axis=axis)
    m = m[:, np.newaxis]
    e_x = np.exp(x - m)
    #print("m shape: ", m.shape)
    #print("e_x shape: ", e_x.shape)
    return e_x / e_x.sum(axis=axis, keepdims=1)

def brier_multi(y_true, y_prob, weight=1):
    '''
    Calculate the Brier score for multiclass predictions
    :param y_true: Groundtruth
    :param y_prob: Prediction probs
    :param weight: Weight ratio
    :return: Brier score
    '''
    return np.mean(np.sum((y_prob - y_true) ** 2, axis=1) * weight)

def unpack_dict(domains_data):
    logits = []
    labels = []
    representations = []
    if 'label' not in domains_data.keys():
        for domain in domains_data.keys():
            logits.append(domains_data[domain]['logit'])
            labels.append(domains_data[domain]['label'])
            if 'representation' in domains_data[domain].keys():
                representations.append(domains_data[domain]['representation'])
    else:
        logits = domains_data['logit']
        labels = domains_data['label']
        if 'representation' in domains_data.keys():
            representations = domains_data['representation']

    return logits, labels, representations

def get_importance_weight(source_train_feature, calib_feature, source_val_feature, path):
    """
    :param source_train_feature: shape [n_tr, d], features from training set
    :param calib_feature: shape [n_t, d], features from test set
    :param source_val_feature: shape [n_v, d], features from validation set
    :return:
    """
    weight_file = os.path.join(path, 'imp_weight.p')
    if os.path.exists(weight_file):
        logging.info(
            f"Loading Importance Weight from file: {weight_file}"
        )
        with open(weight_file, 'rb') as f:
            weight = pickle.load(f)
    else:
        print("-"*30 + "get_weight" + '-'*30)
        n_tr, d = source_train_feature.shape
        n_c, _d = calib_feature.shape
        n_v, _d = source_val_feature.shape
        print("n_tr: ", n_tr, "n_v: ", n_v, "n_t: ", n_c, "d: ", d)

        sample_num = n_c
        if n_tr < n_c:
            sample_index = np.random.choice(n_tr,  n_c, replace=True)
            source_train_feature = source_train_feature[sample_index]
            sample_num = n_c
        elif n_tr > n_c:
            sample_index = np.random.choice(n_c, n_tr, replace=True)
            calib_feature = calib_feature[sample_index]
            sample_num = n_tr

        combine_feature = np.concatenate((source_train_feature, calib_feature))
        combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
        domain_classifier = LogisticRegression(max_iter=1000, verbose=1)
        domain_classifier.fit(combine_feature, combine_label)
        domain_out = domain_classifier.predict_proba(source_val_feature)
        weight = domain_out[:, :1] / domain_out[:, 1:]
        with open(os.path.join(path, 'imp_weight.p'), 'wb') as f:
            pickle.dump(weight, f)
    return weight

class calibrationError:
    def __init__(self, conf, pred, ground_truth, bin_size=0.1):
        '''
        Class to calculate Calibration Errors and bin info

        Args:
            conf (numpy.ndarray): list of confidences
            pred (numpy.ndarray): list of predictions
            ground_truth (numpy.ndarray): list of true labels
            bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        '''
        self.conf = conf
        self.pred = pred
        self.ground_truth = ground_truth
        self.bin_size = bin_size

        self.ECE = None
        self.MCE = None
        self.BIN_INFO = dict()

    def _compute_acc_bin(self, conf_thresh_lower, conf_thresh_upper):
        '''
        # Computes accuracy and average confidence for bin

        Args:
            conf_thresh_lower (float): Lower Threshold of confidence interval
            conf_thresh_upper (float): Upper Threshold of confidence interval

        Returns:
            (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
        '''

        filtered_tuples = [x for x in zip(self.pred, self.ground_truth, self.conf) if
                           x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
        if len(filtered_tuples) < 1:
            return 0, 0, 0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct) / len_bin  # accuracy of BIN
            return accuracy, avg_conf, len_bin


    def calculate_errors(self):

        upper_bounds = np.arange(self.bin_size, 1 + self.bin_size, self.bin_size)  # Get bounds of bins

        n = len(self.conf)
        ece = 0  # Starting error
        cal_errors = []
        accuracies = []
        confidences = []
        bin_lengths = []

        for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
            acc, avg_conf, len_bin = self._compute_acc_bin(conf_thresh - self.bin_size, conf_thresh)
            accuracies.append(acc)
            confidences.append(avg_conf)
            bin_lengths.append(len_bin)
            ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE
            cal_errors.append(np.abs(acc - avg_conf))

        self.ECE = ece
        self.MCE = max(cal_errors)
        self.BIN_INFO["accuracies"] = accuracies
        self.BIN_INFO["confidences"] = confidences
        self.BIN_INFO["bin_lengths"] = bin_lengths

        return self.ECE * 100, self.MCE * 100, self.BIN_INFO

class CaliGenCELoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, rho=0.2):
        super(CaliGenCELoss, self).__init__()
        self.rho = rho
        self.num_classes = num_classes
        self.kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        y_kl = y_true[:, 0:self.num_classes]
        y_cce = y_true[:, self.num_classes:]
        kl = self.kl(y_kl, y_pred)
        cce = self.cce(y_cce, y_pred)
        return (1.0 - self.rho) * cce + self.rho * kl

class CaliGenFocalLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, rho=0.2):
        super(CaliGenFocalLoss, self).__init__()
        self.rho = rho
        self.num_classes = num_classes
        self.kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.cfl = categorical_focal_loss

    def call(self, y_true, y_pred):
        y_kl = y_true[:, 0:self.num_classes]
        y_cfl = y_true[:, self.num_classes:]
        kl = self.kl(y_kl, y_pred)
        cfl = self.cfl(y_cfl, y_pred)
        return (1.0 - self.rho) * cfl + self.rho * kl


class GFileHandler(python_logging.StreamHandler):
    """Writes log messages to file using tf.io.gfile."""

    def __init__(self, filename, mode, flush_secs=1.0):
        super().__init__()
        tf.io.gfile.makedirs(os.path.dirname(filename))
        if mode == 'a' and not tf.io.gfile.exists(filename):
            mode = 'w'
        self.filehandle = tf.io.gfile.GFile(filename, mode)
        self.flush_secs = flush_secs
        self.flush_timer = None

    def flush(self):
        self.filehandle.flush()

    def emit(self, record):
        msg = self.format(record)
        self.filehandle.write(f'{msg}\n')
        if self.flush_timer is not None:
            self.flush_timer.cancel()
        self.flush_timer = threading.Timer(self.flush_secs, self.flush)
        self.flush_timer.start()


def add_gfile_logger(workdir, *, basename='train', level=python_logging.INFO):
    """Adds GFile file logger to Python logging handlers."""
    fh = GFileHandler(f'{workdir}/{basename}.log', 'a')
    fh.setLevel(level)
    fh.setFormatter(logging.PythonFormatter())
    python_logging.getLogger('').addHandler(fh)
