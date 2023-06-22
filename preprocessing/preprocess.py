import pickle
import ml_collections
import random
import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import shuffle

from preprocessing import corruptions
#import corruptions

def unpickle_batch(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def get_corrupted_example(x, corruption_type, severity=1):
    """Return corrupted images.
    Args:
      x: numpy array, uncorrupted image.
    Returns:
      numpy array, corrupted images.
    """
    x = np.clip(x, 0, 255)

    return {
        'gaussian_noise': corruptions.gaussian_noise,
        'shot_noise': corruptions.shot_noise,
        'impulse_noise': corruptions.impulse_noise,
        'defocus_blur': corruptions.defocus_blur,
        'glass_blur': corruptions.glass_blur,
        'zoom_blur': corruptions.zoom_blur,
        'fog': corruptions.fog,
        'brightness': corruptions.brightness,
        'contrast': corruptions.contrast,
        'elastic_transform': corruptions.elastic_transform,
        'pixelate': corruptions.pixelate,
        'jpeg_compression': corruptions.jpeg_compression,
        'gaussian_blur': corruptions.gaussian_blur,
        'saturate': corruptions.saturate,
        'speckle_noise': corruptions.speckle_noise,
    }[corruption_type](x, severity)

CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'zoom_blur',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'gaussian_blur',
    'saturate',
    'speckle_noise'
]
valid_filter = [
    'gaussian_noise',
    'brightness',
    'pixelate',
    'gaussian_blur',
    'fog',
    'contrast',
    'elastic_transform',
    'saturate'
    ]


def process_cifar10c(config: ml_collections.ConfigDict):
    target_path = config.dataset_dir
    data_file = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    try:
        os.system(f"wget {data_file}")
        os.system("tar -xf cifar-10-python.tar.gz")
    except:
        print("Please download manually from the given Link: ")
        print("Website of the Data: https://www.cs.toronto.edu/~kriz/cifar.html")
        print("Download and extract the python version")

    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    data_X = []
    data_y = []
    for file in files:
        batch = unpickle_batch(os.path.join('cifar-10-batches-py', file))
        X = batch[b'data']
        X = X.reshape(len(X), 3, 32, 32)
        X = X.transpose(0, 2, 3, 1)
        data_X.append(X)
        data_y.append(batch[b'labels'])
    data_X = np.concatenate(data_X)
    data_y = np.concatenate(data_y)

    data_cw = {i: data_X[data_y == i] for i in range(10)}

    splits = ['train', 'valid']

    random.seed(6)
    samples = {'train': 20, 'valid': 20}  # , 'test': 200}
    for s in splits:
        data_path = os.path.join(target_path, s)
        Path(data_path).mkdir(parents=True, exist_ok=True)
        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                sample_X = []
                sample_y = []
                if (s == 'train' and corruption in valid_filter):
                    for i in range(10):
                        data_cw[i], x_class, _, y_class = train_test_split(data_cw[i],
                                                                           np.full(len(data_cw[i]), i, dtype=int),
                                                                           test_size=samples[s], random_state=6)

                        sample_X.append(x_class)
                        sample_y.append(y_class)
                    sample_X = np.concatenate(sample_X)
                    sample_y = np.concatenate(sample_y).flatten()
                    sample_X, sample_y = shuffle(sample_X, sample_y, random_state=6)

                    print("Generating images for corruption: ", corruption)

                    print("Severity level: ", severity)
                    images = []
                    for image in sample_X:
                        images.append(get_corrupted_example(image, corruption, severity))

                    name_string = corruption + "_" + str(severity)

                    images = np.asarray(images)

                    np.save(os.path.join(data_path, name_string + '_X.npy'), images)
                    np.save(os.path.join(data_path, name_string + '_y.npy'), sample_y)
                elif s == 'valid':
                    for i in range(10):
                        data_cw[i], x_class, _, y_class = train_test_split(data_cw[i],
                                                                           np.full(len(data_cw[i]), i, dtype=int),
                                                                           test_size=samples[s], random_state=6)

                        sample_X.append(x_class)
                        sample_y.append(y_class)
                    sample_X = np.concatenate(sample_X)
                    sample_y = np.concatenate(sample_y).flatten()
                    sample_X, sample_y = shuffle(sample_X, sample_y, random_state=6)

                    print("Generating images for corruption: ", corruption)

                    print("Severity level: ", severity)
                    images = []
                    for image in sample_X:
                        images.append(get_corrupted_example(image, corruption, severity))

                    name_string = corruption + "_" + str(severity)

                    images = np.asarray(images)

                    np.save(os.path.join(data_path, name_string + '_X.npy'), images)
                    np.save(os.path.join(data_path, name_string + '_y.npy'), sample_y)

        if s == 'valid':
            sample_X = []
            sample_y = []
            for i in range(10):
                data_cw[i], x_class, _, y_class = train_test_split(data_cw[i], np.full(len(data_cw[i]), i, dtype=int),
                                                                   test_size=samples[s], random_state=6)

                sample_X.append(x_class)
                sample_y.append(y_class)

            sample_X = np.concatenate(sample_X)
            sample_y = np.concatenate(sample_y).flatten()

            np.save(os.path.join(data_path, 'original_X.npy'), sample_X)
            np.save(os.path.join(data_path, 'original_y.npy'), sample_y)

    data = []
    label = []
    for i in range(10):
        data.append(data_cw[i])
        label.append(np.full(len(data_cw[i]), i, dtype=int))
    data = np.concatenate(data)
    label = np.concatenate(label).flatten()
    data, label = shuffle(data, label, random_state=6)
    data_path = os.path.join(target_path, 'train')
    np.save(os.path.join(data_path, 'original_X.npy'), data)
    np.save(os.path.join(data_path, 'original_y.npy'), label)

    # Processing Test images
    data_path = os.path.join(target_path, 'test')
    Path(data_path).mkdir(parents=True, exist_ok=True)

    batch = unpickle_batch(os.path.join('cifar-10-batches-py', 'test_batch'))
    data_X = batch[b'data']
    data_X = data_X.reshape(len(data_X), 3, 32, 32)
    data_X = data_X.transpose(0, 2, 3, 1)
    data_y = batch[b'labels']

    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            images = []
            for image in data_X:
                images.append(get_corrupted_example(image, corruption, severity))

            name_string = corruption + "_" + str(severity)

            images = np.asarray(images)

            np.save(os.path.join(data_path, name_string + '_X.npy'), images)
            np.save(os.path.join(data_path, name_string + '_y.npy'), data_y)

    np.save(os.path.join(data_path, 'original_X.npy'), data_X)
    np.save(os.path.join(data_path, 'original_y.npy'), data_y)

def process_OfficeHome(config: ml_collections.ConfigDict):
    image_path = config.dataset_dir

    domains = ['Product', 'Art', 'Clipart', 'Real World']

    classess = [f.path.split('/')[-1] for f in os.scandir(os.path.join(image_path, domains[0])) if f.is_dir()]
    classess.sort()
    classess = {c: i for i, c in enumerate(classess)}

    for domain in domains:
        print('Processing domain: ', domain)
        train_data = []
        train_label = []
        valid_data = []
        valid_label = []
        for c in classess:
            class_data = []
            images = glob.glob(os.path.join(image_path, domain, c, '*.jpg')) + glob.glob(
                os.path.join(image_path, domain, c, '*.png'))
            for image in images:
                d = np.asarray(Image.open(image))
                d = tf.image.convert_image_dtype(d, tf.float32)  # equivalent to dividing image pixels by 255
                d = tf.image.resize(d, (224, 224))
                class_data.append(d)
            class_data = np.array(class_data)
            l = np.full(class_data.shape[0], classess[c], dtype=int)
            train_X, valid_X, train_y, valid_y = train_test_split(class_data, l, test_size=0.2, random_state=6)
            train_data.append(train_X)
            train_label.append(train_y)
            valid_data.append(valid_X)
            valid_label.append(valid_y)

        train_data = np.concatenate(train_data)
        train_label = np.concatenate(train_label).flatten()
        valid_data = np.concatenate(valid_data)
        valid_label = np.concatenate(valid_label).flatten()

        train_data, train_label = shuffle(train_data, train_label, random_state=6)
        valid_data, valid_label = shuffle(valid_data, valid_label, random_state=6)

        train_path = os.path.join(image_path, 'train')
        valid_path = os.path.join(image_path, 'valid')
        Path(train_path).mkdir(parents=True, exist_ok=True)
        Path(valid_path).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(train_path, ''.join(domain.split(' ')) + '_X.npy'), train_data)
        np.save(os.path.join(train_path, ''.join(domain.split(' ')) + '_y.npy'), train_label)
        np.save(os.path.join(valid_path, ''.join(domain.split(' ')) + '_X.npy'), valid_data)
        np.save(os.path.join(valid_path, ''.join(domain.split(' ')) + '_y.npy'), valid_label)

def process_DomainNet(config: ml_collections.ConfigDict):
    image_path = config.dataset_dir

    clipart = "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"
    infograph = "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"
    painting = "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"
    quickdraw = "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
    real = "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"
    sketch = "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"

    files = {'clipart': clipart, 'infograph': infograph, 'painting': painting, 'quickdraw': quickdraw,
             'real': real, 'sketch': sketch}

    domains = ['quickdraw', 'sketch', 'infograph', 'clipart', 'real', 'painting']

    try:
        for domain in domains:
            os.system(f"wget '{files[domain]}'")
            os.system(f"unzip {domain}.zip -d {image_path}")
            os.system(f"rm {domain}.zip")
    except:
        print("Please download manually from the given Link: ")
        print("Website of the Data: http://ai.bu.edu/M3SDA/")
        print("Download and extract the to DataSets/DomainNet folder")

    classess = [f.path.split('/')[-1] for f in os.scandir(os.path.join(image_path, domains[0])) if f.is_dir()]
    classess.sort()
    classess = {c: i for i, c in enumerate(classess)}

    for domain in domains:
        print('Processing domain: ', domain)
        train_data = []
        train_label = []
        valid_data = []
        valid_label = []
        for c in classess:
            class_data = []
            images = glob.glob(os.path.join(image_path, domain, c, '*.jpg')) + glob.glob(
                os.path.join(image_path, domain, c, '*.png'))
            for image in images:
                d = np.asarray(Image.open(image))
                d = tf.image.convert_image_dtype(d, tf.float32)  # equivalent to dividing image pixels by 255
                d = tf.image.resize(d, (224, 224))
                d = tf.image.convert_image_dtype(d, tf.uint8)
                class_data.append(d)
            class_data = np.array(class_data)
            l = np.full(class_data.shape[0], classess[c], dtype=int)
            train_X, valid_X, train_y, valid_y = train_test_split(class_data, l, test_size=0.1, random_state=6)
            train_data.append(train_X)
            train_label.append(train_y)
            valid_data.append(valid_X)
            valid_label.append(valid_y)

        train_data = np.concatenate(train_data)
        train_label = np.concatenate(train_label).flatten()
        valid_data = np.concatenate(valid_data)
        valid_label = np.concatenate(valid_label).flatten()

        train_data, train_label = shuffle(train_data, train_label, random_state=6)
        valid_data, valid_label = shuffle(valid_data, valid_label, random_state=6)

        train_path = os.path.join(image_path, 'train')
        valid_path = os.path.join(image_path, 'valid')
        Path(train_path).mkdir(parents=True, exist_ok=True)
        Path(valid_path).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(train_path, domain + '_X.npy'), train_data)
        np.save(os.path.join(train_path, domain + '_y.npy'), train_label)
        np.save(os.path.join(valid_path, domain + '_X.npy'), valid_data)
        np.save(os.path.join(valid_path, domain + '_y.npy'), valid_label)

        #os.system(f"rm -rf {image_path}/{domain}")

def process(config: ml_collections.ConfigDict):
    process_data = {'cifar10c': process_cifar10c, 'OfficeHome': process_OfficeHome, 'DomainNet': process_DomainNet}
    process_data[config.dataset](config)

