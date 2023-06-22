import ml_collections, os

def get_config():
    """Returns config values."""
    config = ml_collections.ConfigDict()
    # dataset and directory
    config.dataset = None
    config.dataset_dir = None

    # training weight base directory
    config.model = "resnet" # or efficientnet or vgg16 or resnet

    config.weights_path = "model_weights" # weights base path
    config.logits_path = "logits" # logits base path
    config.calib_path = "calib_models" # calibration models base path
    config.res_path = 'results'
    config.kfold_path = 'results_kfold'

    # train common params
    config.EPOCHS = 350
    config.PATIENCE = 100
    config.WEIGHT_DECAY = 0.0001
    config.BATCH_SIZE = 128
    config.loss = 'crossentropy' # update for focal loss

    # mode = train, evaluate, calibrate
    config.mode = 'calibrate'

    # If weights or logits exists then continue training or replace weights or WARN
    config.weights_exists = 'error' # should be either 'error' or 'continue' or 'replace'
    config.logits_exists = 'error' # should be either 'error' or 'replace'

    # Common Calibration params
    config.BINS = 15
    config.CLUSTERS = 9 # for OfficeHome 8
    config.caligen_first_layer = 512 # for domainnet 1024
    config.caligen_second_layer = 128 # for domainnet 512
    config.model_repr = 2048 # for efficientnet 1280

    config.from_logits = 'No' # To load saved logits, set it to 'Yes'

    return config.lock()


DATASET_PRESETS = {
    'cifar10c': ml_collections.ConfigDict(
        {'dataset': 'cifar10c',
         'dataset_dir': os.path.join('..', 'DataSets', 'cifar10c'),
         'params': ml_collections.ConfigDict(
             {'train': 'train',
              'valid': 'valid',
              'test': 'test',
              'CROP': 32,
              'NUM_CLASSES': 10}),
         'domains': ml_collections.ConfigDict(
             {'train_corruptions': ['original', 'gaussian_noise', 'brightness',
                                    'pixelate', 'gaussian_blur'],
              'calib_corruptions': ['fog', 'contrast', 'elastic_transform',
                                    'saturate'],
              'test_corruptions': ['shot_noise', 'impulse_noise', 'defocus_blur',
                                   'glass_blur', 'zoom_blur', 'jpeg_compression',
                                   'speckle_noise']})
         }),
    'officehome': ml_collections.ConfigDict(
        {'dataset': 'OfficeHome',
         'dataset_dir': os.path.join('..', 'DataSets', 'OfficeHome'),
         'CLUSTERS': 8,
         'params': ml_collections.ConfigDict(
             {'train': 'train',
              'valid': 'valid',
              'test': 'train',
              'CROP': 224,
              'NUM_CLASSES': 65}),
         'domains': ['Art', 'Clipart', 'Product', 'RealWorld']
         }),
    'domainnet': ml_collections.ConfigDict(
        {'dataset': 'DomainNet',
         'dataset_dir': os.path.join('..', 'DataSets', 'DomainNet'),
         'caligen_first_layer': 1024, # for domainnet 1024
         'caligen_second_layer': 512, # for domainnet 512
         'params': ml_collections.ConfigDict(
             {'train': 'train',
              'valid': 'valid',
              'test': 'train',
              'CROP': 224,
              'NUM_CLASSES': 345}),
         'domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
         })
}


def with_dataset(config: ml_collections.ConfigDict,
                 dataset: str) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict(config.to_dict())
    if 'cifar10c' in dataset:
        dataset, severity = dataset.split(',')
        config.severity = severity
    config.update(DATASET_PRESETS[dataset.lower()])

    return config

