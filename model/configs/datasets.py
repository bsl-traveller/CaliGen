from model.configs import common
#from resnet.configs import models


def get_config(dataset):
    """Returns default parameters for dataset preparation on `dataset`."""
    config = common.with_dataset(common.get_config(), dataset)

    return config