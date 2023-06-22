# CaliGen
## Official code for the paper Generality-training of a Classifier for Improved Calibration in Unseen Contexts


### This Project include code for CaliGen model.

The environment.yml file contains the required packages and libraries

#### Step 1 (Dataset download):

cifar10c:
cifar10 dataset is available at https://www.cs.toronto.edu/~kriz/cifar.html
Please download the python version.
The code with preprocess can also download and preprocess the dataset

```python -u -m main --workdir=$(pwd)/logs/cifar10c-$(date +%s) --config=$(pwd)/model/configs/datasets.py:cifar10c,1 --config.mode=preprocess```

Office-Home:
Please download the dataset manually from https://www.hemanthdv.org/officeHomeDataset.html
After downloading the dataset, Please unzip it and rename it as OfficeHome and place it in DataSets folder

```python -u -m main --workdir=$(pwd)/logs/OfficeHome-$(date +%s) --config=$(pwd)/model/configs/datasets.py:OfficeHome --config.mode=preprocess```

DomainNet:
Please download the saperate datasets for 'clipart', 'infograph', 'painting', 'quickdraw', 'real' and 'sketch'
from http://ai.bu.edu/M3SDA/
Download clean version.
place each extracted folder in DataSets/DomainNet folder

```python -u -m main --workdir=$(pwd)/logs/DomainNet-$(date +%s) --config=$(pwd)/model/configs/datasets.py:DomainNet --config.mode=preprocess```

### Step 2 (Model training):
Choose the DNN architecture, dataset and loss function to train the model
For DNN: ResNet, dataset: Office-Home and loss: crossentropy, use the following command:

```python -u -m main --workdir=$(pwd)/logs/DomainNet-$(date +%s) --config=$(pwd)/model/configs/datasets.py:OfficeHome --config.mode=train --config.loss=crossentropy```,

#### Step 3 (Calibration training):
The calibrated model weights are now in calib_models after stage 2.

If you want to calibrate run for each dataset, loss and model.

```python -u -m main --workdir=$(pwd)/logs/OfficeHome-$(date +%s) --config=$(pwd)/model/configs/datasets.py:OfficeHome --config.model=resnet --config.loss=crossentropy --config.mode=calibrate```


#### Step 4 (Eval Performance):
To evaluate performance run for each dataset.

```python -u -m main --workdir=$(pwd)/logs/OfficeHome-$(date +%s) --config=$(pwd)/model/configs/datasets.py:OfficeHome --config.model=resnet --config.loss=crossentropy --config.mode=eval_performance```


#### Step 5 (Notebooks):
To visualise the results, there are jupyter notebooks in Notebooks folder
