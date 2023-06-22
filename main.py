from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
import tensorflow as tf

from model import model_utils
from model import steps
from calibration import calibrate
from preprocessing import preprocess


FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string('workdir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    model_utils.add_gfile_logger(_WORKDIR.value, basename=FLAGS.config.mode)

    logging.info(f'Config: {FLAGS.config}')

    if FLAGS.config.mode == 'preprocess':
        preprocess.process(FLAGS.config)
    if FLAGS.config.mode == 'train':
        steps.train(FLAGS.config)
    elif FLAGS.config.mode == 'limitation':
        steps.train_limitation(FLAGS.config)
        calibrate.evaluate_limitation(FLAGS.config)
    elif FLAGS.config.mode == 'calibrate':
        calibrate.calibrate(FLAGS.config)
    elif FLAGS.config.mode == 'eval_performance':
        calibrate.evaluate_performance(FLAGS.config)
    elif FLAGS.config.mode == 'ablation':
        #calibrate.evaluate_performance(FLAGS.config, ablation=True)
        calibrate.calibrate_KFold(FLAGS.config, ablation=True)
    elif FLAGS.config.mode == 'kfold':
        calibrate.calibrate_KFold(FLAGS.config)
    else:
        raise app.UsageError(f'Unknown config.mode: {FLAGS.config.mode}')

if __name__ == '__main__':
    app.run(main)

