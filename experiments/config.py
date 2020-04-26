from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6
__C.OUTPUT_PATH = ''
__C.MODELS = edict()
__C.MODELS.AttnGAN = edict()
__C.MODELS.AttnGAN.RUN = True
__C.MODELS.AttnGAN.WORKING_DIR = ''
__C.MODELS.AttnGAN.MODEL_PY_PATH = ''
__C.MODELS.AttnGAN.CONFIG_PATH = ''
__C.MODELS.AttnGAN.CONFIG_PY_PATH = ''
__C.MODELS.AttnGAN.G_NET_WEIGHTS_PATH = ''
__C.MODELS.AttnGAN.TEXT_ENCODER_WEIGHTS_PATH = ''
__C.MODELS.AttnGAN.IMAGE_ENCODER_WEIGHTS_PATH = ''
#__C.MODELS.DMM_GAN = edict()




def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)