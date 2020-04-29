from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict


__C = edict()
cfg = __C


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # recursively merge dicts
        if type(v) is edict:
            try:
                b[k] = edict()
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