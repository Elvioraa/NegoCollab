"""Opt-in DataLoader batch override loaded before the stage runner imports Torch.

The runner passes its YAML batch size to DataLoader. Only explicit
``shuffle=True`` loaders may be overridden; YAML is the default, not a limit.
Without NEGO_RUNTIME_BATCH_SIZE, do not import Torch or patch any class.
"""
import os
import sys


def _install(value):
    import functools
    import inspect
    import re

    if re.fullmatch(r'[0-9]+', value) is None or int(value) < 1:
        raise ValueError('NEGO_RUNTIME_BATCH_SIZE must be a positive integer')
    runtime_batch_size = int(value)

    from torch.utils.data import DataLoader

    original_init = DataLoader.__init__
    if getattr(original_init, '__nego_runtime_batch_hook__', False):
        return
    signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def runtime_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_batch_size = bound.arguments['batch_size']
        training = bound.arguments['shuffle'] is True
        if training:
            if type(original_batch_size) is not int or original_batch_size < 1:
                raise ValueError('Runtime batch override requires a positive original YAML batch_size')
            bound.arguments['batch_size'] = runtime_batch_size
        # Preserve Torch's own sampler, batch_sampler and argument validation.
        original_init(*bound.args, **bound.kwargs)
        print('[nego_runtime_batch] shuffle={} actual_batch_size={} original_yaml_batch_size={}'.format(
            bound.arguments['shuffle'], self.batch_size, original_batch_size), file=sys.stderr, flush=True)

    runtime_init.__nego_runtime_batch_hook__ = True
    # Patch in place so all DataLoader import aliases retain the same class.
    DataLoader.__init__ = runtime_init
    print('[nego_runtime_batch] enabled NEGO_RUNTIME_BATCH_SIZE={}'.format(runtime_batch_size),
          file=sys.stderr, flush=True)


if 'NEGO_RUNTIME_BATCH_SIZE' in os.environ:
    try:
        _install(os.environ['NEGO_RUNTIME_BATCH_SIZE'])
    except Exception as exc:
        # Python normally reports sitecustomize exceptions and keeps running.
        # A requested override must fail closed instead of silently using YAML.
        print('[nego_runtime_batch] FATAL: {}'.format(exc), file=sys.stderr, flush=True)
        os._exit(2)
