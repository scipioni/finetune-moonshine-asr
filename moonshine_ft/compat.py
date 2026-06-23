"""
Runtime compatibility patches for third-party libraries on Python 3.14+.

Python 3.14 changed pickle.save_dict to call _batch_setitems(items, obj) with
two arguments. datasets 3.x (≤3.6.0) overrides _batch_setitems(self, items)
with only one, causing a TypeError. datasets 4.x fixed this but requires
torchcodec for audio decoding (CUDA-only; incompatible with ROCm).

Remove this file once we can upgrade to datasets 4.x on ROCm.
"""
import sys

if sys.version_info >= (3, 14):
    try:
        import dill as _dill
        import datasets.utils._dill as _datasets_dill

        # Replicate datasets 3.x logic but forward extra args to the parent
        # (Python 3.14 changed pickle.save_dict to call _batch_setitems(items, obj);
        # datasets 3.x only accepts (items), so the second arg raises TypeError).
        # Mirrors the fix in datasets 4.x: _batch_setitems(self, items, *args, **kwargs).
        def _patched_batch_setitems(self, items, *args, **kwargs):
            if getattr(self, '_legacy_no_dict_keys_sorting', False):
                return _dill.Pickler._batch_setitems(self, items, *args, **kwargs)
            try:
                items = sorted(items)
            except Exception:
                try:
                    from datasets.fingerprint import Hasher
                    items = sorted(items, key=lambda x: Hasher.hash(x[0]))
                except Exception:
                    pass
            return _dill.Pickler._batch_setitems(self, items, *args, **kwargs)

        _datasets_dill.Pickler._batch_setitems = _patched_batch_setitems
    except (ImportError, AttributeError):
        pass
