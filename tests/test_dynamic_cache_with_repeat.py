# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import importlib
import importlib.machinery
import os
import sys
import types
import unittest

import torch

# `import minference` needs vllm, and kivi.py/retr_attn.py need kivi_gemv/papyfaiss
# (guarded by a check that misfires truthy on current transformers), none of which
# DynamicCacheWithRepeat touches. Load the module under test directly instead.
for _name in ("kivi_gemv", "papyfaiss"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _stub

_minference_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minference"
)
if "minference" not in sys.modules:
    _pkg = types.ModuleType("minference")
    _pkg.__path__ = [_minference_dir]
    sys.modules["minference"] = _pkg
if "minference.modules" not in sys.modules:
    _modules_pkg = types.ModuleType("minference.modules")
    _modules_pkg.__path__ = [os.path.join(_minference_dir, "modules")]
    sys.modules["minference.modules"] = _modules_pkg

DynamicCacheWithRepeat = importlib.import_module(
    "minference.modules.kvcompression"
).DynamicCacheWithRepeat


class DynamicCacheWithRepeatTest(unittest.TestCase):
    def test_empty_cache_seq_length_is_zero(self):
        cache = DynamicCacheWithRepeat(config=None)
        self.assertEqual(cache.get_seq_length(), 0)

    def test_prefill_then_decode_updates_seq_length(self):
        cache = DynamicCacheWithRepeat(config=None)
        key_states = torch.randn(1, 2, 5, 4)
        value_states = torch.randn(1, 2, 5, 4)
        out_k, out_v = cache.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs={"update_global_past_kv": True},
        )
        torch.testing.assert_close(out_k, key_states)
        torch.testing.assert_close(out_v, value_states)
        self.assertEqual(cache.get_seq_length(), 5)

        # decoding step: one more token appended to layer 0
        key_states2 = torch.randn(1, 2, 1, 4)
        value_states2 = torch.randn(1, 2, 1, 4)
        cache.update(
            key_states2,
            value_states2,
            layer_idx=0,
            cache_kwargs={"update_global_past_kv": True},
        )
        self.assertEqual(cache.get_seq_length(), 6)


if __name__ == "__main__":
    unittest.main()
