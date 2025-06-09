import os
import json


CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
DEFAULT_CONFIG_FILE = "Qwen2.5_3B_kv_out_v32_fit_o_best_pattern.json"
if "MI_CONFIG" in os.environ:
    DEFAULT_CONFIG_FILE = os.environ["MI_CONFIG"]


def get_minference_config(config_file: str = DEFAULT_CONFIG_FILE):
    with open(os.path.join(CONFIG_DIR, config_file)) as f:
        data = json.loads(f.read())
    config = []
    for layer_data in data:
        v_size_list = [None] * len(layer_data)
        s_size_list = [None] * len(layer_data)
        for k, v in layer_data.items():
            assert v[0] in ['vertical_and_slash', 'flex_vertical_and_slash']
            v_size_list[int(k)] = v[1]
            s_size_list[int(k)] = v[2]
        config.append([v_size_list, s_size_list])
    return config
