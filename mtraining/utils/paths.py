import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ATTN_CONFIG_DIR = os.path.join(BASE_DIR, 'train_attn_configs')
ACTIVE_PARAM_CONFIG_DIR = os.path.join(BASE_DIR, "models", "active_param_configs")


EXPR_DATA_SAVE_PATH = {
    'base_path': None,
    "ckpt_save_path": None,
    "compile_save_path": None,
}

def update_expr_data_by_base_path(base_path):
    EXPR_DATA_SAVE_PATH['base_path'] = base_path

def update_expr_data_save_path(
        ckpt_save_path, 
        compile_save_path, 
    ):
    if ckpt_save_path is None:
        EXPR_DATA_SAVE_PATH['base_path'] = os.getenv("EFFI_EXPR_STORE_DIR")
    else:
        EXPR_DATA_SAVE_PATH['base_path'] = os.path.dirname(ckpt_save_path)
        if "rank_" in ckpt_save_path:
            EXPR_DATA_SAVE_PATH['base_path'] = os.path.dirname(os.path.dirname(ckpt_save_path))

    EXPR_DATA_SAVE_PATH['ckpt_save_path'] = ckpt_save_path
    EXPR_DATA_SAVE_PATH['compile_save_path'] = compile_save_path