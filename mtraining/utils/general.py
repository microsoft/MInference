import os
import torch
import torch.distributed as dist

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnscaler.cli.trainer_args import AggregatedOutputs
from nnscaler.runtime.module import ParallelModule
from .paths import ACTIVE_PARAM_CONFIG_DIR, BASE_DIR

import logging
logger = logging.getLogger(__name__)

def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = default_pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = default_eos_token
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = default_bos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = default_unk_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer

def get_module_path(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    module_path = str(model.__class__.__module__)
    del model

    return module_path

def aggregate_outputs_fn(loss_outputs, sync_group) -> AggregatedOutputs:
    losses, ntokens_info = [], []
    for _, loss, ntokens, _ in loss_outputs:
        losses.append(loss)
        ntokens_info.append(ntokens)

    
    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    dist.all_reduce(loss_sum, group=sync_group)

    ntokens_sum = torch.sum(torch.tensor(ntokens_info, dtype=torch.float64, device=torch.cuda.current_device()))
    dist.all_reduce(ntokens_sum, group=sync_group)
    
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    dist.all_reduce(num_batches, group=sync_group)

    return AggregatedOutputs(
        loss_sum=loss_sum.item() / ntokens_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=ntokens_sum.item(),
    )


def load_comm_profile_data(args):
    if args.plan_ngpus in [2, 4, 8, 16]:
        logger.info(f"Use nnscaler's built-in communication profiling data for {args.plan_ngpus} GPUs")
        return

    from nnscaler.autodist.util import get_default_profile_path
    profile_dir = os.path.join(get_default_profile_path(), 'comm')
    profile_path = os.path.join(profile_dir, f"intra_{args.plan_ngpus}.json")

    if not os.path.exists(profile_path):
        import shutil
        logger.info(f"Communication profiling data not found in {profile_dir} for {args.plan_ngpus} GPUs. Use built-in communication profiling data (collected on A100-SXM4-40GB)")
        src_file_path = os.path.join(BASE_DIR, "utils/comm_prof/NVIDIA_A100-SXM4-40GB", f"intra_{args.plan_ngpus}.json")
        if not os.path.exists(src_file_path):
            raise FileNotFoundError(f"Communication profiling data not found in {src_file_path} nor in nnscaler's built-in library for {args.plan_ngpus} GPUs")
        os.makedirs(profile_dir, exist_ok=True)

        num_dev = 2
        while num_dev <= args.plan_ngpus:
            src_file_path = os.path.join(BASE_DIR, "utils/comm_prof/NVIDIA_A100-SXM4-40GB", f"intra_{num_dev}.json")
            profile_path = os.path.join(profile_dir, f"intra_{num_dev}.json")
            if os.path.exists(profile_path):
                logger.info(f"Communication profiling data already exists in {profile_path} for {num_dev} GPUs")
                num_dev *= 2
                continue
            else:
                logger.info(f"Copying {src_file_path} to {profile_path}")
                shutil.copy(src_file_path, profile_path)
                num_dev *= 2
                


def is_active(module_name: str, keep_active: List[str]):
    for active_module_subname in keep_active:
        if active_module_subname.lower() in module_name.lower():
            return True
    return False

def read_active_param_list(active_param_config_name: str):
    print(f"Reading active param list from {active_param_config_name}...")
    with open(os.path.join(ACTIVE_PARAM_CONFIG_DIR, f'{active_param_config_name}.txt'), "r") as f:
        return f.read().splitlines()

def freeze_model_params_(model, keep_active: List[str], prefix=""):
    if dist.get_rank() == 0:
        print("-" * 80)
        print(f"Only keeping parameters with substring in {keep_active} active...")

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_model_params_(module, keep_active, prefix + name + ".")
        else:
            param_name = prefix + name
            if not is_active(param_name, keep_active):
                print(f"Freezing {param_name}...")
                for param in module.parameters():
                    param.requires_grad = False
            else:
                print(f"Keeping {param_name} active...")

    if dist.get_rank() == 0:
        print("-" * 80)


def freeze_model_params(model, active_param_config_name: str, prefix=""):
    print(f"active param config name: {active_param_config_name}")
    keep_active = read_active_param_list(active_param_config_name)
    print(f"keep active: {keep_active}")

    freeze_model_params_(model, keep_active, prefix)
