#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import os
import yaml
import torch
import shutil
import argparse
import numpy as np
import torch.distributed as dist

from typing import Dict, List, Optional
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel

from nnscaler.utils import set_default_logger_level
from nnscaler.cli.trainer_args import (
    CheckpointConfig,
    DatasetConfig,
    HookMapConfig,
    ModelConfig,
    OptimizerConfig,
    DataloaderConfig,
    LogConfig,
    DatasetSamplerConfig,
)
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger

from .trainer import CustomTrainer as Trainer, CustomTrainerArgs as TrainerArgs
from .utils import chunk_linear_cross_entropy, get_tokenizer, aggregate_outputs_fn, get_resume_path
from MTraining.ops.minfer import ExprMInferConfig as MInferenceConfig, ExprMInference as MInference
from MTraining.ops import AttnType, overwrite_attn_implementation, load_moba_config, MoBAConfig
from MTraining.models import MODEL_TO_ATTN_FUNC, MODEL_ID_TO_MODEL_CLS, MODEL_ID_TO_PREFIX
from .utils.paths import (
    MINFER_CONFIG_DIR, SPARSE_PATTERN_CONFIG_DIR, SPARSE_HEAD_MAP_DIR, 
    update_expr_data_save_path
)
from MTraining.utils.train_utils import freeze_model_params, load_comm_profile_data
from MTraining.utils.expr_data import update_expr_data

import logging
logger = logging.getLogger(__name__)
set_default_logger_level('INFO')

IGNORE_IDX = -100


def init_by_attn_type(model_id: str, attn_type: AttnType):
    attn_dict = MODEL_TO_ATTN_FUNC[model_id]

    if attn_type == AttnType.BASELINE:
        print(f"{__name__} | Using Baseline Model...")
    elif attn_type == AttnType.FLEX_PREFILL:
        print(f"{__name__} | Using FlexPrefill-equipped Model ...")
    elif attn_type == AttnType.RING_ATTN:
        print(f"{__name__} | Using Ring Attention Zigzag-equipped Model ...")
    elif attn_type == AttnType.RING_ATTN_STRIPE:
        print(f"{__name__} | Using Ring Attention Stripe-equipped Model ...")
    elif attn_type == AttnType.MF_MB:
        print(f"{__name__} | Using MInference-equipped Model ...")
    elif attn_type == AttnType.MOBA:
        print(f"{__name__} | Using MoBA-equipped Model ...")
    elif attn_type == AttnType.ZIGZAG_MOBA:
        print(f"{__name__} | Using ZigZag MoBA-equipped Model ...")
    elif attn_type == AttnType.XATTN:
        print(f"{__name__} | Using XAttention-equipped Model ...")
    else:
        raise ValueError(f"Invalid attn_type: {attn_type}")

    overwrite_attn_implementation(attn_dict, attn_type)


class BaselineModel(torch.nn.Module):
    def __init__(
            self, 
            model_id, 
            config_path: str=None, 
            # merged_ckpt_path: str=None,
            active_param_config_name: str=None
    ):
        super().__init__()
        model_cls: PreTrainedModel = MODEL_ID_TO_MODEL_CLS[model_id]

        if not config_path:
            self.model = model_cls.from_pretrained(
                model_id,
                attn_implementation='flash_attention_2'
            )
        else:
            model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            model_config._attn_implementation = 'flash_attention_2'
            self.model = model_cls.from_pretrained(
                model_id,
                config=model_config,
            )

        if active_param_config_name:
            freeze_model_params(self.model, active_param_config_name)

        print(f'{__class__.__name__} Self-Attention Class: {self.model.model.layers[0].self_attn.__class__.__name__}')

    def forward(self, samples):
        with torch.autocast(device_type="cuda", dtype=self.model.config.torch_dtype):
            outputs = self.model.model(
                input_ids=samples['net_input']['src_tokens'],
                use_cache=False,
                return_dict=False,
            )
            hidden_states = outputs[0]
            losses = chunk_linear_cross_entropy(hidden_states, self.model.lm_head.weight, samples['target'], IGNORE_IDX, 1024)
            loss = torch.sum(losses)

        return loss, loss.data, samples['ntokens'], samples['nsentences']

class MInferModel(BaselineModel):
    def __init__(
            self, 
            model_id, 
            config_path: str=None, 
            minfer_config: Dict={},
            **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            config_path=config_path,
            **kwargs,
        )

        # --------------------------------------------
        # MInference implementation: "fa", "stripe"
        minfer_implementation: str = minfer_config.pop('implementation', 'fa')

        # --------------------------------------------
        # Sparse iteratio, layer and head control
        start_sparse_iter: int = minfer_config.pop('start_sparse_iter', 0)
        start_sparse_layer: int = minfer_config.pop('start_sparse_layer', 0)
        adaptive_sparse: bool = minfer_config.pop('adaptive_sparse', False)
        sparse_head_map_name: str = minfer_config.pop('sparse_head_map_name', None)
        if sparse_head_map_name is not None:
            active_sparse_map_path: str = os.path.join(
                SPARSE_HEAD_MAP_DIR,
                f'{sparse_head_map_name}.npy',
            )
            print(f"{__name__} | Active Sparse Head Map Path: {active_sparse_map_path}")
            active_sparse_map: np.ndarray = np.load(active_sparse_map_path)
            active_sparse_map: List[List[bool]] = active_sparse_map.tolist()
        else:
            active_sparse_map: List[List[bool]] = None
        
        # ----------------------------------------------
        # Ring Attention specific
        granularity: int = minfer_config.pop('granularity', 128)

        # --------------------------------------------
        # Standard MInference Setup
        minfer_attn_type = minfer_config.pop('attn_type', 'minference')
        minfer_config['config_path'] = os.path.join( 
            SPARSE_PATTERN_CONFIG_DIR,
            f'{minfer_config.pop("pattern_config_name")}.json',
        )
        print(f"{__name__} | MInference Pattern Config Path: {minfer_config['config_path']}")
        minfer = MInference(
            attn_type=minfer_attn_type,
            model_name=model_id,
            **minfer_config,
        )
        minfer_config: MInferenceConfig = minfer.config
        
        # --------------------------------------------
        # We still need to attach the function object to the model
        # otherwise the states of the function will be lost as nnscaler will only load the model from file
        # but not call this procedure again
        from ops.minfer_func import MInferAttnFunc
        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.minfer_attn_func = MInferAttnFunc()
                m.minfer_attn_func.init_minfer_params(
                    config_path=minfer_config.config_path,
                    minfer_implementation=minfer_implementation,
                    
                    start_sparse_iter=start_sparse_iter,
                    start_sparse_layer=start_sparse_layer,
                    adaptive_sparse=adaptive_sparse,
                    active_sparse_map=active_sparse_map,

                    granularity=granularity,
                )
        self.model.apply(update_module)

class FlexPrefillModel(BaselineModel):
    def __init__(
            self, 
            model_id, 
            config_path: str=None, 
            attn_config: Dict={},
            **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            config_path=config_path,
            **kwargs,
        )

        from ops.flex_prefill_func import FlexPrefillFunc
        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.flex_prefill_attn_func = FlexPrefillFunc(attn_config)
        self.model.apply(update_module)

class XAttnModel(BaselineModel):
    def __init__(
            self, 
            model_id, 
            config_path: str=None, 
            xattn_params: Dict={},
            **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            config_path=config_path,
            **kwargs,
        )

        # --------------------------------------------
        implementation: str = xattn_params.pop('implementation', 'fa')
        granularity: int = xattn_params.pop('granularity', 128)

        # --------------------------------------------
        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.granularity = granularity
                m.xattn_params = xattn_params
                m.implementation = implementation
        self.model.apply(update_module)


class MoBAModel(BaselineModel):
    def __init__(
            self, 
            model_id, 
            config_path: str=None, 
            moba_config_dict: Dict={},
            **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            config_path=config_path,
            **kwargs,
        )

        # --------------------------------------------
        print(f"MoBAConfig: {moba_config_dict}")
        moba_config = MoBAConfig(**moba_config_dict) 
        moba_topk, moba_chunk_size = moba_config.moba_topk, moba_config.moba_chunk_size   

        # --------------------------------------------
        # We still need to attach the function object to the model
        # otherwise the states of the function will be lost as nnscaler will only load the model from file
        # but not call this procedure again
        Attention = self.model.model.layers[0].self_attn.__class__
        def update_module(m):
            if isinstance(m, Attention):
                m.moba_topk = moba_topk
                m.moba_chunk_size = moba_chunk_size
        self.model.apply(update_module)

ATTN_TO_MODEL = {
    AttnType.BASELINE: BaselineModel,
    AttnType.FLEX_PREFILL: FlexPrefillModel,
    AttnType.MF_MB: MInferModel,
    AttnType.RING_ATTN: BaselineModel,
    AttnType.RING_ATTN_STRIPE: BaselineModel,
    AttnType.MOBA: MoBAModel,
    AttnType.ZIGZAG_MOBA: MoBAModel,
    AttnType.XATTN: XAttnModel,
}


def load_minfer_config(minfer_config_name: str) -> MInferenceConfig:
    minfer_config_path = os.path.join(MINFER_CONFIG_DIR, f'{minfer_config_name}.yaml')
    if not os.path.exists(minfer_config_path):
        print(f"{__name__} | MInference config {minfer_config_name} not found in {minfer_config_path}. Use empty minfer config")
        minfer_config = {}
    else:
        print(f"{__name__} | MInference config {minfer_config_name} found in {minfer_config_path}")
        with open(minfer_config_path, 'r') as f:
            minfer_config = yaml.safe_load(f)
        print('-' * 20)
        print("MInference Config:")
        print(minfer_config)
        print('-' * 20)

    return minfer_config

def build_model_args(args, minfer_config: MInferenceConfig) -> Dict:
    model_args = {
        'model_id': args.model_id,
        'config_path': args.model_config_path,
        "active_param_config_name": args.active_param_config_name,
    }
    if args.attn_type == AttnType.MF_MB: 
        model_args['minfer_config'] = minfer_config
    elif args.attn_type == AttnType.FLEX_PREFILL:
        model_args['attn_config'] = minfer_config
    elif args.attn_type == AttnType.XATTN:
        model_args['xattn_params'] = minfer_config
    elif args.attn_type == AttnType.MOBA or args.attn_type == AttnType.ZIGZAG_MOBA:
        model_args['moba_config_dict'] = minfer_config

    return model_args


def main(args):
    update_expr_data_save_path(args.attn_save_path, args.ckpt_save_dir, args.compile_save_path)
    update_expr_data(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        load_comm_profile_data(args)

    init_by_attn_type(args.model_id, args.attn_type)
    minfer_config = load_minfer_config(args.minfer_config_name)

    # broadcast_strategy = 'all' if args.run_mode == 'run' else 'none'
    broadcast_strategy = 'all'
    # ---------------------------------
    # Compute config
    if args.run_mode == 'compile':
        if args.runtime_ngpus is None:
            raise ValueError('runtime_ngpus must be specified in compile mode')
        runtime_ngpus = args.runtime_ngpus
    elif args.run_mode == 'run':
        world_size = int(os.getenv('WORLD_SIZE'))
        if args.runtime_ngpus is None:
            runtime_ngpus = world_size
        else:
            if args.runtime_ngpus != world_size:
                raise ValueError(f'runtime_ngpus ({args.runtime_ngpus}) must match the number of GPUs in run mode ({world_size})')
            runtime_ngpus = args.runtime_ngpus

    if runtime_ngpus % args.plan_ngpus != 0:
        raise ValueError('runtime_ngpus must be a multiple of plan_ngpus')

    scaling_factor: int = runtime_ngpus // args.plan_ngpus
    grad_accu_step: int = args.global_batch_size // (args.micro_batch_size * scaling_factor)
    
    model_prefix = MODEL_ID_TO_PREFIX[args.model_id]
    pas_config = {
        'recompute_modules': f'{model_prefix}DecoderLayer',
    }
    if args.mem_constraint > 0: pas_config['mem_constraint'] = args.mem_constraint
    compute_config = ComputeConfig(
        plan_ngpus=args.plan_ngpus,
        trace_strategy=args.trace_strategy,
        runtime_ngpus=runtime_ngpus,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        # autodist config:
        pas_config=pas_config,
    )

    # ---------------------------------
    ## Setup Dataset ##
    dataset = load_from_disk(args.dataset_path)
    tokenizer = get_tokenizer(args.model_id)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        mini_batch = data_collator(samples)
        _mini_batch = {}

        src_tokens = mini_batch.pop('input_ids')
        seq_len = src_tokens.size(-1)
        _mini_batch['src_tokens'] = src_tokens

        shift_labels = mini_batch['labels'][..., 1:]
        _mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', IGNORE_IDX).contiguous()

        return {
            "nsentences": len(samples),
            "ntokens": len(samples) * seq_len,
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }
    dataset_config = DatasetConfig(
        type=(lambda split: dataset),
        train_args={'split': 'train'},
    )
    dataloader_config = DataloaderConfig(
        train_args={
            'collate_fn': collate_fn,
            'drop_last': True,
        },
    )
    sampler_config = DatasetSamplerConfig(
        # default class: torch.utils.data.distributed.DistributedSampler
        train_args={
            'shuffle': True,
            'seed': args.seed,
        },
    )
    
    # ---------------------------------
    # Model Config
    model_args = build_model_args(args, minfer_config)
    model_config = ModelConfig(
        type=ATTN_TO_MODEL[args.attn_type],
        args=model_args,
    )

    # ---------------------------------
    # optimizer hyperparameters are from YaRN
    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={
            'lr': 2e-5, 
            'betas': (0.9, 0.95), 
            'weight_decay': 0.0, 
            'fused': True
        },
        clip_gnorm=1.0,
        loss_reduction='sum',
        grad_reduction='per-token-mean',
        aggregate_outputs_fn=aggregate_outputs_fn,
    )

    
    # ---------------------------------
    # Checkpoint Config
    checkpoint_config = CheckpointConfig(
        save_dir=args.ckpt_save_dir if args.ckpt_save_dir else f'./checkpoints_{args.name}',
        every_n_epochs=args.ckpt_n_epoch,
        every_n_train_steps=args.ckpt_n_step,
        save_type='deduped',
        # resume_from=(args.resume_from or 'last') if args.check_resume else None,
        resume_from=args.resume_from,
    )

    # ---------------------------------
    # Log Config
    log_config = LogConfig(
        type=TensorBoardLogger,
        args={
            'name': args.name,
            'root_dir': args.tf_log_dir or f'./runs_{args.name}',
        },
    )

    # ---------------------------------
    trainer_args = TrainerArgs(
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        grad_accumulation_steps=grad_accu_step,

        pas_policy='autodist',
        precision='bf16',
        seed=args.seed,
        gen_reuse=args.reuse_type,

        gen_savedir=args.compile_save_path,
        instance_name=args.name,
        run_mode=args.run_mode,
        max_epochs=args.n_epochs,
        max_train_steps=args.n_iter,
        enable_progress_bar=not args.disable_progressbar,
        
        compute_config=compute_config,
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        checkpoint=checkpoint_config,
        log=[log_config],
        
        broadcast_strategy=broadcast_strategy,
        dataset_sampler=sampler_config,   

        transfer_config={
            "transfer_config_dir": args.transfer_config_dir,
            "transfer_force": args.transfer_force,
        },
        merged_ckpt_path=args.resume_merged_ckpt,
    )

    trainer = Trainer(
        train_args=trainer_args,
        save_data_steps=args.attn_save_step,
        enable_prof=args.enable_prof,
    )
    trainer.run()

def print_args(args: argparse.Namespace):
    print("=" * 80)
    print(f"Start Experiment:\t{args.name}")
    print(f"Seed:\t{args.seed}")
    print(f"Reuse Type:\t{args.reuse_type}")
    print(f"Run Mode:\t{args.run_mode}")
    print(f"Total number of GPUs:\t{args.runtime_ngpus}")
    print(f"GPU unit size:\t{args.plan_ngpus}")
    print(f"Model ID:\t{args.model_id}")

    print('-' * 40)
    if args.n_iter:
        print(f"Number of Iterations:\t{args.n_iter} (number of tokens: {args.n_iter * args.global_batch_size * args.seq_len})")
    else:
        print(f"Number of Epochs:\t{args.n_epochs}")


    print(f'Global Batch Size:\t{args.global_batch_size}')
    print(f'Micro Batch Size:\t{args.micro_batch_size}')

    scaling_factor = args.runtime_ngpus // args.plan_ngpus
    grad_accu_step = args.global_batch_size // (args.micro_batch_size * scaling_factor)
    print(f"Scaling Factor (INFERRED):\t{scaling_factor}")
    print(f"Gradient Accumulation Steps (INFERRED):\t{grad_accu_step}")
    print(f"Save Attention Data Every {args.attn_save_step} Steps")

    print('-' * 40)
    print(f"Model Config Path:\t{args.model_config_path}")
    print(f"Dataset path:\t{args.dataset_path}")
    print(f'MInferenece Config Name:\t{args.minfer_config_name}')
    print(f"Compile Save Path:\t{args.compile_save_path}")
    print(f"Attention Save Path:\t{args.attn_save_path}")
    print(f"Tensorboard Log Path:\t{args.tf_log_dir}")
    print(f"Checkpoint Save Path:\t{args.ckpt_save_dir}")
    print(f"Resume from Checkpoint:\t{args.check_resume}")
    print(f"Path to the checkpoint to resume from:\t{args.resume_from}")
    print(f"Path to the merged checkpoint to resume from:\t{args.resume_merged_ckpt}")  

    print(f"Enable profiling: {args.enable_prof}")
    print(f"Trace Strategy:\t{args.trace_strategy}")
    if args.transfer_config_dir:
        print(f"Transfer Configs from another experiment:\t{args.transfer_config_dir}")
        print(f"Force Transfer Configs:\t{args.transfer_force}")
    
    if args.active_param_config_name:
        print(f"Active Param Config Name:\t{args.active_param_config_name}")

    if args.ckpt_n_step:
        print(f"Checkpoint Save Every {args.ckpt_n_step} Steps")
    else:
        print(f"Checkpoint Save Every {args.ckpt_n_epoch} Epochs")
    print("=" * 80, flush=True)

if __name__ == '__main__':
    ## Parse Args ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--name', type=str, default='phi-grad', help='name of the experiment')
    parser.add_argument('--seq_len', type=int, default=131072, help='sequence length')
    parser.add_argument('--attn_type', type=str, default=AttnType.BASELINE, choices=AttnType.__dict__.values(), help='minference type')
    parser.add_argument('--reuse_type', type=str, default='match', choices=['match', 'override', 'moo', 'graph'], help='reuse type')
    parser.add_argument('--run_mode', type=str, default='run', choices=['run', 'compile'], help='run or compile')
    parser.add_argument('--trace_strategy', type=str, default='cuda_run_cpu_offload', 
                        choices=['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache'], 
                        help='trace strategy')
    parser.add_argument('--plan_ngpus', type=int, required=True, help='specify the scale unit size')
    parser.add_argument('--runtime_ngpus', type=int, required=True, help='specify the number of GPUs to use')
    
    parser.add_argument('--n_iter', type=int, default=0, help='Number of iterations')
    parser.add_argument('--n_epochs', type=int, default=0, help='Number of epochs')
    parser.add_argument('--nB_tokens', type=int, default=0, help='Number of tokens (in B) to process')
    parser.add_argument('--global_batch_size', type=int, default=4, help='global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=1, help='micro batch size')
    parser.add_argument('--mem_constraint', type=int, default=0, help='memory constraint')

    parser.add_argument('--model_id', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='transformers model id')
    parser.add_argument('--model_config_path', type=str, default=None, help='path to the model config')
    parser.add_argument('-s', '--attn_save_step', type=int, default=1, help='Save attention data every n steps')

    parser.add_argument('--minfer_config_name', type=str, default=None, help='Name of Minference config file')
    parser.add_argument('--compile_save_path', type=str, default='./.nnscaler', help='path to save compiled code')
    parser.add_argument('--attn_save_path', type=str, default=None, help='path to save attention data')
    parser.add_argument('--tf_log_dir', type=str, default=None, help='path to save tensorboard logs')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to the dataset')
    parser.add_argument('--check_resume', action='store_true', help='whether to resume from checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the checkpoint to resume from')
    parser.add_argument('--resume_merged_ckpt', type=str, default=None, help='path (dir) to the merged checkpoint to resume from')

    parser.add_argument('--enable_prof', action='store_true', help='enable profiling')

    parser.add_argument('--ckpt_save_dir', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--ckpt_n_epoch', type=int, default=1, help='save checkpoint every n epochs')
    parser.add_argument('--ckpt_n_step', type=int, default=0, help='save checkpoint every n steps')
    parser.add_argument('--transfer_config_dir', type=str, default="none", help='path to transfer configs from another experiment')
    parser.add_argument('--transfer_force', action='store_true', help='force transfer configs')
    parser.add_argument('--active_param_config_name', type=str, default=None, help='path to the active param list')

    parser.add_argument('-p', '--disable_progressbar', action='store_true', help='transformers model id',)

    args = parser.parse_args()
    
    # -------------------------------------------------
    # Preprocessing args
    if args.ckpt_n_epoch <= 0: args.ckpt_n_epoch = None
    if args.ckpt_n_step <= 0: args.ckpt_n_step = None

    if args.nB_tokens > 0:
        args.n_iter = args.nB_tokens * 1e9 // args.global_batch_size // args.seq_len + 1
        args.n_epochs = 0
    if args.n_iter <= 0: args.n_iter = None
    if args.n_epochs <= 0: args.n_epochs = None

    if args.minfer_config_name is None or args.minfer_config_name.lower() == 'none': args.minfer_config_name = None
    if args.transfer_config_dir.lower() == 'none': args.transfer_config_dir = None
    if args.active_param_config_name.lower() == 'none': args.active_param_config_name = None

    # set a new field of args 'args.orig_resume_from' to store the original resume_from value
    args.orig_resume_from = args.resume_from
    args.resume_from = get_resume_path(
        args.check_resume, args.resume_from, args.ckpt_save_dir, args.runtime_ngpus
    )


    print_args(args)
    main(args)