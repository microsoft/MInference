import os
import time
import copy
import torch
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from datetime import timedelta
from collections import defaultdict
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Optional, Callable

import torch.distributed
from torch.profiler import profile, ProfilerActivity

import nnscaler
from nnscaler.utils import accum_mode
from nnscaler.runtime.utils import microbatches
from nnscaler.runtime.module import ParallelModule
from nnscaler.utils import is_running_distributed
from nnscaler.cli.trainer import (
    Trainer, _StepStat, TrainerArgs, TrainStatus, AggregatedTrainHook, TrainHook
)

from .utils.paths import EXPR_DATA_SAVE_PATH
from .utils.general import fix_model_state_dict
from .utils.custom_parallel import parallelize as custom_parallelize

logger = logging.getLogger(__name__)

@dataclass
class CustomTrainerArgs(TrainerArgs):
    transfer_config: Optional[Dict[str, Any]] = None
    merged_ckpt_path: Optional[str] = None

ITERATOR_COUNTER = defaultdict(int)
def get_iter_cnt(rank: int):
    global ITERATOR_COUNTER
    return ITERATOR_COUNTER.get(rank, 0)

ITER_BATCH_IDX_DICT = {}
def get_iter_batch_idx(rank: int, iter_cnt: int):
    global ITER_BATCH_IDX_DICT
    return ITER_BATCH_IDX_DICT.get(rank, {}).get(iter_cnt, 0)

def custom_train_step(
    model: ParallelModule,
    rank: int, iter_idx: int,
    samples: List[Any],
    is_dummy_batch: Optional[List[bool]] = None,
    scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[Any]:
        """
        The training step function. It should be called in the training loop.
        Please note:
            1. This function is only supported in end2end mode.
            2. Gradient accumulation is done inside this function.
                You shouldn't do gradient accumulation outside this function,
                because the gradients will be cleared in the beginning of this function
        Args:
            samples (List[Any]): a list of samples.
                if pipeline is used, it must have the same length as configured to pas policy
            is_dummy_batch (Optional[List[bool]]): indicates whether the each micro-batch is dummya
            scale_fn (Optional[Callable[[torch.Tensor], torch.Tensor]]): the function to scale the loss
        Results:
            List[Any]: a list of outputs for each sample
        """
        global ITER_BATCH_IDX_DICT
        model._warn_uninitialized_non_persistent_buffers(raise_error=True)

        if not model.compute_config.use_end2end:
            raise RuntimeError("train_step() is only supported in end2end mode")
        if is_dummy_batch and len(samples) != len(is_dummy_batch):
            raise ValueError("The length of samples and is_dummy_batch should be the same")

        model._scale_loss(is_dummy_batch, scale_fn)

        # sync_grad will be done in _train_step
        # so we never need to call it manually
        model._sync_grad_required = False
        sample_count = len(samples)
        dataloader = microbatches(samples, cycle=False)

        if model.use_scheduler:
            if len(samples) != model.nmicros_per_scheduler_step:
                raise ValueError(f"Expected {model.nmicros_per_scheduler_step} samples, but got {sample_count}")
            # only one step, so begin/end are both True
            with accum_mode(begin=True, end=True):
                return model._train_step(dataloader), None
        else:
            outputs = []
            latencies = []
            for idx in range(sample_count):
                ITER_BATCH_IDX_DICT[rank][iter_idx] = idx

                sample_start_time = time.perf_counter()
                with accum_mode(begin=(idx==0), end=(idx==sample_count-1)):
                    # loss, loss.data, samples['ntokens'], samples['nsentences']
                    output = model._train_step(dataloader)
                sample_time = time.perf_counter() - sample_start_time
                latencies.append(sample_time)
    
                num_tokens = output[2] 
                if rank == 0:
                    print(
                        f"| {__name__} | rank={rank} | iter_idx={iter_idx}, batch_idx={idx}, loss={output[1] / num_tokens:.4f},"
                        f" num_tokens={num_tokens}, latency={sample_time:.4f}s"
                    )

                outputs.append(output)
            return outputs, latencies

class CustomTrainer(Trainer):
    def __init__(
        self,
        argv: Optional[List[str]] = None,
        *,
        train_args: Optional[Union[Dict[str, Any], CustomTrainerArgs]] = None,
    ):
        """
        Custom trainer with an additional parameter.

        Args:
            argv (Optional[List[str]]): Command line arguments. If not specified, sys.argv[1:] will be used.
            train_args: A dict used to construct TrainerArgs or a TrainerArgs object itself.
            additional_param (Optional[Any]): Additional parameter for custom functionality.
        """
        # Call the parent class's initializer with the existing parameters
        super().__init__(argv=argv, train_args=train_args)
        self.train_args: CustomTrainerArgs
        
        torch.distributed.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=2),
        )
        self.train_step_func = custom_train_step

    def _train_epoch(self, epoch):
        VAL_STATUS_NO = 0     # not validated or saved
        VAL_STATUS_VAL = 1    # validated but not saved
        VAL_STATUS_SAVE = 2   # validated and saved
        has_validated = VAL_STATUS_NO   # 3 states

        resume_from_idx = self.train_status.finished_train_steps % self.total_train_steps_per_epoch
        data_iter = enumerate(self._global_batch_iterator(num_skip_first=resume_from_idx))

        max_epoch = self.max_train_steps // self.total_train_steps_per_epoch
        if self.max_train_steps % self.total_train_steps_per_epoch != 0:
            max_epoch += 1
        ndigits = len(str(max_epoch))
        epoch_format = f"0{ndigits}d"
        epoch_desc = f'Epoch {format(epoch, epoch_format)}'

        if self.rank == 0:
            progress = tqdm(
                None,
                total=self.total_train_steps_per_epoch,
                initial=resume_from_idx,
                desc=epoch_desc,
                disable=not self.train_args.enable_progress_bar,
            )
        else:
            progress = None


        # ---------------------------------------------------------------------------------
        train_info_save_path = os.path.join(EXPR_DATA_SAVE_PATH['base_path'], 'train_info', f"epoch_{epoch}.log")
        os.makedirs(os.path.dirname(train_info_save_path), exist_ok=True)
        if self.rank == 0:
            # Check whether the file already exists
            # If it exists, assume existing log file has name 'epoch_<epoch_idx>_<num>.log' ('epoch_<epoch_idx>.log` is assumed to have num 0)
            # Find the greatest <num> for the current epoch and increment it to build the new file name
            existing_files = [f for f in os.listdir(os.path.dirname(train_info_save_path)) \
                              if f.startswith(f'epoch_{epoch}_') or f.startswith(f'epoch_{epoch}.log')]
            if existing_files:
                # Extract the numbers from the filenames
                existing_nums = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(f'epoch_{epoch}_')]
                if not existing_nums:
                    existing_nums = [0]
                new_num = max(existing_nums) + 1
                train_info_save_path = os.path.join(os.path.dirname(train_info_save_path), f'epoch_{epoch}_{new_num}.log')
            else:
                # If no existing files, use the original path
                train_info_save_path = os.path.join(os.path.dirname(train_info_save_path), f'epoch_{epoch}.log')
            with open(train_info_save_path, 'w') as f: f.write('')

        step_stat: Optional[_StepStat] = None
        num_tokens_trained = 0
        for i, batches in data_iter:
            idx = i + resume_from_idx
            
            global ITERATOR_COUNTER, ITER_BATCH_IDX_DICT
            ITERATOR_COUNTER[self.rank] = idx
            ITER_BATCH_IDX_DICT[self.rank] = {idx: 0}
            if self.rank == 0:
                progress.update(1)
            step_start_at = time.perf_counter()
            step_stat = _StepStat()
            step_metrics = {}
            has_validated = VAL_STATUS_NO
            num_batches = len(batches)
            batches, is_dummy_batch = self._fix_batches(batches)

            self.model.train()

            self.hook.before_zero_grad(self)
            self.optimizer.zero_grad()
            self.hook.after_zero_grad(self)

            self.hook.on_train_step_start(self, batches[:num_batches], idx)
            losses, latencies = self.train_step_func(self.model, self.rank, idx, batches, is_dummy_batch)  
            self.hook.on_train_step_end(self, losses[:num_batches], batches[:num_batches], idx)

            aggregate_outputs = self.train_args.resolved_aggregate_outputs_fn or self.aggregate_outputs
            aggregated_outputs = aggregate_outputs(losses[:num_batches], self.sync_group)
            if self.train_args.optimizer.loss_reduction == 'mean':
                loss = aggregated_outputs.loss_sum / aggregated_outputs.num_batches
            else:
                loss = aggregated_outputs.loss_sum
            step_stat.train_loss = loss
            num_tokens_trained += aggregated_outputs.num_tokens
            self.hook.after_aggregate_train_step_outputs(self, aggregated_outputs, loss, idx)

            self.hook.before_sync_grad(self)
            self.optimizer.sync_shard_grad()
            self.hook.after_sync_grad(self)

            # scale gradients
            multiplier = self.train_args.scaling_factor
            if self.train_args.optimizer.grad_reduction == 'sum':
                # do nothing. `multiplier` is already correct
                pass
            elif self.train_args.optimizer.grad_reduction == 'mean':
                if not aggregated_outputs.num_batches:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_batches` field")
                multiplier /= aggregated_outputs.num_batches
            else:
                assert self.train_args.optimizer.grad_reduction == 'per-token-mean'
                if not aggregated_outputs.num_tokens:
                    raise RuntimeError("`aggregate_outputs` doesn't set `num_tokens` field")
                multiplier /= aggregated_outputs.num_tokens
            self.optimizer.scale_grads(multiplier)

            # clip gradients
            self.hook.before_gnorm_clip(self)
            if self.train_args.optimizer.clip_gnorm:
                step_stat.gnorm = self.optimizer.clip_gnorm(self.train_args.optimizer.clip_gnorm)
            else:
                step_stat.gnorm = self.optimizer.clip_gnorm()
            self.hook.after_gnorm_clip(self, step_stat.gnorm)
            step_stat.gnorm = step_stat.gnorm.item()

            # update parameters
            step_stat.lr = self.optimizer.param_groups[0]['lr']
            self.hook.before_optimizer_step(self)
            self.optimizer.step()
            self.hook.after_optimizer_step(self)
            if self.lr_scheduler and self.train_args.lr_scheduler.interval == 'step':
                self.lr_scheduler.step()

            self.train_status.finished_train_steps += 1
            self._log_mem_stats(tag='train')
            step_metrics = {k:v for k, v in asdict(step_stat).items() if v is not None}
            step_metrics['train_wall'] = time.perf_counter() - step_start_at
            step_metrics['num_tokens_processed'] = num_tokens_trained
            self.log_metrics(step_metrics, tag='train')
            if self.rank == 0:
                progress.set_postfix(step_metrics)
                formatted_metrics = self._format_metrics(epoch_desc, idx + 1, step_metrics)
                with open(train_info_save_path, 'a') as f:
                        f.write(f"{formatted_metrics}\n")
    
                if self.train_args.enable_log_progress \
                    and self.train_status.finished_train_steps % self.train_args.log_progress_every_n_train_steps == 0:

                    logger.info(formatted_metrics)
                    step_metrics = {}

            # validate and save checkpoint
            if self.train_args.checkpoint.every_n_train_steps and \
                self.train_status.finished_train_steps % self.train_args.checkpoint.every_n_train_steps == 0:
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE

            # max_train_steps is reached
            if self.train_status.finished_train_steps >= self.max_train_steps:
                if step_metrics and self.train_args.enable_log_progress:
                    logger.info(self._format_metrics(epoch_desc, idx + 1, step_metrics))
                    step_metrics = {}
                if not has_validated:
                    self._validate_and_save(step_stat)
                    has_validated = VAL_STATUS_SAVE
                if self.rank == 0:
                    # disable refresh the progress bar to avoid redundant progress bar
                    progress.leave = False
                    progress.close()
                break

            if not has_validated and self.train_args.val_every_n_train_steps and \
                self.train_status.finished_train_steps % self.train_args.val_every_n_train_steps == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL

            # time.sleep(1)
        else:
            # Do per-epoch operations here.
            # if the loop exits with `break` (max_train_steps is reached)
            # those operations have done in the loop
            if step_stat is None:
                return  # no train step runs. Nothing to do.
            if has_validated < VAL_STATUS_SAVE \
                and self.train_args.checkpoint.every_n_epochs \
                and (epoch + 1) % self.train_args.checkpoint.every_n_epochs == 0:
                self._validate_and_save(step_stat)
                has_validated = VAL_STATUS_SAVE
            if not has_validated and self.train_args.val_every_n_epochs \
                and (epoch + 1) % self.train_args.val_every_n_epochs == 0:
                self._validate(step_stat)
                has_validated = VAL_STATUS_VAL
    
    def _setup(self):
        self.train_args.init_env(self)
        compile_only = self.train_args.compile_mode

        if is_running_distributed():
            nnscaler.init()
            if torch.distributed.get_rank() == 0:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)

        def _create_model():
            model = self.train_args.create_model()
            if self.train_args.param_dtype == self.train_args.buffer_dtype:
                if self.train_args.param_dtype is not None:
                    model = model.to(self.train_args.param_dtype)
            else:
                # separate param and buffer dtype
                # TODO: a little hacky. A better way?
                # 3 kinds of tensors are converted in Module._apply:
                # model parameters, its grad, and buffer
                # param_dtype controls the first two, (but grad is `None` here)
                # and buffer_dtype controls the last one
                buf_ids = { id(buf) for buf in model.buffers(recurse=True) }
                if self.train_args.param_dtype is not None:
                    model._apply(
                        lambda t: t.to(self.train_args.param_dtype)
                            if t.is_floating_point() and id(t) not in buf_ids
                            else t)
                if self.train_args.buffer_dtype is not None:
                    model._apply(
                        lambda t: t.to(self.train_args.buffer_dtype)
                            if t.is_floating_point() and id(t) in buf_ids
                            else t)
            if self.train_args.tracing_from_weights:
                model.load_state_dict(torch.load(self.train_args.tracing_from_weights))
            return model

        # create dataset and dataloader
        for stage in ['train', 'val', 'test']:
            self.dataset[stage] = self.train_args.create_dataset(stage)

        # load a dummy input from training dataset
        self.dummy_input = self._load_dummy_input()
        self.dummy_input = self._fix_input(self.dummy_input)

        for stage in ['train', 'val', 'test']:
            self.dataloader[stage] = self.train_args.create_dataloader(stage, self.dataset[stage])
            if self.dataloader[stage] is not None \
                and not self.dataloader[stage].drop_last \
                and len(self.dataset[stage]) % (self.train_args.micro_batch_size * self.train_args.scaling_factor) != 0:
                    warnings.warn(
                        f"Length of {stage} dataset ({len(self.dataset[stage])}) "
                        f"is not multiple of micro_batch_size * scale_factor ({self.train_args.micro_batch_size * self.train_args.scaling_factor}). "
                        f"In this case, the train_step for the last batch of samples can fail! "
                        f"You can specify `drop_last=True` in DataLoader to fix this problem."
                    )

        # setup compute config
        compute_config = copy.deepcopy(self.train_args.compute_config)
        compute_config.pas_config['__pas_name'] = self.train_args.pas_policy
        # autodist configs
        compute_config.pas_config['update_freq'] = self.train_args.update_freq
        compute_config.pas_config['use_bf16'] = self.train_args.param_dtype == torch.bfloat16
        compute_config.pas_config['use_fp16'] = self.train_args.param_dtype == torch.float16

        compute_config.user_config['__from_trainer_args'] = {
            'mbs': self.train_args.micro_batch_size,
            'gbs': self.train_args.global_batch_size,
            'precision': self.train_args.precision,
            'model_args': self.train_args.model.args,
        }

        # parallalize model
        pmodel_class = custom_parallelize(
            self.train_args.model_type,
            self._create_dummy_forward_args(),
            self.train_args.resolved_pas_policy,
            compute_config,
            module_fn=_create_model,
            gen_savedir=self.train_args.gen_savedir,
            reuse=self.train_args.gen_reuse,
            instance_name=self.train_args.instance_name,
            broadcast_strategy=self.train_args.broadcast_strategy,
            load_module=not compile_only,
            transfer_config=self.train_args.transfer_config,
        )
        if compile_only:
            return

        torch.distributed.barrier()
        self.rank = torch.distributed.get_rank()

        self.total_train_steps_per_epoch = len(self.dataloader['train']) // self.train_args.update_freq
        if len(self.dataloader['train']) % self.train_args.update_freq != 0:
            self.total_train_steps_per_epoch += 1  # will add extra dummy batches

        if self.train_args.max_epochs and self.train_args.max_train_steps:
            self.max_train_steps = min(
                self.total_train_steps_per_epoch * self.train_args.max_epochs,
                self.train_args.max_train_steps
            )
        elif self.train_args.max_train_steps:
            self.max_train_steps = self.train_args.max_train_steps
        else:
            assert self.train_args.max_epochs, "max_epochs or max_train_steps should be specified"
            self.max_train_steps = self.total_train_steps_per_epoch * self.train_args.max_epochs

        _, self.sync_group = self.train_args.compute_config.get_sync_group()
        self.model = pmodel_class()
        self.model.cuda()
        self.optimizer = self.train_args.create_parallel_optimizer(self.model)
        
        def reducer_pre_hook(reducer, grad):
            grad.div_(self.train_args.scaling_factor)
        self.optimizer.register_reducer_pre_hook(reducer_pre_hook)
        self.lr_scheduler = self.train_args.create_lr_scheduler(self.optimizer)
        self.loggers = self.train_args.create_loggers()

        supported_hook_components = [
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ]
        self.hook = AggregatedTrainHook(
            [x for x in supported_hook_components if isinstance(x, TrainHook)]
            + [self.train_args.create_hook()]
        )

        self._log_config(self.train_args.to_dict())
        self._load_checkpoint()

        if self.train_args.merged_ckpt_path is not None:
            print(f"Rank {self.rank} | {__name__} | loading merged checkpoint from {self.train_args.merged_ckpt_path}")
            merged_ckpt_path = os.path.join(self.train_args.merged_ckpt_path, "pytorch_model.bin")
            model_state_dict = torch.load(merged_ckpt_path, map_location='cpu')

            first_key = list(model_state_dict.keys())[0]
            if len(first_key.split('.')) == 1:
                # For Ring-Attention models, the merged checkpoint is directly copied from one of the shards and has different key names.
                model_state_dict = fix_model_state_dict(self.model, model_state_dict)
            
            first_key = list(model_state_dict.keys())[0]
            if 'model.model' not in first_key:
                # Our merging logic also removes the prefix `model.` from the state dict keys when saving
                model_state_dict = {'model.' + k: v for k, v in model_state_dict.items()}
            if self.rank % int(os.getenv("GPU_PER_NODE", "8")) == 0:
                print(f"Rank {self.rank} | {__name__} | loaded model state dict.keys(): {model_state_dict.keys()}")
            
            # in our merge program, `model` is poped out and we directly pass the model_state_dict instead of model_state_dict['model']
            nnscaler.load_merged_state_dict(
                self.model, model_state_dict,
                self.optimizer, None,
            )

        self.hook.after_setup(self)
    
    def _load_checkpoint(self):
        resume_from = self.train_args.checkpoint.get_resume_checkpoint_dir()
        if not resume_from:
            return
        logger.info(f"Resuming from {resume_from}")
        if resume_from.is_file():
            resume_from = resume_from   # when we load from merged checkpoint
        else:
            resume_from = resume_from / f'{self.rank}.ckpt'
        state_dict = torch.load(resume_from, map_location='cpu')
        self.hook.on_load_checkpoint(self, state_dict)
        ckpt_save_type = state_dict['train_args']['checkpoint']['save_type']

        if ckpt_save_type == 'merged': # it is a merged state dict
            nnscaler.load_merged_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
                )
        elif ckpt_save_type == 'sharded':
            nnscaler.load_sharded_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
            )
        elif ckpt_save_type == 'deduped':
            nnscaler.load_deduped_state_dict(
                self.model, state_dict['model'],
                self.optimizer, state_dict['optimizer'],
            )
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_save_type}")

        if 'lr_scheduler' in state_dict:
            if state_dict['lr_scheduler'] and not self.lr_scheduler:
                raise ValueError("lr_scheduler is not set in the current trainer")
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.train_status = TrainStatus(**state_dict['train_status'])
        self.rng_states_from_resume = state_dict.get('rng_states')  # resumed in _global_batch_iterator()