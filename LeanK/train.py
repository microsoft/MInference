# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import os
import types

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from leank.data import (
    MultiplePasskeyRetrievalDataset,
    PasskeyRetrievalDataset,
    get_dataset,
    get_supervised_dataloader,
)
from leank.loss import l1_loss
from leank.patch import (
    enable_training,
    full_attn_forward_llama,
    full_attn_forward_qwen,
    get_scaling_factors,
    load_scaling_factors,
    map_scaling_factors,
    scaled_attn_forward_llama,
    scaled_attn_forward_qwen,
)
from leank.utils import (
    convert_to_list,
    get_tokenizer,
    parse_args,
    save_scaling_factors,
    seed_everything,
    sparsify_scaling_factors,
    visualize_patterns,
)
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)


def train(
    args, model, rank, world_size, train_dataloader, optimizer, scheduler, resume_step
):
    model.train()

    if rank == 0:
        pbar = tqdm(range(args.num_steps))

    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    while True:
        if global_step >= args.num_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping step {global_step} to resume to {resume_step}"
                    )
                continue

            @torch.no_grad()
            def clamp_(x, min_val, max_val):
                x.clamp_(min_val, max_val)

            map_scaling_factors(model, func=lambda x: clamp_(x, 0, 1))

            length_context = batch["length_context"]
            batch.pop("length_context")

            batch = {k: v.to(f"cuda:{local_rank}") for k, v in batch.items()}

            input_ids = batch["input_ids"]

            n_seq = (input_ids.shape[-1] + world_size - 1) // world_size
            pad_len = n_seq * world_size - input_ids.shape[-1]
            input_ids = torch.cat(
                (
                    torch.Tensor([0] * pad_len)
                    .unsqueeze(0)
                    .to(input_ids.device)
                    .to(input_ids.dtype),
                    input_ids,
                ),
                dim=-1,
            )
            sample_len = input_ids.shape[-1]

            seq_parallel_chunk_start = n_seq * rank
            seq_parallel_chunk_end = seq_parallel_chunk_start + n_seq
            position_ids = torch.arange(
                seq_parallel_chunk_start,
                seq_parallel_chunk_end,
                device=input_ids.device,
            ).unsqueeze(0)

            scaling_factors_list = []

            for layer in model.layers:
                module = layer.self_attn
                if isinstance(module._checkpoint_wrapped_module, LlamaAttention):
                    module.forward = types.MethodType(full_attn_forward_llama, module)
                elif isinstance(module._checkpoint_wrapped_module, Qwen2Attention):
                    module.forward = types.MethodType(full_attn_forward_qwen, module)
                else:
                    assert False
                scaling_factors_list.append(
                    module.scaling_factors.full_tensor().to(model.device)
                )

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids[
                        :, seq_parallel_chunk_start:seq_parallel_chunk_end
                    ],
                    position_ids=position_ids,
                    length_context=length_context[0] + pad_len,
                )
                if args.stage2:
                    mask_round = sparsify_scaling_factors(
                        scaling_factors_list, args.ratio, args.align
                    )

            original_hidden_states = outputs[0]

            for i, layer in enumerate(model.layers):
                module = layer.self_attn
                if isinstance(module._checkpoint_wrapped_module, LlamaAttention):
                    module.forward = types.MethodType(scaled_attn_forward_llama, module)
                elif isinstance(module._checkpoint_wrapped_module, Qwen2Attention):
                    module.forward = types.MethodType(scaled_attn_forward_qwen, module)
                else:
                    assert False
                module.mask_round = mask_round[i] if args.stage2 else None

            outputs = model(
                input_ids=input_ids[:, seq_parallel_chunk_start:seq_parallel_chunk_end],
                position_ids=position_ids,
                length_context=length_context[0] + pad_len,
            )
            pruned_hidden_states = outputs[0]

            labels = batch["labels"]
            labels = torch.cat(
                (
                    torch.Tensor([-100] * pad_len)
                    .unsqueeze(0)
                    .to(labels.device)
                    .to(labels.dtype),
                    labels,
                ),
                dim=-1,
            )
            labels = labels[:, seq_parallel_chunk_start:seq_parallel_chunk_end]
            label_mask = labels != -100
            num_labels = label_mask.sum()
            global_num_labels = num_labels.clone().detach()
            dist.all_reduce(global_num_labels)

            # filter out label == IGNORE_INDEX (-100)
            original_hidden_states = original_hidden_states[label_mask].float()
            pruned_hidden_states = pruned_hidden_states[label_mask].float()

            distill_loss = (
                (original_hidden_states - pruned_hidden_states)
                .pow(2)
                .mean(dim=-1)
                .sum()
                * world_size
                / global_num_labels
            )

            scaling_factors = get_scaling_factors(model)
            scaling_factors = [
                h.full_tensor().to(model.device) for h in scaling_factors
            ]

            reg_loss = l1_loss(torch.cat(scaling_factors).float())

            loss = distill_loss + args.reg_weight * reg_loss

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.AVG)

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            if local_step != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if rank == 0:
                scaling_factors_list = convert_to_list(scaling_factors)

                if not args.disable_wandb:
                    fig = visualize_patterns(scaling_factors_list)

                    wandb.log(
                        {
                            "distill_loss": distill_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "loss": loss.item(),
                            "attn_heads": fig,
                            "step": global_step,
                            "sample_len": sample_len,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    plt.close(fig)
                    del fig

                pbar.set_description(
                    f"Len={sample_len}/{global_num_labels}|Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|Loss={loss.item(): 3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:
                if rank == 0:
                    save_scaling_factors(
                        scaling_factors_list,
                        os.path.join(
                            args.output_dir,
                            f"scaling_factors_step={global_step}.tsv",
                        ),
                    )
                    os.system(f"rm {args.output_dir}/scaling_factors_latest.tsv")
                    os.system(
                        f"cp {args.output_dir}/scaling_factors_step={global_step}.tsv {args.output_dir}/scaling_factors_latest.tsv"
                    )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}-rank={rank}.pt",
                    ),
                )

                # copy the scaling_factors and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}-rank={rank}.pt {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )

            if global_step >= args.num_steps:
                break

            torch.cuda.empty_cache()

    if rank == 0:
        pbar.close()


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_name)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    if args.rope_theta is not None:
        print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
        config.rope_theta = args.rope_theta

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        attn_implementation="flash_attention_2",
    )

    if args.resume and os.path.exists(
        os.path.join(
            args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
        )
    ):
        # load the latest state in the output_dir
        state = torch.load(
            os.path.join(
                args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
            )
        )
        resume_step = state["global_step"]
        scaling_factors = load_scaling_factors(
            args.output_dir, filename=f"scaling_factors_step={resume_step}.tsv"
        )
    else:
        resume_step = -1
        scaling_factors = None

    if args.stage2:
        scaling_factors = load_scaling_factors(
            args.stage1_rst_path, filename=f"scaling_factors.tsv"
        )
    else:
        scaling_factors = None

    enable_training(
        model,
        args.sink_size,
        args.recent_size,
        initial_value=args.initial_value,
        enable_ulysses_attention=True,
        scaling_factors=scaling_factors,
    )

    model = model.model

    for param in model.parameters():
        param.requires_grad = False

    num_attn_heads = 0
    for name, param in model.named_parameters():
        if "scaling_factors" in name:
            param.requires_grad = True
            num_attn_heads += param.numel()

    setup()

    torch.cuda.set_device(local_rank)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    apply_activation_checkpointing(model)

    mesh = DeviceMesh(device_type="cuda", mesh=[i for i in range(world_size)])

    apply_fsdp(
        model,
        mesh,
        mp_policy,
        modules_to_shard={LlamaDecoderLayer, Qwen2DecoderLayer},
    )

    if rank == 0:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"Trainable parameter: {name} with shape {param.shape}, dtype {param.dtype}, device {param.device}"
                )

    if args.dataset_format == "passkey_retrival_mixed_tasks":
        train_dataset = PasskeyRetrievalDataset(
            tokenizer,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
        )
    elif args.dataset_format == "multi_key_retrival":
        train_dataset = PasskeyRetrievalDataset(
            tokenizer,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            type_haystack=["needle"],
        )
    elif args.dataset_format == "multi_val_retrival":
        train_dataset = PasskeyRetrievalDataset(
            tokenizer,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            type_haystack=["essay"],
        )
    elif args.dataset_format == "duo_data":
        haystack_dataset = get_dataset("leank/data/booksum.jsonl.zst", split="train")
        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
        )
    else:
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")

    train_dataloader = get_supervised_dataloader(
        train_dataset, tokenizer, args.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / (args.num_steps // 5), 0.1),
            max((args.num_steps - step) / (args.num_steps // 5), 0.1),
        ),
    )

    if rank == 0:
        experiment_config = vars(args)
        if not args.disable_wandb:
            wandb.init(project="LeanK", config=experiment_config)
            if args.exp_name is not None:
                wandb.run.name = args.exp_name

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f)

    train(
        args,
        model,
        rank,
        world_size,
        train_dataloader,
        optimizer,
        scheduler,
        resume_step,
    )

    scaling_factors = get_scaling_factors(model)
    scaling_factors = [h.full_tensor() for h in scaling_factors]

    if rank == 0:
        print("Training finished")
        if args.output_dir is not None:
            if args.stage2:
                with torch.no_grad():
                    mask_final = sparsify_scaling_factors(
                        scaling_factors, args.ratio, args.align
                    )
                    torch.save(
                        mask_final,
                        os.path.join(
                            args.output_dir,
                            f"mask_ratio{args.ratio}_align{args.align}.pth",
                        ),
                    )

            scaling_factors_list = convert_to_list(scaling_factors)
            save_scaling_factors(
                scaling_factors_list,
                os.path.join(args.output_dir, "scaling_factors.tsv"),
            )

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.stage2:
        config = config["training"]["stage2"]
    else:
        config = config["training"]["stage1"]
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    seed_everything(args.seed)
    main(args)
