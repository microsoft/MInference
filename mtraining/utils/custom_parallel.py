import os
import torch
import shutil
import inspect
import torch.distributed as dist

from pathlib import Path
from typing import Callable, Any, Dict, Optional, Tuple, Type, Union, TypeVar, List, Set, Literal

from nnscaler.graph import IRGraph
from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.module import AttrMeta, CubeModule, ParallelModule, OriginModuleMetadata, ExtraState

from nnscaler.parallel import (
    ComputeConfig, ReuseType, BroadcastGenFilesStrategy, RegenStatus,
    _prepare_namespace, _compile_flags, _clean_files, _is_any_gencode_loaded, _gencode,
    _broadcast_gen_files, _load_parallel_module_class,
    _GENCODE_FILE_TEMPLATE, _GRAPH_DUMP_FILE, _FORWARD_ARGS_DUMP_FILE, _PREDEFINED_POLICIES

)

import logging
logger = logging.getLogger(__name__)

def compute_config_safe_equals(a: Optional['ComputeConfig'], b: Optional['ComputeConfig']) -> bool:
    """
    Return False if a and b are from incompatible version of ComputeConfig
    This is only for backward compatibility, and will be removed in future
    and can use `==` when we save dict version of ComputeConfig to file.
    """
    res = True
    try:
        for key in a.__dataclass_fields__:
            if getattr(a, key) != getattr(b, key):
                print(f"{key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)")
                
                if key == "user_config":
                    continue
                else:
                    print(f"compute_config_safe_equals | {key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)")
                    res = False
        return res
    except AttributeError:
        logger.warning("Failed to compare ComputeConfig. They are incompatible.")
        return False

GRAPH_CONFIG_FIELDS = ['constant_folding', 'user_config', 'inference_only', 'end2end_mode', 'trace_strategy']
def graph_config_equals(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Return False if a and b are from incompatible version of ComputeConfig
    This is only for backward compatibility, and will be removed in future
    and can use `==` when we save dict version of ComputeConfig to file.
    """
    res = True
    try:
        for key in GRAPH_CONFIG_FIELDS:
            if getattr(a, key) != getattr(b, key):
                print(f"graph_config_equals | {key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)")
                if key != "user_config":
                    res = False
        return res
    except AttributeError:
        logger.warning("Failed to compare GraphConfig. They are incompatible.")
        return False


TRACE_FILE_EXTENSIONS = [
    FxModuleParser.ATTR_CONTENT_FILE_0,  # init weights file(fullmodel.pt.*), 
    FxModuleParser.ATTR_MAP_FILE, # param name mapping (dist_param_map.pt)\
    _GRAPH_DUMP_FILE, # graph dump (graph.ckp), 
    _FORWARD_ARGS_DUMP_FILE, # forward args dump(forward_args.pkl), 
    ParallelModule.ORIGIN_MODULE_METADATA_FILE # origin module metadata (origin_module_metadata.pt), 
]
def transfer_metadata(out_dir, transfer_config: Dict[str, Any]):
    transfer_config_dir, transfer_force = transfer_config['transfer_config_dir'], transfer_config['transfer_force']
    if not os.path.exists(transfer_config_dir):
        # if transfer_config_dir is not set, use the default directory
        transfer_config_dir = transfer_config_dir.replace("compile_config/", "compile_config/rank_0/")
    assert os.path.exists(transfer_config_dir), f"Source directory {transfer_config_dir} for transferring does not exist"

    # transfer files in src_dir with postfix not being .py by executing `cp`
    print(f"{__name__} | Transfering files from {transfer_config_dir} to {out_dir} (local_rank={os.getenv('LOCAL_RANK')})")
    for file in os.listdir(transfer_config_dir):
        if file in TRACE_FILE_EXTENSIONS or file.startswith(FxModuleParser.ATTR_CONTENT_FILE_STEM):
            src_file = os.path.join(transfer_config_dir, file)
            dst_file = os.path.join(out_dir, file)

            print(f"{__name__} | Copying {src_file} to {dst_file} (local_rank={os.getenv('LOCAL_RANK')})" )
            if not os.path.exists(dst_file) or transfer_force:
                shutil.copyfile(src_file, dst_file)
        
            if not os.path.exists(dst_file):
                raise FileNotFoundError(f"{__name__} | Copy failed ({dst_file} does not exist after copying)")
        
    # Create a file 'transferred.sign' to indicate that the transfer is done
    with open(os.path.join(out_dir, "transferred.sign"), 'w') as f:
        f.write("Transferred from " + transfer_config_dir)

def _prepare_and_check_reusable(
        gen_savedir: str,
        module_or_module_class: Union[Type[torch.nn.Module], torch.nn.Module],
        compute_config: ComputeConfig,
        instance_name: Optional[str] = None,
        reuse: ReuseType = ReuseType.MATCH,
        transfer_config: Dict[str, Any] = None,
    ) -> Tuple[str, bool, bool]:
    """
    Prepare the output directory for code generation, and also check if the existing code is reusable.

    Args:
        gen_savedir (str): the directory to save generated code
        module_or_module_class (Union[Type[torch.nn.Module], torch.nn.Module]): the original module or module class
        compute_config (ComputeConfig): the environment resource
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        reuse (ReuseType): specify which part can be reused.

    Returns:
        Tuple[str, bool]: the output directory and whether the existing code is reusable.

    Raises:
        RuntimeError: if the existing code is not reusable,
            will raise RuntimeError if the code is not reusable but the module is already loaded.
    """
    namespace, outdir = _prepare_namespace(gen_savedir, module_or_module_class, instance_name)
    reusable = False
    transferred = False

    config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE

    # Empty + Transfer -> config match, graph match, tracing file present -> generate code by MATCH or MOO
    # Empty w.o. Transfer -> Empty -> generate code by MATCH or MOO
    has_transferred = os.path.exists(os.path.join(outdir, "transferred.sign"))
    if transfer_config is not None and transfer_config.get("transfer_config_dir", None) is not None \
            and (not has_transferred or transfer_config['transfer_force']):
        # transfer_config_dir: Optional[str] = None,
        transfer_metadata(outdir, transfer_config)
        ComputeConfig.safe_dump_to_file(compute_config, config_file)
        transferred = True
    
    # decision matrix for code generation
    # reuse flag | dir condition(imported, empty, match, unmatched) | action
    # ---------------------------------------------------------
    #   OVERRIDE   | empty           | generate
    #   OVERRIDE   | imported        | raise error
    #   OVERRIDE   | whatever match  | generate
    #   OVERRIDE   | unmatch         | generate
    #   GRAPH      | empty           | generate
    #   GRAPH      | imported        | raise error
    #   GRAPH      | graph match     | reuse graph, and regenerate code
    #   GRAPH      | all match       | reuse graph, and regenerate code
    #   GRAPH      | unmatch         | generate
    #   MATCH      | empty           | generate
    #   MATCH      | match           | reuse(do nothing)
    #   MATCH*     | whatever unmatch| raise error (except when there's no python source code, see below)
    #   MATCH      | imported        | doesn't matter
    #   MOO        | empty           | generate
    #   MOO        | match           | reuse(do nothing)
    #   MOO        | match graph     | reuse graph, and regenerate code
    #   MOO        | imported        | raise error if whatever unmatch
    #  *: The precondition for `except` part is the compute config should match.
    #     you can take it as a continous operation after a failed generation.
    old_config: Optional[ComputeConfig] = ComputeConfig.safe_load_from_file(config_file)
    is_config_match = compute_config_safe_equals(old_config, compute_config)
    # is_graph_config_match = old_config is not None and old_config.graph_config == compute_config.graph_config
    is_graph_config_match = old_config is not None and graph_config_equals(old_config.graph_config, compute_config.graph_config)
    trace_meta_files = [
        outdir / FxModuleParser.ATTR_CONTENT_FILE_0,  # init weights file(fullmodel.pt.*), 
        outdir / FxModuleParser.ATTR_MAP_FILE, # param name mapping (dist_param_map.pt)
    ]

    if reuse == ReuseType.MATCH or reuse == ReuseType.MOO:
        # check if the module is already generated
        expected_output_files = [outdir / _GENCODE_FILE_TEMPLATE.format(rank) for rank in range(compute_config.runtime_ngpus)]
        expected_output_files.extend(trace_meta_files)
        expected_output_files.append(config_file)
        expected_output_files.append(outdir / _GRAPH_DUMP_FILE) # graph dump (graph.ckp), 
        expected_output_files.append(outdir / _FORWARD_ARGS_DUMP_FILE) # forward args dump(forward_args.pkl), 
        expected_output_files.append(outdir / ParallelModule.ORIGIN_MODULE_METADATA_FILE) # origin module metadata (origin_module_metadata.pt), 
        existing_output_files = [
            f for f in outdir.glob('*')
            if f.is_file() and (  # just take fullmodel.pt.0 to compare
                not f.name.startswith(FxModuleParser.ATTR_CONTENT_FILE_STEM)
                or f.name == FxModuleParser.ATTR_CONTENT_FILE_0
            ) and not f.name.endswith('.sign')
        ]

        print(f"{__name__} | compute config match: {is_config_match}")
        print(f"{__name__} | graph config match: {is_graph_config_match}")
        print(f"{__name__} | existing output files: {existing_output_files}")
        print(f"{__name__} | expected output files: {expected_output_files}")
        
        if existing_output_files: # if the directory is not empty
            if is_config_match \
                and all([output_file.exists() for output_file in expected_output_files]) \
                and len(existing_output_files) == len(expected_output_files):

                print(f"{__name__} | Reuse existing files in {outdir}")
                reusable = True  # everything is matched.
            elif is_config_match \
                and all(f.suffix != '.py'  for f in existing_output_files):
                # No python source code is generated.
                # which means its last generation failed.
                # in this case, we can reuse the same directory safely.
                logger.info(f'Output directory {outdir} is not empty. '
                            f'But no python source code is present. '
                            f'Will reuse the directory and the graph dump if present.')
                # we have to trace the graph again if not all meta files are present.
                print(f"{__name__} | compute config match but no python code exists in {outdir}")
                if not all([meta_file.exists() for meta_file in trace_meta_files]):
                    print(f"{__name__} | compute config match but no python code exists in {outdir} and not all meta files are present")
                    _clean_files(outdir)
            elif reuse == ReuseType.MATCH:
                raise RuntimeError(f'Output directory {outdir} is not empty. '
                                   f'And the existing files do not match with current config. '
                                   f'You can remove the directory and try again, '
                                   f'or set reuse to ReuseType.NONE/ReuseType.OVERRIDE to regenerate the code.')
            else:
                assert reuse == ReuseType.MOO
                if _is_any_gencode_loaded(namespace):
                    raise RuntimeError(f'Output directory {outdir} is already loaded. '
                                       f'You can not override a loaded module.')
                elif is_graph_config_match:
                    # reuse the graph dump
                    print(f"{__name__} | MOO | graph match -> reuse graph but clean the current code")    
                    _clean_files(outdir, '*.py')
                else:
                    _clean_files(outdir)
    else:
        # check if the module is already loaded
        if _is_any_gencode_loaded(namespace):
            raise RuntimeError(f'Output directory {outdir} is already loaded. '
                               f'You can not override a loaded module.')
        # clear existing generated files
        if reuse == ReuseType.OVERRIDE \
            or not is_graph_config_match \
            or not all([meta_file.exists() for meta_file in trace_meta_files]):
            # we have to trace the graph again if not all meta files are present even when reuse=graph.
            print(f"{__name__} | OVERRIDE | Override existing files in {outdir}")
            glob_pattern = '*'
        else:
            print(f"{__name__} | GRAPH | keep the graph dump in {outdir} and regenerate the code")
            glob_pattern = '*.py'  # so we can keep graph dumps.
        _clean_files(outdir, glob_pattern)

    return outdir, reusable, transferred


def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype:  Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
    transfer_config: Optional[Dict[str, Any]] = None,
) -> Union[None, ParallelModule, Type[ParallelModule]]:
    """
    Convert a torch.nn.Module object or class to ParallelModule object or class.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distinguish them.

    Currently you must use a shared file system to share the generated files (like mounted Azure Blob)
    Or you can unset load_module flag, and manually copy the generated files to other nodes.
    After all nodes have the generated files, you can call parallelize() again with load_module flag set.

    Note: if reuse is not set to ReuseType.MATCH,
    the generated code in outdir will be removed EVEN IF the code generation fails in this call.

    if the input is a module object.
    * The module object will be copied to cpu to handle possible insufficient gpu memory.
    * The training flag will be the same as the original module

    This function can be used to convert both module object and module class to parallel module or parallel module class.
    Among key-value arguments,
    module_fn and module_dtype control how to create the module object.
    whereas init_module_params controls how to load parallel module object after conversion is done.

    1. If the input is a module object, it will return a ParallelModule object if load_module is True.
       This is useful when the module is created by a factory function.

       a. module_fn is ignored.
       b. module_dtype is used to control the dtype of the input module.
       c. init_module_params is used to control whether to initialize the parallel module parameters when load it.

    2. If the input is a module class, it will return a ParallelModule sub class if load_module is True.

       a. module_fn is used to create the module object, or module's__init__ if not prent.
       b. module_dtype is used to control the dtype of the created module (by constructor or module_fn).
          Of course, it can be merged into module_fn.
       c. init_module_params is ignored.

    After the module is converted, you can use it to create module object by calling it like a module class.
    The module class is defined like:

    ::

        class GenModule(nnscaler.runtime.module.ParallelModule):
            def __init__(self, init_params=True):
                super().__init__()
                ...
            ...

    So you can use `init_params` in `__init__` to control whether to initialize the module parameters.
    For example, if you don't want to initialize module params:

    ::

        module = GenModule(init_params=False)

    Args:
        module_or_module_class (Union[torch.nn.Module, Type[torch.nn.Module]]): the module or module class to be compiled
        dummy_forward_args (Dict[str, Any]): the dummy input for the module forward
        pas_policy (Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]]): the pas policy,
            it can be a name of builtin policies, or a custom policy function.
        compute_config (ComputeConfig): the environment resource
        reuse (ReuseType): specify which part can be reused.
        gen_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        load_module (bool): whether to load the generated module or module class after conversion is done.
        init_module_params (bool): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
            Otherwise, they will be empty tensor.
            This parameter will be passed to the module constructor,
            so it is only used when module_or_module_class is a module object, and load_module is true.
        module_dtype (Optional[torch.dtype]): the dtype of the module. Keep the module as it is if it is None.
        module_fn (Optional[Callable[[], torch.nn.Module]]): the function to create the module. Will use __init__ if it is None.
        broadcast_strategy (Union[str, BroadcastGenFilesStrategy]): the broadcast strategy for generated files.
            Please note that the broadcasting will only be done in torchrun environment,
            and will throw an error if dist is not initialized and broadcast_strategy is not NONE.
    Returns:
        Union[ParallelModule, Type[ParallelModule], None]:
            if load_module flag is set, return the converted ParallelModule object or class
            if load_module flag is not set, return None
    """
    if (
        isinstance(module_or_module_class, ParallelModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, ParallelModule))
    ):
        # already done
        return module_or_module_class if load_module else None

    if (
        isinstance(module_or_module_class, CubeModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, CubeModule))
    ):
        raise RuntimeError("Old style CubeModule is not supported")

    if isinstance(pas_policy, str):
        if not pas_policy in _PREDEFINED_POLICIES:
            raise ValueError(f"Invalid pas_policy: {pas_policy}")
        pas_policy = _PREDEFINED_POLICIES[pas_policy]

    is_module_class = inspect.isclass(module_or_module_class)
    module_class = module_or_module_class if is_module_class else module_or_module_class.__class__
    reuse = ReuseType(reuse) if isinstance(reuse, str) else reuse
    broadcast_strategy = BroadcastGenFilesStrategy(broadcast_strategy) if isinstance(broadcast_strategy, str) else broadcast_strategy

    # Call it here just to ensure the device group is initialized.
    # If the user initializes dist
    #     and doesn't call `nnscaler.init()` before calling this function, this is necessary.
    if dist.is_initialized():
        _ = DeviceGroup()

    # generate code only in node0
    # if it is not in a torchrun environment, just generate.
    if not dist.is_initialized() or dist.get_rank() == 0:
        outdir, reusable, transferred = _prepare_and_check_reusable(
            gen_savedir, module_class, compute_config, instance_name, reuse,
            transfer_config
        )
        if not reusable:
            config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE
            ComputeConfig.safe_dump_to_file(compute_config, config_file)  # always refresh compute config
            with _compile_flags(compute_config):
                regen_status = _gencode(
                    module_or_module_class,
                    dummy_forward_args,
                    pas_policy,
                    compute_config,
                    outdir,
                    module_dtype=module_dtype,
                    module_fn=module_fn,
                )
        else:
            regen_status = RegenStatus.NONE
            logger.info(f"Reuse generated code in {outdir}")
        
        if regen_status == RegenStatus.CODE and transferred:
            regen_status = RegenStatus.ALL

    if dist.is_initialized():
        # code generation can take very long time (for example, over 1 hour)
        # It is not always OK to use dist.barrier() directly.
        # because the default timeout for nccl is 30 minutes
        # (we can't control the timeout setting if dist is not initialized by us)
        DeviceGroup().long_barrier()

    if broadcast_strategy != BroadcastGenFilesStrategy.NONE:
        if not dist.is_initialized(): # we only support loading in torchrun environment
            raise RuntimeError("Broadcast generated files failed: dist is not initialized.")
        dist.barrier()
        # sync regen_status
        curr_rank = dist.get_rank()
        if curr_rank == 0:
            sent_obj = [regen_status]
        else:
            sent_obj = [None]
        dist.broadcast_object_list(
            sent_obj,
            src=0,
        )
        if curr_rank != 0:
            regen_status = sent_obj[0]

        # narrow down broadcast_strategy according to regen_status
        if regen_status == RegenStatus.NONE:
            # we don't need to broadcast anything
            broadcast_strategy = BroadcastGenFilesStrategy.NONE
        elif regen_status == RegenStatus.CODE:
            # narrow ALL/NO_WEIGHTS down to code
            broadcast_strategy = BroadcastGenFilesStrategy.CODE
        else:
            # we don't need to narrow broadcast_strategy in this case
            # keep the original broadcast_strategy
            assert regen_status == RegenStatus.ALL

        # broadcast generated files according to regen_status
        if broadcast_strategy != BroadcastGenFilesStrategy.NONE:
            _broadcast_gen_files(
                module_class,
                gen_savedir=gen_savedir,
                instance_name=instance_name,
                broadcast_strategy=broadcast_strategy,
            )
        elif os.getenv("FORCE_BROADCAST") == "1":
            # force broadcast generated files
            print(f"Force broadcast generated files in {gen_savedir}")
            _broadcast_gen_files(
                module_class,
                gen_savedir=gen_savedir,
                instance_name=instance_name,
                broadcast_strategy=BroadcastGenFilesStrategy.ALL,
            )

    if load_module:
        if not dist.is_initialized(): # we only support loading in torchrun environment
            raise RuntimeError("Load ParallelModule failed: dist is not initialized.")
        dist.barrier()
        parallel_module_class = _load_parallel_module_class(
            module_class,
            gen_savedir=gen_savedir,
            instance_name=instance_name,
        )
        if is_module_class:
            return parallel_module_class
        else:
            parallel_module = parallel_module_class(init_module_params)
            parallel_module.train(module_or_module_class.training)  # set training state to the same as original module
            return parallel_module
