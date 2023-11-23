"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import contextlib
import distutils.util
from typing import List
import torch
import torch._dynamo as torchdynamo
from torchbenchmark.util.model import is_staged_train_test

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends(exclude_tags=None)
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--tritonmm", type=str, help="torchinductor.config.triton.mm configuration"
    )
    parser.add_argument(
        "--dynamic_shapes",
        action='store_true',
        help="dynamic shape and symbolic tracing",
    )
    parser.add_argument(
        "--pt2_debug_log",
        action='store_true',
        help="enable debug log for PT2 (dynamo, inductor, AOTAutograd)",
    )
    parser.add_argument(
        "--full_graph",
        action='store_true',
        help="capture full graph and no python",
    )
    parser.add_argument(
        "--optimize_dynamo_ddp",
        action='store_true',
        help="enable extra optimizations for DDP + dynamo"
    )
    parser.add_argument(
        "--torchinductor_cudagraph",
        type=distutils.util.strtobool,
        default="true",
    )
    parser.add_argument(
        "--torchinductor_fallback_random",
        type=distutils.util.strtobool,
        default="false",
    )
    parser.add_argument(
        "--dynamo_disable_optimizer_step",
        type=distutils.util.strtobool,
        default="false",
    )
    parser.add_argument(
        "--quantize",
        action='store_true',
        help="enable quantize for inducotr",
    )
    parser.add_argument(
        "--PTQ",
        action='store_true',
        help="enable PTQ Quantization",
    )
    parser.add_argument(
        "--cpp_wrapper",
        action='store_true',
        help="enable cpp_wrapper",
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    else:
        dynamo_kwargs = {}
        if args.dynamic_shapes:
            dynamo_kwargs["dynamic"] = True
        if args.full_graph:
            dynamo_kwargs["nopython"] = True
        dynamo_optimizer = torchdynamo.optimize(args.torchdynamo, **dynamo_kwargs)
        if args.pt2_debug_log:
            import logging
            torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG)

    if args.torchdynamo == "inductor":
        import torch._inductor as torchinductor
        if model.device == "cuda":
            torchinductor.config.triton.cudagraphs = bool(args.torchinductor_cudagraph)
        if model.device == "cpu" and model.test == "eval" and args.quantize:
            enable_inductor_quant(model, args)
        # Setup torchinductor.config.triton.mm
        if args.tritonmm == "triton":
            torchinductor.config.triton.mm = "triton"
            # currently can't pass correctness with use_bmm = True
            # torchinductor.config.triton.use_bmm = True

        # used for correctness checks, to avoid triton rand() behaving differently from torch rand().
        torchinductor.config.fallback_random = bool(args.torchinductor_fallback_random)

    if bool(args.dynamo_disable_optimizer_step):
        found_optimizer_step = False
        try:
            model.cfg.optimizer.step = torch._dynamo.disable(model.cfg.optimizer.step)
            found_optimizer_step = True
        except AttributeError:
            pass

        try:
            model.optimizer.step = torch._dynamo.disable(model.optimizer.step)
            found_optimizer_step = True
        except AttributeError:
            pass

        if not found_optimizer_step:
            warnings.warn("--dynamo_disable_optimizer_step is set to True, but the optimizer could not be found on this model")

    if model.test == "train":
        if is_staged_train_test(model):
            model.forward = dynamo_optimizer(model.forward)
        else:
            model.train = dynamo_optimizer(model.train)
    else:
        model.model = dynamo_optimizer(model.model)

    if args.optimize_dynamo_ddp:
        @contextlib.contextmanager
        def optimize_ddp_ctx(val: bool):
            old_value = torchdynamo.config.optimize_ddp
            try:
                torchdynamo.config.optimize_ddp = val
                yield
            finally:
                torchdynamo.config.optimize_ddp = old_value
        model.add_context(lambda: optimize_ddp_ctx(True))

    torchdynamo.reset()

def enable_inductor_quant(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    import copy
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    if args.PTQ:
        from torch._export import capture_pre_autograd_graph, dynamic_dim
    module, example_inputs = model.get_module()
    # Generate the FX Module
    if args.PTQ:
        exported_model = capture_pre_autograd_graph(
                module,
                example_inputs
            )
    else:
         exported_model, guards = torchdynamo.export(
                 module,
                 *copy.deepcopy(example_inputs),
                 aten_graph=True,
             )

    # Create X86InductorQuantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    if args.cpp_wrapper:
        torch._inductor.config.cpp_wrapper = True
    # PT2E Quantization flow
    prepared_model = prepare_pt2e(exported_model, quantizer)
    # Calibration
    prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model).eval()
    model.set_module(converted_model)
