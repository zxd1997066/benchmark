"""
TorchBench Model Analyzer
Runs on a model and generates its metadata.
Used in the TorchBench New Model Analyzer CI.
"""
# This CI will error out if the model allows tuning batch size, and
# 1. The runtime and FLOPS do not increase with batch size increase, or
# 2. The suggested eval batch size does not match the default value
from dataclasses import dataclass
from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics

from typing import Optional, List, Dict

# We will enumerate all batch sizes from 1, 2, 4, ..., up to 4096
# Or stop when it hits CUDA OOM
DEFAULT_MAX_BATCH_SIZE = 4096

@dataclass
class TorchBenchModelMetadata:
    # Whether the model implements staged train test
    staged_train: bool
    # Whether the model allows customizing batch size
    allow_customize_batch_size: bool
    train_metrics_by_batch_size: Dict[int, TorchBenchModelMetrics]
    eval_metrics_by_batch_size: Dict[int, TorchBenchModelMetrics]
    suggested_train_batch_size: Optional[int]
    suggested_eval_batch_size: Optional[int]

def run(args: List[str]):
    pass
