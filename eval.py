from __future__ import annotations
import argparse
import json
import os

import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 10),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def _is_npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _visible_device_count():
    """Return the number of devices actually usable by this process.

    Honors ``ASCEND_RT_VISIBLE_DEVICES`` (NPU) and ``CUDA_VISIBLE_DEVICES``
    (CUDA) so that ``torch.multiprocessing.spawn`` does not over-spawn
    workers when the process is restricted to a subset of physical devices.
    """
    if _is_npu_available():
        vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        if vis:
            return len([d for d in vis.split(",") if d.strip() != ""])
        return torch.npu.device_count()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        return len([d for d in vis.split(",") if d.strip() != ""])
    return torch.cuda.device_count()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=("Comma-separated dataset names to evaluate (e.g. gsm8k). Defaults to all tasks."),
    )
    args = parser.parse_args()
    if args.tasks is not None:
        selected = {name.strip() for name in args.tasks.split(",")}
        args.tasks = [task for task in TASKS if task[0] in selected]
        assert args.tasks, f"No matching tasks for --tasks={args.tasks}"
    else:
        args.tasks = list(TASKS)
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=_visible_device_count(),
    )
