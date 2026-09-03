import argparse
import importlib.util
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import scipy.io as sio


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATE_PATH = "/media/zouhe/Elements/baseline/baseline_test/evaluate.py"
VARIANTS = [
    "full",
    "wo_stage1",
    "wo_pretrain_loss",
    "wo_spatial_loss",
    "wo_spectral_loss",
    "wo_ssat",
]
REQUIRED_MAT_FIELDS = ("proposed", "I_MS_LR", "I_MS", "I_PAN")


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行ZUP完整模型与五组消融，并自动完成Full-Resolution评测"
    )
    parser.add_argument("--run_name", required=True, help="本次实验目录名称")
    parser.add_argument(
        "--output_root",
        default=os.path.join(SCRIPT_DIR, "ablation_results"),
        help="消融实验输出根目录",
    )
    parser.add_argument("--data_path", required=True, help="WV3 Full-Resolution H5数据路径")
    parser.add_argument("--teacher_source", choices=["mat", "model"], default="mat")
    parser.add_argument("--teacher_result_dir", default=None, help="教师MAT结果目录")
    parser.add_argument("--u2net_path", default="FusionMamba/weights/420.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sensor", default="WV3")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=19)
    parser.add_argument("--pre_lr", type=float, default=0.015)
    parser.add_argument("--pre_epochs", type=int, default=8)
    parser.add_argument("--sde_lr", type=float, default=0.005)
    parser.add_argument("--sde_epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=0.00575)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alfa", type=float, default=0.15)
    return parser.parse_args()


def validate_args(args):
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")
    if args.teacher_source == "mat" and not os.path.isdir(args.teacher_result_dir or ""):
        raise NotADirectoryError(f"教师MAT目录不存在: {args.teacher_result_dir}")
    if not os.path.isfile(EVALUATE_PATH):
        raise FileNotFoundError(f"评测脚本不存在: {EVALUATE_PATH}")
    if args.start_id < 0 or args.end_id < args.start_id:
        raise ValueError("图片编号范围无效")
    if args.sensor.upper() != "WV3":
        raise ValueError("当前消融入口仅支持WV3 Full-Resolution评测")
    with h5py.File(args.data_path, "r") as dataset:
        num_samples = dataset["lms"].shape[0]
    if args.end_id >= num_samples:
        raise ValueError(f"end_id={args.end_id}超出数据集范围0~{num_samples - 1}")
    if args.teacher_source == "mat":
        missing_teacher = [
            data_id
            for data_id in range(args.start_id, args.end_id + 1)
            if not os.path.isfile(
                os.path.join(args.teacher_result_dir, f"output_mulExm_{data_id}.mat")
            )
        ]
        if missing_teacher:
            raise FileNotFoundError(f"缺少教师MAT，图片编号: {missing_teacher}")
    if args.run_name in (".", ".."):
        raise ValueError("run_name不能是当前目录或上级目录")
    if os.path.sep in args.run_name or (os.path.altsep and os.path.altsep in args.run_name):
        raise ValueError("run_name不能包含路径分隔符")


def run_command(command, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"\n执行命令: {' '.join(command)}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def teacher_arguments(args):
    command = [f"--teacher_source={args.teacher_source}"]
    if args.teacher_result_dir:
        command.append(f"--teacher_result_dir={os.path.abspath(args.teacher_result_dir)}")
    if args.u2net_path:
        command.append(f"--u2net_path={args.u2net_path}")
    return command


def train_shared_models(args, run_dir, data_id):
    sensor = args.sensor.upper()
    pretrain_path = os.path.join(
        run_dir, "checkpoints", "shared", "model_pretrain", sensor, f"{data_id}.pth"
    )
    sde_path = os.path.join(
        run_dir, "checkpoints", "shared", "model_SDE", sensor, f"{data_id}.pth"
    )

    pretrain_command = [
        sys.executable,
        "pretrain.py",
        f"--lr={args.pre_lr}",
        f"--epochs={args.pre_epochs}",
        f"--batch_size={args.batch_size}",
        f"--device={args.device}",
        f"--data_id={data_id}",
        f"--sensor={args.sensor}",
        f"--ratio={args.ratio}",
        f"--temperature={args.temperature}",
        f"--data_path={os.path.abspath(args.data_path)}",
        f"--output_path={pretrain_path}",
    ]
    run_command(
        pretrain_command,
        os.path.join(run_dir, "logs", f"image_{data_id}", "pretrain.log"),
    )

    sde_command = [
        sys.executable,
        "main_SDE_amp.py",
        f"--lr={args.sde_lr}",
        f"--epochs={args.sde_epochs}",
        f"--batch_size={args.batch_size}",
        f"--device={args.device}",
        f"--satellite={args.sensor}",
        f"--name={data_id}",
        f"--data_path={os.path.abspath(args.data_path)}",
        f"--output_path={sde_path}",
    ]
    run_command(
        sde_command,
        os.path.join(run_dir, "logs", f"image_{data_id}", "sde.log"),
    )
    return pretrain_path, sde_path


def train_and_test_variant(args, run_dir, data_id, variant, pretrain_path, sde_path):
    checkpoint_path = os.path.join(
        run_dir, "checkpoints", variant, f"{args.sensor.upper()}_{data_id}_best.pth"
    )
    mat_path = os.path.join(
        run_dir, "mat", variant, "wv3_full", f"{data_id}.mat"
    )

    train_command = [
        sys.executable,
        "train_SDE.py",
        f"--ablation={variant}",
        f"--data_id={data_id}",
        f"--lr={args.lr}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--device={args.device}",
        f"--sensor={args.sensor}",
        f"--ratio={args.ratio}",
        f"--temperature={args.temperature}",
        f"--alfa={args.alfa}",
        f"--data_path={os.path.abspath(args.data_path)}",
        f"--output_path={checkpoint_path}",
    ]
    train_command.extend(teacher_arguments(args))
    if variant != "wo_stage1":
        train_command.append(f"--pretrain_path={pretrain_path}")
    if variant != "wo_spatial_loss":
        train_command.append(f"--sde_path={sde_path}")
    run_command(
        train_command,
        os.path.join(run_dir, "logs", f"image_{data_id}", f"train_{variant}.log"),
    )

    test_command = [
        sys.executable,
        "testba.py",
        f"--data_id={data_id}",
        f"--data_path={os.path.abspath(args.data_path)}",
        f"--sensor_type={args.sensor}",
        f"--device={args.device}",
        "--mode=normal",
        "--process_model=1",
        "--student_only",
        f"--checkpoint_path={checkpoint_path}",
        f"--output_path={mat_path}",
    ]
    run_command(
        test_command,
        os.path.join(run_dir, "logs", f"image_{data_id}", f"test_{variant}.log"),
    )


def verify_mat_results(run_dir, sample_ids):
    errors = []
    for variant in VARIANTS:
        for data_id in sample_ids:
            mat_path = os.path.join(run_dir, "mat", variant, "wv3_full", f"{data_id}.mat")
            if not os.path.isfile(mat_path):
                errors.append(f"缺少文件: {mat_path}")
                continue
            data = sio.loadmat(mat_path)
            missing = [field for field in REQUIRED_MAT_FIELDS if field not in data]
            if missing:
                errors.append(f"{mat_path} 缺少字段: {', '.join(missing)}")
                continue
            proposed = np.asarray(data["proposed"])
            lms = np.asarray(data["I_MS"])
            pan = np.asarray(data["I_PAN"]).squeeze()
            if proposed.shape != lms.shape or proposed.shape[:2] != pan.shape:
                errors.append(
                    f"{mat_path} 形状错误: proposed={proposed.shape}, "
                    f"I_MS={lms.shape}, I_PAN={pan.shape}"
                )
            if not all(np.isfinite(np.asarray(data[field])).all() for field in REQUIRED_MAT_FIELDS):
                errors.append(f"{mat_path} 包含NaN或Inf")
    if errors:
        raise RuntimeError("MAT完整性检查失败:\n" + "\n".join(errors))


def load_evaluator():
    evaluate_dir = os.path.dirname(EVALUATE_PATH)
    if evaluate_dir not in sys.path:
        sys.path.insert(0, evaluate_dir)
    spec = importlib.util.spec_from_file_location("zup_ablation_evaluate", EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_results(args, run_dir, sample_ids):
    evaluator = load_evaluator()
    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    reduced_name, _ = evaluator.DATASET_H5["wv3"]
    evaluator.TEST_DATA_DIR = data_dir
    evaluator.DATASET_H5["wv3"] = (reduced_name, os.path.basename(args.data_path))

    all_results = []
    for variant in VARIANTS:
        result_spec = {
            "name": f"ZUP_{variant}",
            "type": "self_contained",
            "base_dir": os.path.join(run_dir, "mat", variant, "wv3_full"),
            "result_dir": ".",
            "file_pattern": "%d.mat",
            "num_samples": max(sample_ids) + 1,
            "field_fused": "proposed",
            "field_gt": None,
            "field_pan": "I_PAN",
            "field_ms": "I_MS_LR",
            "field_lms": "I_MS",
            "datasets": ["wv3"],
            "modes": ["full"],
            "sensor_map": {"wv3": args.sensor.upper()},
        }
        results = evaluator.evaluate_special_baseline(
            result_spec, modes=["full"], datasets=["wv3"]
        )
        if not results or results[0].get("num_valid") != len(sample_ids):
            raise RuntimeError(f"{variant}评测样本数不完整")
        all_results.extend(results)

    metrics_dir = os.path.join(run_dir, "metrics")
    evaluator.save_csv(all_results, metrics_dir)
    evaluator.print_summary(all_results)


def main():
    args = parse_args()
    validate_args(args)
    run_dir = os.path.abspath(os.path.join(args.output_root, args.run_name))
    if os.path.exists(run_dir):
        raise FileExistsError(f"实验目录已存在，拒绝覆盖: {run_dir}")
    os.makedirs(run_dir)

    config = vars(args).copy()
    config["seed"] = 10
    config["variants"] = VARIANTS
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    sample_ids = list(range(args.start_id, args.end_id + 1))
    for offset, data_id in enumerate(sample_ids, 1):
        print(f"\n{'#' * 72}")
        print(f"图片 {data_id} ({offset}/{len(sample_ids)}): 训练共享前置模型")
        print(f"{'#' * 72}")
        pretrain_path, sde_path = train_shared_models(args, run_dir, data_id)
        for variant in VARIANTS:
            print(f"\n图片 {data_id}: 开始实验 {variant}")
            train_and_test_variant(
                args, run_dir, data_id, variant, pretrain_path, sde_path
            )

    verify_mat_results(run_dir, sample_ids)
    evaluate_results(args, run_dir, sample_ids)
    print(f"\n消融实验完成，全部结果保存在: {run_dir}")


if __name__ == "__main__":
    main()
