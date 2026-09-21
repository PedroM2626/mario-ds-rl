import os
import sys
import glob
import json
import subprocess
import time
import argparse
import mlflow

def find_latest_logdir(nedreamer_dir):
    log_base = os.path.join(nedreamer_dir, "logdir")
    # Paths are like logdir/YYYY-MM-DD/HH-MM-SS
    date_dirs = glob.glob(os.path.join(log_base, "*"))
    if not date_dirs:
        return None
    latest_date_dir = max(date_dirs, key=os.path.getmtime)
    time_dirs = glob.glob(os.path.join(latest_date_dir, "*"))
    if not time_dirs:
        return None
    latest_time_dir = max(time_dirs, key=os.path.getmtime)
    return latest_time_dir

def upload_to_mlflow(logdir, run_id, exp_name="MarioDS_RL_100k"):
    metrics_path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"metrics.jsonl not found in {logdir}")
        return

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(exp_name)
    
    print(f"Uploading metrics from {metrics_path} to MLflow run: {run_id}")
    
    with mlflow.start_run(run_name=run_id):
        # We will log the config from hydra overrides if it exists
        overrides_path = os.path.join(logdir, ".hydra", "overrides.yaml")
        if os.path.exists(overrides_path):
            with open(overrides_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("- "):
                        line = line[2:]
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.lstrip("+-")
                            mlflow.log_param(k, v)
        
        mlflow.log_param("algo", "ne-dreamer")
        
        with open(metrics_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    step = data.get("step")
                    if step is not None:
                        for k, v in data.items():
                            if k != "step":
                                if k == "episode/score":
                                    mlflow.log_metric("rollout/ep_rew_mean", v, step=step)
                                elif k == "episode/length":
                                    mlflow.log_metric("rollout/ep_len_mean", v, step=step)
                                else:
                                    mlflow.log_metric(f"ne/{k}", v, step=step)
                except Exception as e:
                    pass
        print("MLflow upload complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="nedreamer_100k_1")
    parser.add_argument("--steps", type=int, default=100000)
    args, unknown = parser.parse_known_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    nedreamer_dir = os.path.join(project_dir, "nedreamer")
    
    if not os.path.exists(nedreamer_dir):
        print("NEDreamer repository not found.")
        sys.exit(1)
        
    cmd = [
        sys.executable,
        os.path.join(nedreamer_dir, "train.py"),
        "env=marionds",
        "model=size12M",
        r"logdir=logdir/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        f"env.steps={args.steps}",
        "env.eval_episode_num=1",
        "env.time_limit=108000",
        "model.compile=False",
        "batch_size=8",
        "batch_length=32"
    ]
    cmd.extend(unknown)
    
    print(f"Running NE-Dreamer: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, cwd=nedreamer_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f}s.")
    
    # Upload to mlflow
    latest_logdir = find_latest_logdir(nedreamer_dir)
    if latest_logdir:
        upload_to_mlflow(latest_logdir, args.run_id)
    else:
        print("Could not find logdir to upload.")

if __name__ == "__main__":
    main()
