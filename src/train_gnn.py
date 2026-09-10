"""Treino PPO + GNN (extrator MeanMPNN puro-torch) sobre MarioGraphEnv."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_env import MarioGraphEnv
from gnn_extractor import GNNExtractor, OBS_DIM


def make_env(rom_path, state_path, rank):
    def _init():
        import time as _t
        import shutil
        import tempfile
        _t.sleep(rank * 2.0)  # init escalonado: DeSmuMEs simultaneos colidem
        tmp = tempfile.gettempdir()
        pid = os.getpid()
        t_rom = os.path.join(tmp, f"mario_gnn_rom_{rank}_{pid}.nds")
        t_st = os.path.join(tmp, f"mario_gnn_state_{rank}_{pid}.ds1")
        shutil.copy2(rom_path, t_rom)
        shutil.copy2(state_path, t_st)
        env = MarioGraphEnv(rom_path=t_rom, state_path=t_st)
        orig_close = env.close

        def _close():
            try:
                orig_close()
            finally:
                for f in (t_rom, t_st):
                    try:
                        if os.path.exists(f):
                            os.remove(f)
                    except Exception:
                        pass
        env.close = _close
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--run-id", type=str, default="gnn_100k")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--rom", type=str,
                    default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    ap.add_argument("--state", type=str,
                    default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    args = ap.parse_args()

    import mlflow
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")
    fns = [make_env(args.rom, args.state, i) for i in range(args.num_envs)]
    env = SubprocVecEnv(fns) if args.num_envs > 1 else DummyVecEnv(fns)
    assert env.observation_space.shape == (OBS_DIM,), env.observation_space.shape
    t0 = time.time()
    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_GNN_MeanMPNN")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("num_envs", args.num_envs)
        mlflow.log_param("obs", f"Box({OBS_DIM}) grafo: 14 nos x 8 + global 8 + mask")
        policy_kwargs = dict(features_extractor_class=GNNExtractor,
                             features_extractor_kwargs=dict(features_dim=256))
        if args.resume and os.path.exists(f"{args.resume}.zip"):
            print(f"Resume de {args.resume}.zip", flush=True)
            model = PPO.load(args.resume, env=env)
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        else:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None,
                        policy_kwargs=policy_kwargs)
            model.learn(total_timesteps=args.timesteps)
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{args.run_id}")
        mlflow.log_artifact(f"models/{args.run_id}.zip")
    dt = time.time() - t0
    print(f"GNN: {args.timesteps} steps em {dt:.0f}s ({args.timesteps/dt:.1f} fps)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
