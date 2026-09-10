"""Treino PPO 100% RAM (sem pixels/CNN): PPO MLP sobre MarioRamEnv."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ram_env import MarioRamEnv


def make_env(rom_path, state_path, rank, geo=False):
    def _init():
        import time as _t
        import shutil
        import tempfile
        _t.sleep(rank * 2.0)  # init escalonado: DeSmuMEs simultaneos colidem
        tmp = tempfile.gettempdir()
        pid = os.getpid()
        t_rom = os.path.join(tmp, f"mario_ram_rom_{rank}_{pid}.nds")
        t_st = os.path.join(tmp, f"mario_ram_state_{rank}_{pid}.ds1")
        shutil.copy2(rom_path, t_rom)
        shutil.copy2(state_path, t_st)
        env = MarioRamEnv(rom_path=t_rom, state_path=t_st, geo=geo)
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
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--run-id", type=str, default="ram_ppo_100k")
    ap.add_argument("--resume", type=str, default=None,
                    help="Modelo p/ continuar (ex. models/ram_ppo_100k sem .zip)")
    ap.add_argument("--rom", type=str,
                    default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    ap.add_argument("--state", type=str,
                    default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    ap.add_argument("--geo", action="store_true",
                    help="+6 flags de pit da ROM (obs 17->23)")
    args = ap.parse_args()

    import mlflow
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")
    fns = [make_env(args.rom, args.state, i, geo=args.geo) for i in range(args.num_envs)]
    env = SubprocVecEnv(fns) if args.num_envs > 1 else DummyVecEnv(fns)
    t0 = time.time()
    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_MLP_RAMonly")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("num_envs", args.num_envs)
        mlflow.log_param("geo", args.geo)
        mlflow.log_param("obs", f"Box({env.observation_space.shape[0]}) RAM"
                         + ("+pit6" if args.geo else ""))
        if args.resume and os.path.exists(f"{args.resume}.zip"):
            print(f"Resume de {args.resume}.zip", flush=True)
            model = PPO.load(args.resume, env=env)
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        else:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
            model.learn(total_timesteps=args.timesteps)
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{args.run_id}")
        mlflow.log_artifact(f"models/{args.run_id}.zip")
    dt = time.time() - t0
    print(f"RAM-ONLY: {args.timesteps} steps em {dt:.0f}s ({args.timesteps/dt:.1f} fps)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
