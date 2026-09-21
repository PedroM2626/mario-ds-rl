import os
import mlflow

def main():
    # Set tracking URI and experiment name
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")

    checkpoint_path = "logs/runs/dreamer_v3/MarioNDS-Dreamer-v0/2026-06-04_14-35-35_dreamer_v3_MarioNDS-Dreamer-v0_42/version_0/checkpoint/ckpt_1000_0.ckpt"
    video_path = "logs/runs/dreamer_v3/MarioNDS-Dreamer-v0/2026-06-04_14-35-35_dreamer_v3_MarioNDS-Dreamer-v0_42/version_0/evaluation/version_1/test_videos/rl-video-episode-0.mp4"
    
    print("Logging evaluation details to MLflow...")
    with mlflow.start_run(run_name="dreamer_v3_evaluation_stochastic"):
        # Log MLOps metadata and parameters
        mlflow.log_param("model_type", "DreamerV3")
        mlflow.log_param("evaluation_mode", "Stochastic")
        mlflow.log_param("checkpoint_path", checkpoint_path)
        mlflow.log_param("trained_steps", 1000)
        
        # Log evaluation reward metric
        mlflow.log_metric("eval_reward", 231.20)
        
        # Log gameplay video artifact
        if os.path.exists(video_path):
            mlflow.log_artifact(video_path, artifact_path="videos")
            print(f"Logged video artifact: {video_path}")
        else:
            print(f"Warning: Video not found at {video_path}")
            
        # Log model checkpoint artifact
        if os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, artifact_path="models")
            print(f"Logged model checkpoint artifact: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            
    print("MLflow logging complete.")

if __name__ == "__main__":
    main()
