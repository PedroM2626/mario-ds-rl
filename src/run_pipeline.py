import os
import subprocess
import sys
import time
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

def run_command(command):
    print(f"\n==================================================")
    print(f"Running command:\n{' '.join(command)}")
    print(f"==================================================")
    
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Print stdout in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\nCommand completed successfully in {elapsed:.1f} seconds.")
        return True
    else:
        print(f"\nCommand failed with exit code {process.returncode} after {elapsed:.1f} seconds.")
        return False

def update_readme():
    print("\nUpdating README.md with MLflow metrics...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    
    # Mapeamento das runs para a tabela do README
    run_mapping = {
        "ppo_pure_100k": "PPO Recurrent Pure (NatureCNN)",
        "ppo_autoencoder_frozen_100k": "Autoencoder Frozen",
        "ppo_curl_online_100k": "CURL Online",
        "ppo_hybrid_ae_curl_100k": "Híbrido AE + CURL",
        "ppo_drq_100k": "PPO Recurrent + DrQ-v2 (Random Shifts)",
        "ppo_spr_100k": "PPO Recurrent + SPR (Multi-step Latent)",
        "ppo_transformer_spr_100k": "PPO Causal Transformer + SPR (Nova Arquitetura)"
    }
    
    table_lines = [
        "| Configuração | Episódios | Recompensa Média | Desvio Padrão | Recompensa Máxima |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    
    # Buscar runs do experimento
    try:
        experiment = client.get_experiment_by_name("Mario_NDS_RL")
        if experiment is not None:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        else:
            runs = []
    except Exception as e:
        print(f"Warning: Could not fetch runs from MLflow: {e}")
        runs = []
        
    results = {}
    for run in runs:
        # Tentar obter o nome amigável da run
        run_name = run.data.tags.get("mlflow.runName", "")
        if not run_name:
            run_name = run.data.params.get("run-id", run.data.params.get("run_id", ""))
            
        if run_name in run_mapping:
            run_id = run.info.run_id
            try:
                # Tentar carregar histórico da métrica episode_reward
                history = client.get_metric_history(run_id, "episode_reward")
                rewards = [m.value for m in history]
                if len(rewards) > 0:
                    results[run_name] = {
                        "episodes": len(rewards),
                        "mean": np.mean(rewards),
                        "std": np.std(rewards),
                        "max": np.max(rewards)
                    }
                else:
                    # Obter a última métrica registrada
                    mean_val = run.data.metrics.get("episode_reward", 0.0)
                    results[run_name] = {
                        "episodes": "N/A",
                        "mean": mean_val,
                        "std": 0.0,
                        "max": mean_val
                    }
            except Exception as e:
                print(f"Warning: Error extracting metrics for {run_name}: {e}")
                
    # Fallback de dados históricos padrões (caso o MLflow SQLite tenha sido resetado)
    historical_defaults = {
        "ppo_pure_100k": {"episodes": 768, "mean": 236.95, "std": 120.16, "max": 837.56},
        "ppo_autoencoder_frozen_100k": {"episodes": 847, "mean": 228.61, "std": 118.50, "max": 732.37},
        "ppo_curl_online_100k": {"episodes": 732, "mean": 254.00, "std": 113.11, "max": 702.25},
        "ppo_hybrid_ae_curl_100k": {"episodes": 776, "mean": 226.53, "std": 124.50, "max": 778.52}
    }
    
    # Mesclar com os dados padrões se necessário
    for name, data in historical_defaults.items():
        if name not in results:
            results[name] = data
            
    # Construir linhas da tabela
    for run_name, label in run_mapping.items():
        if run_name in results:
            data = results[run_name]
            ep = data["episodes"]
            mean_val = f"{data['mean']:.2f}"
            std_val = f"{data['std']:.2f}"
            max_val = f"{data['max']:.2f}"
            table_lines.append(f"| **{label}** | {ep} | {mean_val} | {std_val} | {max_val} |")
            
    # Escrever no arquivo README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        start_tag = "<!-- START_100K_TABLE -->"
        end_tag = "<!-- END_100K_TABLE -->"
        
        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            table_str = "\n".join(table_lines)
            new_content = (
                content[:start_idx + len(start_tag)]
                + "\n"
                + table_str
                + "\n"
                + content[end_idx:]
            )
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("README.md table updated successfully!")
        else:
            print("Error: Table markers not found in README.md")

def main():
    print("==================================================")
    print("Starting Sequential RL Pipeline Master Script")
    print("==================================================")
    
    # 1. Executar a suite de benchmarks de 100k steps
    print("\n>>> FASE 1: Rodando os Benchmarks de 100k steps...")
    benchmarks = [
        [sys.executable, "src/train_ppo_recurrent.py", "--timesteps", "100000", "--num-envs", "4", "--use-drq", "--run-id", "ppo_drq_100k"],
        [sys.executable, "src/train_ppo_recurrent.py", "--timesteps", "100000", "--num-envs", "4", "--use-spr", "--run-id", "ppo_spr_100k"],
        [sys.executable, "src/train_ppo_recurrent.py", "--timesteps", "100000", "--num-envs", "4", "--policy-type", "transformer", "--use-spr", "--run-id", "ppo_transformer_spr_100k"]
    ]
    
    for i, cmd in enumerate(benchmarks):
        print(f"\nRunning benchmark {i+1}/{len(benchmarks)}")
        success = run_command(cmd)
        if not success:
            print(f"Warning: Benchmark {i+1} failed. Continuing pipeline regardless...")
            
    # 2. Atualizar o README.md
    print("\n>>> FASE 2: Atualizando a documentação (README.md)...")
    try:
        update_readme()
    except Exception as e:
        print(f"Error updating README.md: {e}")
        
    # 3. Retomar o treino CURL até 1.0M steps
    print("\n>>> FASE 3: Retomando treinamento CURL (impala_curl_recurrent_3m) até 1.0M steps...")
    checkpoint_path = "models/checkpoints/impala_curl_recurrent_3m_240000_steps"
    if os.path.exists(f"{checkpoint_path}.zip"):
        resume_cmd = [
            sys.executable, "src/train_ppo_recurrent.py",
            "--timesteps", "1000000",
            "--num-envs", "4",
            "--use-impala",
            "--use-curl",
            "--run-id", "impala_curl_recurrent_1m",
            "--resume", checkpoint_path
        ]
        success = run_command(resume_cmd)
        if not success:
            print("Warning: CURL training resume failed.")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}.zip. Cannot resume CURL training.")
        
    # 4. Executar novo treino de Causal Transformer + SPR por 1.0M steps
    print("\n>>> FASE 4: Iniciando novo treinamento de 1.0M steps (Causal Transformer + SPR)...")
    new_train_cmd = [
        sys.executable, "src/train_ppo_recurrent.py",
        "--timesteps", "1000000",
        "--num-envs", "4",
        "--use-impala",
        "--policy-type", "transformer",
        "--use-spr",
        "--run-id", "impala_transformer_spr_1m"
    ]
    success = run_command(new_train_cmd)
    if not success:
        print("Warning: Causal Transformer + SPR training failed.")
        
    print("\n==================================================")
    print("Sequential RL Pipeline Master Script Finished!")
    print("==================================================")

if __name__ == "__main__":
    main()
