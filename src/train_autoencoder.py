import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from tqdm import tqdm

class MarioAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(MarioAutoencoder, self).__init__()
        
        # Encoder (Matches Stable-Baselines3 Nature CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=0),
            nn.Sigmoid() # Images are 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder_fc(encoded)
        x = x.view(-1, 64, 7, 7)
        decoded = self.decoder_conv(x)
        return decoded, encoded

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder on collected Mario frames")
    parser.add_argument("--dataset", type=str, default="data/mario_dataset.npz", help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found. Run collect_data.py first.")
        return

    print("Loading dataset...")
    data = np.load(args.dataset)["images"]
    
    # Preprocess: shape is (N, 84, 84, 1), convert to (N, 1, 84, 84) and normalize to 0-1
    data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    tensor_data = torch.from_numpy(data)
    dataset = TensorDataset(tensor_data, tensor_data) # Input and Target are the same
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MarioAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_Autoencoder")
    
    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("dataset_size", len(data))
        
        print("Starting training...")
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, leave=False)):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        os.makedirs("models", exist_ok=True)
        model_path = "models/autoencoder.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, artifact_path="models")
        print(f"Autoencoder saved to {model_path}!")

if __name__ == "__main__":
    main()
