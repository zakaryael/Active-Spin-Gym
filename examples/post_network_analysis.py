from stable_baselines3 import PPO
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plt.xkcd()  # Enable XKCD style

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model(model_path):
    model = PPO.load(model_path)

    print("Model Architecture:")
    print(model.policy)
    print(f"\nTotal trainable parameters: {count_parameters(model.policy)}")

    print("\nDetailed Layer Analysis:")
    total_connections = 0

    for name, module in model.policy.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None

            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")
            print(f"Weight shape: {weight.shape}")
            
            # Calcul des statistiques seulement si le tenseur n'est pas vide
            if weight.numel() > 0:
                print(f"Weight stats: mean={weight.mean().item():.4f}, std={weight.std().item():.4f}, "
                      f"min={weight.min().item():.4f}, max={weight.max().item():.4f}")

            if isinstance(module, torch.nn.Linear):
                print(f"Neurons: input={weight.shape[1]}, output={weight.shape[0]}")
                connections = weight.shape[0] * weight.shape[1]
                total_connections += connections
                print(f"Connections: {connections}")

            if isinstance(module, torch.nn.Conv2d):
                print(f"Filters: {weight.shape[0]}, Channels: {weight.shape[1]}")
                connections = weight.shape[0] * weight.shape[1] * weight.shape[2] * weight.shape[3]
                total_connections += connections
                print(f"Connections: {connections}")

            if bias is not None:
                print(f"Bias shape: {bias.shape}")
                if bias.numel() > 0:
                    print(f"Bias stats: mean={bias.mean().item():.4f}, std={bias.std().item():.4f}, "
                          f"min={bias.min().item():.4f}, max={bias.max().item():.4f}")

            # Visualize weight distribution
            plt.figure(figsize=(10, 5))
            sns.histplot(weight.cpu().numpy().flatten(), kde=True)
            plt.title(f"Weight Distribution: {name}")
            plt.savefig(f"weight_dist_{name.replace('.', '_')}.png")
            plt.close()

            # Visualize weight matrix/filters
            if len(weight.shape) == 2:  # Linear layer
                plt.figure(figsize=(10, 10))
                sns.heatmap(weight.cpu().numpy(), cmap='viridis', center=0)
                plt.title(f"Weight Matrix: {name}")
                plt.savefig(f"weight_matrix_{name.replace('.', '_')}.png")
                plt.close()
            elif len(weight.shape) == 4:  # Conv layer
                fig, axes = plt.subplots(8, 8, figsize=(20, 20))
                for i, ax in enumerate(axes.flat):
                    if i < weight.shape[0]:
                        ax.imshow(weight[i, 0].cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                plt.suptitle(f"Conv Filters: {name}")
                plt.savefig(f"conv_filters_{name.replace('.', '_')}.png")
                plt.close()

    print(f"\nTotal connections in the network: {total_connections}")
# Use the function
# take the model path from the command line
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the model architecture and weights.")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file.")
    args = parser.parse_args()
    #concatenate the model path with the model folder
    args.model_path = os.path.join(args.model_path, "model")
    analyze_model(args.model_path)