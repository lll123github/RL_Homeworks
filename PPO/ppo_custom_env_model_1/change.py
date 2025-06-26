from pyexpat import model
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO

def save_model_weights_txt(model, filename_prefix="ppo_custom_weights"):
    # 获取自定义特征提取器内的模型（你的 Actor 网络）
    actor_model = model.policy.features_extractor.model

    with open(f"{filename_prefix}.txt", "w") as f:
        f.write("Actor Network Weights:\n")
        for i, layer in enumerate(actor_model.children()):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.cpu().numpy()
                f.write(f"Layer {i} weight shape: {weight.shape}\n")
                np.savetxt(f, weight, fmt="%.6f")
                f.write("\n")


model = PPO.load("ppo_custom_env_model_1.oth")
save_model_weights_txt(model)
print("✅ 模型权重已保存至 ppo_custom_weights.txt")