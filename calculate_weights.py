import numpy as np
import torch

# 1) Your aggregated pixel counts
CLASS_COUNTS = {
    "empty":  389_089_847,
    "soil":   382_035_314,
    "road":    59_665_418,
    "forest": 113_768_472,
    "grass":  687_993_981,
    "house":   33_403_362
}

# 2) Build a numpy array of counts in a fixed order
class_names = list(CLASS_COUNTS.keys())
counts = np.array([CLASS_COUNTS[name] for name in class_names], dtype=np.float64)

# 3) Inverse-frequency weights with epsilon guard
epsilon = 1e-6
inv_freq = 1.0 / (counts + epsilon)

# 4) Normalize to sum = number of classes
num_classes = len(counts)
weights = inv_freq / inv_freq.sum() * num_classes

# 5) Convert to torch tensor
weights_tensor = torch.tensor(weights, dtype=torch.float32)

# 6) Display
print("Class    Pixels        Weight")
print("-" * 30)
for name, cnt, w in zip(class_names, counts, weights_tensor):
    print(f"{name:6} {int(cnt):12,d}    {w:.6f}")

# 7) Save for training
torch.save(weights_tensor, "class_weights.pt")
np.save("class_weights.npy", weights)
print("\nSaved class_weights.pt and class_weights.npy")
