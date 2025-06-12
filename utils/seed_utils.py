# Seeding for Reproducibility
# Seed before Data splitting, Model initialization, DataLoader workers (set worker_init_fn), Training

def set_seed(seed=42):
    random.seed(seed)                          # Python's random
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch (CPU ops)
    torch.cuda.manual_seed(seed)               # PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)           # PyTorch (multi-GPU)

    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior for convs
    torch.backends.cudnn.benchmark = False     # Disables auto-tuner for convs (faster but non-deterministic)

set_seed(42)
