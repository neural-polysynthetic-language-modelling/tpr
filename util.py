import torch


def get_device() -> torch.device:

    if torch.cuda.is_available():
        import GPUtil
        best_gpu = GPUtil.getAvailable(order="memory")[-1]
        return torch.device(f"cuda:{best_gpu}")
    else:
        return torch.device("cpu")
