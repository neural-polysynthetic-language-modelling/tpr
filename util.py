import torch


def get_device() -> torch.device:

    if torch.cuda.is_available():
        import os
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >cuda")
        memory_available = [int(x.split()[2]) for x in open("cuda", "r").readlines()]
        best_mem = -1
        best_gpu = -1
        for gpu, gpu_mem in enumerate(memory_available):
            if gpu_mem > best_mem:
                best_mem = gpu_mem
                best_gpu = gpu
        return torch.device(f"cuda:{best_gpu}")
    else:
        return torch.device("cpu")
