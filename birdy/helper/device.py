import torch


def get_best_device():
    """
    Docstring for get_best_device
    """

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("[@] Using Intel XPU")
        return torch.device("xpu")

    if torch.cuda.is_available():
        print("[@] Using CUDA GPU")
        return torch.device("cuda")

    print("[!] Using CPU")
    return torch.device("cpu")
