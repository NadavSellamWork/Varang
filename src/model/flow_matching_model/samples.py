from dataclasses import dataclass
import torch

@dataclass
class Samples:
    def __init__(self, curves: torch.Tensor, domain: str = "time", t: int = 0):
        self.curves = curves
        self.domain = domain
        self.t = t