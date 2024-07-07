import torch
from torch.nn.modules import Module, Linear
from torch.nn import Parameter

class ElementwiseMulModule(Module):
    def __init__(self, shape:int|tuple[int], fill=1.) -> None:
        super().__init__()
        self.l = Parameter(torch.empty(shape), requires_grad=True)
        self._shape = shape
        self._initial = fill
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.l, self._initial)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x * self.l

    def extra_repr(self) -> str:
        return f'ElementwiseMulModule(shape={self._shape}, init={self._initial})'