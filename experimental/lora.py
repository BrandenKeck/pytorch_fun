import torch
import torch.nn as nn

###
# https://www.linkedin.com/pulse/more-efficient-finetuning-implementing-lora-from-scratch-george-davis
###


# let's start building out a LoRALinear layer
class LoRALinear(nn.Module):
  """
  This is a low-rank adapted linear layer that can be used to replace a standard linear layer.
  
  
  Args:
    module: The linear layer module to adapt.
    rank: The rank of the approximation.
    alpha: The alpha parameter.
  """

  def __init__(
    self,
    module: nn.Module,
    # in_dim: int,
    # out_dim: int,
    rank: int = 4,
    alpha: float = 4.0
  ):
    # ensure the module is a linear layer
    assert isinstance(module, nn.Linear), "Module must be a linear layer."

    super().__init__()
    self.rank = rank # rank of the approximation
    self.alpha = alpha # alpha parameter
    self.scaling = self.alpha / self.rank # scaling factor
    self.in_dim = module.in_features # number of input features
    self.out_dim = module.out_features # number of output features

    # make sure that rank is at least 1
    assert self.rank >= 1, "Rank must be at least 1."

    # recreate the linear layer and freeze it
    # note: we will copy over the pretrained weights after initializing
    self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=True)
    self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
    self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
    self.pretrained.weight.requires_grad = False # freeze the weights
    self.pretrained.bias.requires_grad = False # freeze the bias

    # create the A and initialize with Kaiming
    self.A = nn.Linear(self.in_dim, rank, bias=False)
    nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))

    # create B and initialize with zeros
    self.B = nn.Linear(rank, self.out_dim, bias=False)
    nn.init.zeros_(self.B.weight)

    # ensure that the weights in A and B are trainable
    self.A.weight.requires_grad = True
    self.B.weight.requires_grad = True

  def forward(self, x: torch.Tensor):
    """
    Perform the forward pass of the layer.
    
    Args:
      x: The input tensor.
    """
    pretrained_out = self.pretrained(x) # get the pretrained weights
    lora_out = self.A(x) # 
    lora_out = self.B(lora_out)
    lora_out = lora_out * self.scaling
    return pretrained_out + lora_out       
