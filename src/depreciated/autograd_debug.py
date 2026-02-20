import torch
from model import CorrectionPotential

model = CorrectionPotential()

R = torch.randn(10, 3, requires_grad=True)
Z = torch.zeros(10, dtype=torch.long)

E = model(R, Z)
print("E:", E)
F = -torch.autograd.grad(E, R, create_graph=True)[0]
print("F:", F)
print("F.requires_grad:", F.requires_grad)
