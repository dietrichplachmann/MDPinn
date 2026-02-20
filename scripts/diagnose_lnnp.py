#!/usr/bin/env python
"""
Diagnostic: Understand LNNP's step() method behavior
"""

import torch
import inspect
from torchmdnet.module import LNNP

print("="*60)
print("LNNP.step() Source Code")
print("="*60)
print(inspect.getsource(LNNP.step))

print("\n" + "="*60)
print("LNNP.training_step() Source Code")
print("="*60)
print(inspect.getsource(LNNP.training_step))

print("\n" + "="*60)
print("LNNP.validation_step() Source Code")
print("="*60)
print(inspect.getsource(LNNP.validation_step))

print("\n" + "="*60)
print("What step() returns:")
print("="*60)
print("Check the source above - does it return a scalar loss or a dict?")