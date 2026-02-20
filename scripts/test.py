from torchmdnet.datasets import MD17
import inspect

# Check what MD17 actually expects
sig = inspect.signature(MD17.__init__)
print("MD17.__init__ parameters:")
for param in sig.parameters.values():
    print(f"  {param.name}: {param.default if param.default != inspect.Parameter.empty else 'required'}")