# Introduction to Generative AI — Practical Test

## Environment setup

```bash
# Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the dependencies (PyTorch with CUDA 12.9 + notebook libraries)
pip install -r requirements.txt
```

> If your system lacks the `venv` module (missing `ensurepip` error), create the
> environment with `python3 -m venv .venv --without-pip` — pip is included.


## Using the GPU in your code

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyModel().to(device)   # move the model to the GPU
x = data.to(device)            # move the tensors to the GPU
```
