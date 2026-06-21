# Introduction to Generative AI Practical test

## Requisitos

- GPU NVIDIA (testado em RTX 3060, 12 GB) com driver instalado — verifique com `nvidia-smi`
- Python 3.14
- **Não** é necessário instalar o CUDA Toolkit: as libs do CUDA já vêm dentro dos wheels do PyTorch.

## Configuração do ambiente

```bash
# Criar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Atualizar o pip
pip install --upgrade pip

# Instalar as dependências (PyTorch com CUDA 12.9 + libs do notebook)
pip install -r requirements.txt
```

> Se você não tem o módulo `venv` do sistema (erro de `ensurepip` ausente), crie o
> ambiente com `python3 -m venv .venv --without-pip` — o pip já vem incluído.

## Testar a GPU

```bash
source .venv/bin/activate
python cuda_test/test_cuda.py
```

Saída esperada (exemplo na RTX 3060):

```
PyTorch versão:      2.12.1+cu129
CUDA disponível:     True
GPU 0:               NVIDIA GeForce RTX 3060
...
 GPU foi ~4.1x mais rápida que a CPU.
```

## Usando a GPU no seu código

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

modelo = MeuModelo().to(device)   # move o modelo para a GPU
x = dados.to(device)              # move os tensores para a GPU
```
