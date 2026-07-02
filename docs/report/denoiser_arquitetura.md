# Arquitetura do Denoiser — Latent Diffusion EMNIST

Documento breve sobre a classe `LatentDenoiser` (célula 12 de `Latent_Diffusion_EMNIST.ipynb`).

## O que ele faz

O denoiser **não** opera em pixels, e sim no espaço latente do VAE (encoder congelado). Cada
imagem (784 pixels) é codificada em um vetor `z_0` de dimensão `LATENT_DIM = 32`. Dado um latente
ruidoso `z_t`, um timestep `t` e um rótulo de classe, o modelo **prevê o ruído ε** adicionado
(objetivo *ε-prediction*, treinado com MSE). Como os latentes são vetores 1-D, a rede é uma **MLP**
— não há estrutura espacial que justifique convoluções.

## Entradas e fusão

| Entrada | Processamento | Dim. |
|---|---|---|
| `z` (latente ruidoso) | usado direto | 32 |
| `t` (timestep) | embedding sinusoidal → `time_mlp` (256→1024→256) | 256 |
| `labels` (classe) | `nn.Embedding(62, 64)` | 64 |

Os três são **concatenados** (32+256+64) e projetados por `proj_in` para `hidden_dim`.

- **Time embedding sinusoidal**: representação suave e contínua do passo de difusão, generaliza
  entre os 1000 timesteps sem embedding treinável por passo.
- **Label embedding**: habilita a **geração condicional** das 62 classes do EMNIST `byclass`
  (0-9, A-Z, a-z).

## Topologia: "U-Net em MLP"

Imita a topologia de uma U-Net (multiescala + skips), mas com camadas lineares.
Com `hidden_dim = LATENT_DIM * 3 = 96`:

```
proj_in ................. 96
 ├─ down1: 96  → 192   ─┐ (skip1)
 ├─ down2: 192 → 384   ─┤ (skip2)
 ├─ mid:   384 (bottleneck)
 ├─ up1:   [384 ‖ skip2=384] → 192
 └─ up2:   [192 ‖ skip1=192] → 96
proj_out ................ 32  (ε previsto)
```

| Camada | Dimensão de saída |
|---|---|
| `proj_in` | 96 (`hidden`) |
| `down1` | 192 (`2·hidden`) |
| `down2` / `mid` | 384 (`4·hidden`) |
| `up1` | 192 |
| `up2` | 96 |
| `proj_out` | 32 (`latent_dim`) |

### Escolhas de projeto e motivos

1. **Largura cresce e afunila** (`96 → 192 → 384 → 192 → 96`): o gargalo largo (`mid`) dá capacidade
   para modelar a função de denoising; a subida reconstrói o latente.
2. **Skip connections**: saídas de `down1`/`down2` são concatenadas na subida, preservando
   informação que se perderia no gargalo (mesma razão das skips da U-Net original).
3. **Conexões residuais por bloco** (`downN(x) + down_resN(h)`): como a dimensão muda a cada bloco,
   o caminho residual usa uma projeção linear para casar shapes. Estabilizam o gradiente — crítico,
   pois a rede é chamada 1000× durante o sampling.
4. **GELU** em todos os blocos: ativação suave, padrão em modelos de difusão/transformers.
5. **`hidden_dim = LATENT_DIM * 3`**: capacidade escalada em função da dimensão latente.

## Por que essa arquitetura faz sentido

- **Latent diffusion**: rodar difusão no latente pequeno do VAE é muito mais barato que em pixels
  (ideia do Stable Diffusion). Latente vetorial ⇒ MLP é a escolha natural.
- **U-Net MLP** reaproveita o que funciona em difusão (multiescala, skips, residuais, time embedding),
  adaptado a dados 1-D.
- **Condicionamento por classe** permite geração controlada das 62 classes.
- **ε-prediction + schedule linear de 1000 passos** (receita clássica DDPM); o sampler é
  **DDIM determinístico** (η=0), com trajetória reproduzível e menos ruído.
