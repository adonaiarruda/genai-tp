# Revisão da Apresentação — Pendências

Relatório de verificação do `presentation/main.tex` contra os deliverables descritos em
`task_instructions.md` e contra o estado real dos notebooks (`notebooks/VAE_EMNIST_tunning.ipynb`,
`notebooks/Latent_Diffusion_EMNIST.ipynb`, `notebooks/Generate_Samples_EMNIST.ipynb`) e dos
checkpoints entregues (`notebooks/training_checkpoints/vae_best.ckpt` e
`notebooks/training_checkpoints/latent_denoiser_best.ckpt`).

---

## 1. Bug de compilação (crítico)

O slide 7 (`Curvas de Custo por Pixel`) referencia:

```latex
\includegraphics[width=0.75\textwidth]{likelihood_cost_curves.png}
```

Mas o arquivo presente em `presentation/` se chama `likelihood_cost_curves (1).png`
(nome com espaço e sufixo `(1)`). **O LaTeX não compila como está.**

- **Ação:** renomear o arquivo para `likelihood_cost_curves.png` (ou ajustar o `\includegraphics`
  para o nome atual — não recomendado, por causa do espaço no nome do arquivo).

---

## 2. Números incorretos em relação aos checkpoints entregues

Comparação dos valores citados no `.tex` com o que está de fato salvo nos `.ckpt` (fonte de
verdade, pois são os artefatos que serão avaliados):

| Item | O que o `.tex` diz | O que o checkpoint / notebook mostra | Fonte |
|---|---|---|---|
| **KL Annealing (slides 8–9)** | "O modelo com annealing superou a baseline constante ao atingir menor ELBO de teste" | `best_config` salvo no `vae_best.ckpt` tem `kl_anneal_epochs: 0` → o **Stage 2** da busca sequencial de hiperparâmetros (`VAE_EMNIST_tunning.ipynb`, célula "Stage 2 — KL annealing", comparando `constant` vs `step0.2_per5ep`) **escolheu o β=1 constante**, não o annealing em degraus | `vae_best.ckpt['best_config']` |
| **Learning rate do VAE** | Não citado explicitamente (só o LR do denoiser) | `learning_rate: 0.001` (vencedor do Stage 4 da busca) | `vae_best.ckpt['best_config']` |
| **Épocas do Denoiser (slide 12)** | "Treinado por 10 épocas" | `epoch: 30`, `NUM_EPOCHS = 30` no notebook, 30 valores de loss registrados no checkpoint | `latent_denoiser_best.ckpt` |
| **Latent dim / arquitetura (slide 5)** | `m=32`, 3 blocos convolucionais | Confere: `latent_dim=32`, `arch='3blocks'` | `vae_best.ckpt` ✅ (correto) |

⚠️ **O ponto do KL Annealing é uma contradição direta com o artefato entregue** — o slide conta
uma história (annealing venceu) que é o oposto do que o `best_config` registra (constante venceu).
Isso precisa ser corrigido no texto, ou os autores precisam reconferir o resultado do Stage 2 do
tuning antes da apresentação (pode haver sido uma mudança posterior não refletida no notebook, mas
o checkpoint que será avaliado tem `kl_anneal_epochs: 0`).

---

## 3. Deliverable ausente — VAE: training details

O enunciado (`task_instructions.md`) pede explicitamente, para o VAE:
> Training details (learning rate, epochs, batch size)

Isso **não existe em nenhum slide** — o slide 12 tem esses detalhes só para o Denoiser.

- **Valores a usar** (de `VAE_EMNIST_tunning.ipynb` + `vae_best.ckpt`):
  - `latent_dim = 32`
  - `arch = 3blocks`
  - `learning_rate = 0.001`
  - `batch_size = 256`
  - Treinado por **77 épocas** (parou por *early stopping*, não pelo teto configurado de `EPOCHS`)
  - Reconstrução: `Bernoulli` (já coberto no slide 6)

---

## 4. Deliverable ausente — VAE: exemplos de reconstrução e geração

O enunciado pede explicitamente:
> Generation and reconstruction examples

**Não existe nenhuma imagem desse tipo em `presentation/`.** Os notebooks (`VAE_EMNIST.ipynb` e
`VAE_EMNIST_tunning.ipynb`) já têm a função `show_examples()` que mostra 3 originais, 3
reconstruções e 3 amostras geradas de `z ~ N(0, I)` — mas essa função só faz `fig.show()`
(exibição inline durante o treino), **nunca salva a figura em disco**.

- **Ação:** rodar novamente essa célula (ou adaptar `show_examples` para also fazer
  `fig.write_image(...)`) carregando o `vae_best.ckpt`, salvar um PNG em `notebooks/results/`, e
  incluir no `.tex` em um novo slide.

---

## 5. Deliverable ausente — Diffusion: scheduler e sampler

O enunciado pede explicitamente:
> The chosen scheduler. The chosen sampler.

**Nenhum dos dois é mencionado no `.tex`.** O notebook define isso claramente:

- **Scheduler:** `linear_beta_schedule(timesteps)`, com `TIMESTEPS = 1000`,
  `beta_start ≈ 0.0001`, `beta_end ≈ 0.02`. Existe também uma `cosine_beta_schedule` implementada
  no notebook, mas **não é a usada** (está comentada / não ativada).
- **Sampler:** processo reverso **determinístico estilo DDIM (η = 0)** — funções `p_sample` /
  `p_sample_loop`. Não há injeção de ruído estocástico; $z_{t-1}$ é obtido deterministicamente a
  partir do $x_0$ previsto e do ruído previsto pelo modelo (diferente do sampler ancestral
  estocástico do DDPM padrão).

- **Ação:** adicionar essas informações ao slide 12 (`Modelo de Difusão: O LatentDenoiser`) ou
  criar um slide dedicado.

---

## 6. Deliverable ausente — Diffusion: plots de loss e snapshots de geração

O enunciado pede explicitamente:
> Plots of the training error as a function of the number of steps.
> Snapshots of the Generation

Essas imagens **já foram geradas** pelo notebook e estão em `notebooks/results/`, mas **não foram
copiadas para `presentation/` nem referenciadas no `.tex`**:

| Arquivo (em `notebooks/results/`) | Cobre qual deliverable |
|---|---|
| `diffusion_loss_curve.png` | Plot de erro de treino do Denoiser por época |
| `diffusion_samples.png` | Snapshots da geração (16 amostras aleatórias) |
| `diffusion_samples_all_classes.png` | Snapshots da geração condicionada — uma amostra por classe EMNIST (62 classes) |
| `diffusion_generation_process.png` | Bônus: evolução do processo de denoising de $t=T$ (ruído puro) até $t=0$ (amostra final), reforça a explicação do sampler DDIM determinístico |

- **Ação:** copiar esses 4 arquivos para `presentation/` e adicionar slides referenciando-os
  (ao menos os dois primeiros são obrigatórios pelo enunciado; os outros dois são bônus fortes).

---

## 7. Detalhe de arquitetura ausente — geração condicionada por classe

O `LatentDenoiser` (em `notebooks/Latent_Diffusion_EMNIST.ipynb`) tem um `nn.Embedding` de rótulo
(`label_emb`) e **todas as amostras são geradas condicionadas à classe** (dígito ou letra
específica), via `conditioning_labels` passado a `generate_samples`. Essa é uma característica
central da arquitetura implementada (permite pedir "gere um 'A'" especificamente) e **não é
mencionada em nenhum slide** — nem no slide 12 (arquitetura), nem no slide 13 (pipeline de
geração).

- **Ação:** adicionar uma linha sobre o condicionamento por classe no slide 12, e mencionar no
  pipeline do slide 13 que o rótulo desejado é uma entrada do processo, não só o ruído.

---

## Resumo — checklist do que falta adicionar/corrigir no `main.tex`

- [ ] Corrigir nome de arquivo quebrado (`likelihood_cost_curves.png`)
- [ ] Corrigir slide 9: KL annealing **não** venceu no checkpoint final (`kl_anneal_epochs=0`)
- [ ] Corrigir slide 12: Denoiser treinado por **30** épocas, não 10
- [ ] Novo slide: training details do VAE (LR=0.001, batch=256, 77 épocas, latent=32, 3blocks)
- [ ] Novo slide: exemplos de reconstrução e geração do VAE (imagem ainda precisa ser gerada)
- [ ] Slide 12 (ou novo): scheduler = linear beta schedule, T=1000
- [ ] Slide 12 (ou novo): sampler = DDIM determinístico (η=0)
- [ ] Novo slide: curva de loss do diffusion (`diffusion_loss_curve.png`)
- [ ] Novo slide: amostras geradas do diffusion (`diffusion_samples.png` e/ou
      `diffusion_samples_all_classes.png`)
- [ ] Opcional: slide com `diffusion_generation_process.png` (processo de denoising passo a passo)
- [ ] Mencionar geração condicionada por classe (label embedding) na arquitetura/pipeline
