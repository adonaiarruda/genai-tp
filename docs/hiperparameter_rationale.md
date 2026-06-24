# Justificativa da escolha dos hiperparâmetros — VAE EMNIST

Documento de apoio para o projeto **Latent Diffusion on EMNIST**
(etapa do VAE, notebook `notebooks/VAE_EMNIST.ipynb`).

Os hiperparâmetros estão definidos na célula 4 do notebook:

```python
LATENT_DIM        = 64
BATCH_SIZE        = 256
EPOCHS            = 60
LEARNING_RATE     = 1e-3
USE_DIFF_SIGMA_E  = True
RECONSTRUCTION    = 'Bernoulli'
EMNIST_SPLIT      = 'byclass'
```

Atualmente eles são **fixados manualmente**, sem busca/validação automática.
As justificativas abaixo são baseadas em convenções de boas práticas,
restrições de hardware e na teoria do modelo.

---

## `LEARNING_RATE = 1e-3`

- Valor *default* clássico do otimizador **Adam**.
- Funciona de forma robusta para VAEs convolucionais pequenos.
- Não houve *tuning* empírico; é a escolha padrão "que quase sempre funciona".

**Trade-off:** lr maior acelera mas pode desestabilizar o termo KL;
lr menor é mais estável porém exige mais épocas.

---

## `BATCH_SIZE = 256`

- Escolha guiada por **memória de GPU e velocidade**, não por qualidade.
- EMNIST `byclass` tem ~698k imagens de treino; batch grande reduz o número
  de passos por época e produz gradientes mais estáveis.
- Em GPU T4 (16 GB) um batch de 256 cabe com folga.

**Trade-off:** batch maior = menos passos de atualização por época
(interage diretamente com a escolha de `EPOCHS` e `LEARNING_RATE`).

---

## `EPOCHS = 60`

- Escolha por **orçamento de tempo de treino**, não por critério de convergência.
- Não há *early stopping*; é um número redondo.
- Como o dataset é grande (~698k imagens), cada época já equivale a
  ~2700 passos de gradiente, então 60 épocas representam treino substancial.

**Recomendação:** validar olhando a curva de `test_loss` (célula 21) e parar
quando estabiliza. Idealmente adicionar *early stopping*.

---

## `LATENT_DIM = 64`

- Compromisso entre **capacidade de reconstrução** e o requisito do enunciado
  de um *"much smaller latent space"*.
- Pequeno o suficiente para a etapa de difusão/flow operar num espaço tratável;
  grande o suficiente para reconstruir razoavelmente as 62 classes.
- **Não foi justificado por experimento** (não há comparação 16 vs 32 vs 64).

**Ponto de atenção:** este é o hiperparâmetro mais questionável para a
apresentação. Uma ablação de `LATENT_DIM` deixaria a escolha mais defensável.

---

## `RECONSTRUCTION = 'Bernoulli'`

- Escolha **teoricamente motivada**.
- As imagens EMNIST, via `ToTensor()`, ficam normalizadas em `[0,1]`, e cada
  pixel é tratado como probabilidade → a likelihood **Bernoulli** (BCE) é o
  modelo natural para dados em `[0,1]`.
- É coerente com a ativação `Sigmoid` na saída do decoder.
- As alternativas (`Gaussian` → MSE, `Laplacian` → MAE) existem no código,
  mas Bernoulli é a mais adequada à natureza dos dados.

---

## `USE_DIFF_SIGMA_E = True`

- Usa um `sigma` (variância) **diferente por dimensão** no posterior `q(z|x)`,
  em vez de um `sigma` escalar compartilhado.
- Encoder mais expressivo: cada dimensão latente tem sua própria variância.
- Maior flexibilidade na representação aprendida.

---

## `EMNIST_SPLIT = 'byclass'`

- **Imposto pelo enunciado.**
- 62 classes (0-9, A-Z, a-z): inclui dígitos e letras, configurando o caso
  "mais desafiador que MNIST" pedido na tarefa.

---

## Resumo crítico

Não existe um processo formal de seleção de hiperparâmetros no notebook.
Os valores são **defaults sensatos + restrições de hardware + teoria do
likelihood**, e não resultado de busca sistemática. Em particular, faltam:

- *grid/random search*;
- conjunto de **validação** separado para *tuning* (atualmente o `test_loss`
  é usado como se fosse validação);
- ablação de `LATENT_DIM` e do tipo de likelihood;
- *early stopping*.

### Melhorias sugeridas para robustez

1. Ablação de `LATENT_DIM` (16 / 32 / 64) comparando reconstrução + KL.
2. *Early stopping* baseado em `test_loss`.
3. Separação real train / val / test (não usar o test para *tuning*).
