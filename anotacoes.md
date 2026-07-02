## Mudanças


### No VAE


- Adicionar slide de título VAE

- Adicionar os valores testados no grid_search
    - learning rate
        - [1e-2, 1e-3, 1e-4]
    - 2 diferentes arquiteturas
        - [2,3] camadas ocultas na  rede neural
            - mais camadas significa compressão mais suave, mas aumenta complexidade da rede e risco de overfitting
    - uso de kl-anealing  
        - constant β=1  vs  step +0.2 every 5 epochs (0→1)

    - tamanho da dimensão latente
        - [8, 16, 32]


- Alterar imagem do 2D marginal
    - VAE_EMNIST_tunning no final do notebook

- colocar examplos de imagem reconstruída
    - adonai passar assim que terminar treinamento.


### Diffusion

- Adicionar slide de título Diffusion

- Explicar brevemente sobre a arquitetura
    - docs/report/denoiser_arquitetura.md


- Adicionar valores testados no grid search
    - learning_rate [1e-4, 5e-5, 1e-3]
    - tipos de schedulers (linear e cosseno)

- outros hiperparâmetros que não entraram no grid search
    - batch size = 64
    - epocas = 30


- no slide de scheduler 
    - "testamos os schedulers linear e cosseno" 
        - resultado explicar quando ficar pronto

- colocar a equação do sampler no slide 12
    - OK


- imagens para colocar (em results):
    - generated_samples_per_class
    - diffusion_generation_process
    - diffusion_samples


- explicar brevemente o que é a difusão latente condicionada