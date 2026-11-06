Atividades e projetos realizados ao longo do curso de Machine Learning 1 - IMPA-Tech 2026.2

## Picme 2.2 – Cats vs Dogs

### Finalidade

Classificar imagens de gatos e cachorros com uma CNN inspirada na AlexNet (livro *Understanding Deep Learning*, capítulo 10). O trabalho foi feito para o Picme de **Processamento de Imagens Cardíacas**, para praticar a teoria estudada até agora e motivar uma competição entre os estudantes.

A base de dados é a [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) do Kaggle, com aproximadamente 25 mil imagens.

### Modelo final: `cats_and_dogs_224_v2.ipynb`

**Preparação dos dados**
* Arquivos repetidos são retirados. Imagens quase idênticas ficam sempre do mesmo lado na divisão, para não "vazar" do treino para a validação ou o teste.
* Imagens com algum lado menor que 94 px são descartadas.
* Todas as imagens são redimensionadas para 224 x 224, mantendo a proporção. O lado menor vira 224, o excesso do lado maior é cortado no centro e a interpolação é bilinear.
* A divisão em treino, validação e teste (cerca de 70% / 15% / 15%) é feita **antes** do data augmentation.

**Modelo**
* 5 blocos convolucionais (convolução 5x5 → ReLU → max pooling → dropout 0,2). O número de filtros cresce com a profundidade: 32, 64, 64, 128 e 128.
* Camadas totalmente conectadas: 1024 → 32 → 2 (gato ou cachorro).
* Otimizador Adam.

**Regularização**
* Cópias fixas das imagens de treino, giradas ou com ruído.
* Data augmentation aleatório a cada época, só no treino: espelhamento horizontal, zoom e deslocamento, rotação de -15° a +15°, e variação de brilho e contraste.
* Dropout em todos os blocos convolucionais.

**Resultado:** treino de 120 épocas, com os pesos da melhor época pela acurácia na **validação** (época 112, 96,9%). A acurácia no teste foi de **96,4%**.

### Rede pré-treinada: `cats_and_dogs_resnet.ipynb`

Também testamos uma **ResNet18 já treinada na ImageNet**, com fine-tuning para gatos e cachorros. A última camada foi trocada por uma de 2 saídas e o treino usou taxa de aprendizado pequena. Os dados e o augmentation são os mesmos do modelo final. Com só 5 épocas, o resultado foi melhor ainda: **99,1% na validação e 98,8% no teste**.

### Sobre o conjunto de teste

Todos os notebooks têm uma avaliação no conjunto de teste, mas ela serve **apenas por curiosidade**. O resultado do teste **nunca foi usado para mudar nada no modelo**: arquitetura, hiperparâmetros, regularização e a escolha da melhor época foram decididos só com base no treino e na validação.

### Arquivos da pasta

| Arquivo | O que é |
|---|---|
| `cats_and_dogs.ipynb` | Primeira versão do modelo |
| `cats_and_dogs_224.ipynb` | Versão com imagens 224 x 224 |
| `cats_and_dogs_224_v2.ipynb` | **Modelo final** |
| `cats_and_dogs_resnet.ipynb` | Teste com a ResNet18 pré-treinada |
| `historico_decisoes.ipynb` | Registro das decisões tomadas ao longo do trabalho |
| `historico/` | Históricos de treino (`.json`) e anotações dos ensaios (`.txt`) |
| `modelo_cats_and_dogs_224_v2.pt` | Pesos do modelo final |
| `requirements.txt` | Bibliotecas necessárias (`pip install -r requirements.txt`) |
