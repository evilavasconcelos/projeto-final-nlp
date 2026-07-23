# Fine-Tuning de Transformers para Análise de Sentimentos: DistilBERT vs RoBERTa

Este repositório contém a implementação e a análise comparativa de modelos de Processamento de Linguagem Natural (NLP) baseados na arquitetura Transformer para a tarefa de classificação binária de sentimentos utilizando o dataset IMDB. 

Desenvolvido como projeto prático no curso de Ciência da Computação da Universidade Federal do Ceará (UFC - Campus Quixadá), o foco deste trabalho é avaliar o trade-off entre força bruta computacional e otimização de engenharia no treinamento de redes neurais profundas.

## Objetivo do Projeto
O principal objetivo não é apenas treinar um modelo, mas analisar o comportamento da rede neural sob diferentes estratégias de otimização e alocação de hardware. O projeto contrasta dois cenários:
* **RoBERTa:** Focado em precisão máxima e compreensão semântica profunda (12 camadas).
* **DistilBERT:** Focado em eficiência de infraestrutura via Knowledge Distillation (6 camadas), buscando agilidade para ambientes de produção.

## Estrutura dos Experimentos
A fim de gerenciar a memória da GPU (NVIDIA T4) e evitar o Esquecimento Catastrófico (Catastrophic Forgetting), os treinamentos foram divididos em diferentes fases de congelamento de tensores:

* **Experimento 1 (Camadas Congeladas):** Apenas a camada final de classificação foi treinada. Foco na redução do tempo de execução e menor consumo de VRAM.
* **Experimento 2 (Descongelamento Parcial):** Liberação das camadas finais do modelo para ajuste fino de sintaxe específica do dataset.
* **Experimento 3 (Fine-Tuning Completo):** Todas as matrizes liberadas para atualização de gradientes, adaptando o modelo totalmente ao vocabulário de cinema.
* **Avaliação de Épocas:** Variação do tempo de treinamento (2, 3 e 5 épocas) para monitorar as taxas de erro e evitar o efeito de overfitting no conjunto de validação.

## Stack Tecnológica
* **Linguagem:** Python
* **Machine Learning Backend:** PyTorch (Gerenciamento de tensores, cálculo de gradientes via Autograd e comunicação com a arquitetura CUDA).
* **NLP Framework:** Hugging Face `transformers` e `datasets`.
* **Hardware:** Execução paralela em GPU.

## Obtenção dos Dados (Dataset)
Por questões de boas práticas de versionamento e limites de armazenamento do GitHub, o arquivo CSV original não está incluído neste repositório. 

O projeto utiliza o **IMDB Dataset of 50K Movie Reviews**. Para executar o código localmente ou na nuvem:
1. Faça o download do dataset oficial: [IMDB Dataset no Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Extraia o arquivo baixado.
3. Posicione o arquivo `IMDB Dataset.csv` na raiz do projeto (ou faça o upload manual na aba lateral caso esteja utilizando o Google Colab).
