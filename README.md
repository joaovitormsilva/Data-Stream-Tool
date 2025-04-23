# 🌀  Projeto: Geração e Classificação de Fluxos de Dados Gaussianos Interativos

Este projeto permite criar manualmente trajetórias no espaço 2D, que são utilizadas como base para gerar fluxos de dados sintéticos com distribuição Gaussiana, associando-os a diferentes subfluxos (clusters) com base em sua proximidade espacial e temporal. A ferramenta conta com uma interface gráfica interativa implementada com Matplotlib.

## ✨ Funcionalidades

    ✏️ Desenho Interativo de Trajetórias: Clique e arraste para desenhar caminhos no plano cartesiano. Cada trajetória recebe um intervalo de tempo associado.

    🧠 Geração de Fluxos Gaussianos: Para cada timestamp do intervalo global, o script interpola os centroides (trajetórias) e gera pontos com distribuição normal ao redor deles.

    🧮 Classificação Multiclasse: Cada ponto gerado é rotulado com base em sua distância aos centroides ativos (usando KD-Tree), permitindo que ele pertença a múltiplos subfluxos simultaneamente (multirrótulo).

    💾 Exportação para CSV: Todos os pontos gerados são salvos com timestamp global, posição (x, y) e uma codificação binária indicando a quais subfluxos o ponto pertence.

## 🖱️ Controles da Interface

    Start Time e End Time: Insira os timestamps que delimitam o tempo de atividade da trajetória antes de desenhar.

    Clique e arraste com o mouse no gráfico para desenhar uma trajetória.

    Botões disponíveis:

        Generate Stream: Inicia a geração do fluxo com base nas trajetórias desenhadas.

        Save to CSV: Exporta os dados do fluxo para o arquivo stream_data.csv.

        Stop: Encerra a execução da aplicação.

## ⚙️ Tecnologias e Bibliotecas Utilizadas

    Matplotlib: Interface gráfica para desenhar e interagir.

    Numpy: Cálculos vetoriais e geração de distribuições normais.

    SciPy: Interpolação de trajetórias e busca espacial com cKDTree.

    CSV: Exportação de dados gerados.

    Random, Time, Sys: Utilidades auxiliares do Python.

## 🧪 Lógica de Geração de Dados

    Interpolação Temporal: As trajetórias desenhadas são interpoladas linearmente no tempo para definir os centroides ativos a cada timestamp.

    Geração de Pontos: Para cada centro ativo, 100 pontos são amostrados com distribuição normal (desvio padrão controlado).

    Classificação por Proximidade: Um ponto pode pertencer a múltiplos clusters se estiver dentro do raio de dois ou mais centroides. A verificação é feita via cKDTree para otimização.

## 📂 Estrutura do CSV Exportado

| global_timestamp | timestamp | x | y | cluster_1 | cluster_2 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1 | 5 | ... | ... | 1 | 0 | 

Cada linha representa um ponto gerado, com seus dados espaciais e pertencimento aos clusters (substreams).