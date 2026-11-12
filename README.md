# Previsão de Resultados - Premier League

Prevê o resultado de partidas da Premier League antes de elas acontecerem: vitória do mandante, empate ou vitória do visitante (classificação) e o saldo de gols (regressão). Compara Árvore de Decisão, Random Forest, XGBoost, SVM e Regressão Logística, com e sem o estilo de jogo dos times descoberto por K-Means, contra baselines simples e contra as odds das casas de apostas, usadas só como régua de dificuldade.

- **EQUIPE**: Thiago Dallo, Vinicius Fabris e Murilo Cambruzzi.
- **DISCIPLINA:** Machine Learning
- **CURSO:** Engenharia de Software - 4ª fase
- **FACULDADE:** UniSATC

## Tecnologias

- Python 3.11 ou superior
- pandas, numpy
- scikit-learn (Pipeline, GridSearchCV, KMeans, PCA, SVC, árvores, Random Forest)
- XGBoost
- matplotlib
- Jupyter / Google Colab

## O que o projeto faz

- **Variáveis só do passado**: Elo de cada time, média dos últimos 5 jogos em 12 estatísticas, pontos por jogo na temporada e dias de descanso. Um teste no notebook recalcula à mão a forma do Liverpool em 461 jogos para provar que nenhuma variável usa o jogo atual.
- **Divisão temporal**: validação cruzada com janela crescente em 7 temporadas (2018/19 a 2024/25) e teste em 2025/26, tocado uma única vez. Nada de divisão aleatória.
- **K-Means de estilo de jogo**: agrupa os times pelas médias dos últimos 10 jogos (PCA antes, k escolhido pela silhueta), ajustado só nas 4 primeiras temporadas. O grupo do mandante e do visitante vira variável, e cada modelo roda com e sem ela.
- **Modelos**: Regressão Logística, Árvore de Decisão, Random Forest, XGBoost e SVM (RBF com probabilidades calibradas), todos com `GridSearchCV` por log loss.
- **Baselines honestos**: "sempre as frequências do treino", só a diferença de Elo e as odds da Bet365.
- **Métricas de probabilidade**: log loss, Brier e RPS, além de acurácia, F1 macro e recall do empate.
- **Regressão do saldo de gols**: Regressão Linear Múltipla, Ridge, Random Forest e SVR contra a média do treino.
- **Intervalos por bootstrap**: diferença de log loss entre o modelo final, o Elo e as odds, com IC de 95%.

## Os dados

[football-data.co.uk](https://www.football-data.co.uk/englandm.php), Premier League, 12 temporadas completas (2014/15 a 2025/26, 4.560 jogos) mais 100 jogos de 2026/27 usados só como checagem extra. Os CSV estão em `data/`, com a cópia de 12/11/2026.

O Brasileirão foi avaliado e descartado: a melhor base aberta encontrada tem estatísticas de jogo em cerca de 750 partidas, em três temporadas e com falhas.

## O que tem aqui

| Arquivo | Para que serve |
|---------|----------------|
| `Projeto_Final_ML.ipynb` | Notebook completo e executado, pronto para rodar no Google Colab |
| `Projeto_Final_ML_Previsao_Futebol.docx` | Relatório final, estruturado conforme o modelo da disciplina |
| `pipeline.py` | Pipeline equivalente ao notebook, para rodar via terminal |
| `resultados.json` | Métricas e diagnósticos gerados pelo pipeline |
| `previsoes_teste.csv` | Probabilidades de cada modelo para os 380 jogos de teste |
| `data/` | CSV das temporadas |
| `figs/` | Gráficos usados no relatório e no notebook |

## Como rodar

```bash
pip install -r requirements.txt
python pipeline.py
```

O pipeline roda em cerca de 75 segundos e regrava `resultados.json` e `previsoes_teste.csv`. No Colab, abra o notebook e execute a primeira célula, que clona o repositório.

## Resultados

### Classificação: quem ganha o jogo

Log loss (menor é melhor). A validação é a média de 7 temporadas; o teste é a temporada 2025/26 (380 jogos).

| Modelo | Validação | Teste | Brier | RPS | Acurácia | F1 macro |
|--------|-----------|-------|-------|-----|----------|----------|
| Odds Bet365 (só régua) | 0,951 | 1,019 | 0,612 | 0,206 | 0,489 | 0,367 |
| Só Elo | 0,969 | 1,026 | 0,617 | 0,209 | 0,479 | 0,356 |
| Regressão Logística | 0,971 | 1,037 | 0,623 | 0,211 | 0,487 | 0,362 |
| Random Forest (com K-Means) | 0,973 | 1,034 | 0,622 | 0,210 | 0,471 | 0,350 |
| XGBoost (com K-Means) | 0,976 | 1,037 | 0,624 | 0,211 | 0,476 | 0,352 |
| Árvore de Decisão | 0,985 | 1,035 | 0,622 | 0,210 | 0,492 | 0,350 |
| SVM RBF (com K-Means) | 0,985 | 1,039 | 0,627 | 0,214 | 0,466 | 0,354 |
| Sempre as frequências do treino | 1,065 | 1,084 | 0,656 | 0,228 | 0,426 | 0,199 |

Modelo final escolhido pelo menor log loss na validação entre os cinco algoritmos: Regressão Logística sem K-Means. No teste, ele fica 0,010 acima do Só Elo (IC 95% de -0,003 a 0,024, empate estatístico) e 0,018 acima das odds (IC 95% de 0,002 a 0,035, atrás de forma significativa).

- **Modelos simples ganham**: o Só Elo empata com a Regressão Logística completa e fica ligeiramente à frente dos ensembles. Mesmo depois de ampliar as grades duas vezes, os melhores hiperparâmetros ficam nos extremos mais regularizados (C = 0,003 na logística, profundidade 2 na árvore, C = 0,01 e gamma = 0,00003 no SVM), sinal de pouco sinal extra nas variáveis. O SVM em especial tende a um modelo quase constante e não ganha nada do kernel.
- **O K-Means não ajudou de forma consistente**: na validação, a diferença com e sem os grupos é de no máximo 0,001, e a Regressão Logística chega a piorar um pouco. No teste, todos melhoram ou empatam (o SVM 0,005, o Random Forest 0,004, a Regressão Logística 0,003), mas o teste tem 380 jogos e a validação em 7 temporadas não confirma o ganho. Os três grupos são interpretáveis (defensivos e frágeis, dominantes e equilibrados), mas a silhueta de 0,17 mostra que estilo é um contínuo.
- **Empate quase não se prevê**: nenhum modelo o escolhe como resultado mais provável, então o recall do empate é zero. O produto útil são as probabilidades.
- **O teste tem 380 jogos**: diferenças pequenas entre modelos ficam dentro do erro, e a validação em 7 temporadas pesa mais na decisão.

### Regressão: saldo de gols

Teste (380 jogos):

| Modelo | MAE | RMSE | R² |
|--------|-----|------|----|
| Média do treino | 1,292 | 1,618 | 0,000 |
| Regressão Linear Múltipla | 1,229 | 1,539 | 0,094 |
| Ridge | 1,227 | 1,535 | 0,099 |
| SVR RBF | 1,225 | 1,532 | 0,103 |
| Random Forest | 1,232 | 1,526 | 0,110 |

Todos os modelos reduzem o erro em relação à média do treino, mas explicam cerca de 10% da variância. O SVR sem K-Means foi o escolhido pela validação.

## Limitações

- Só uma liga e só dados de jogo: sem escalação, lesões, xG por partida nem odds de fechamento como variável.
- As odds servem apenas de referência de dificuldade. O trabalho não avalia nem recomenda apostas.
- Testado só com Python 3.14, pandas 3.1, scikit-learn 1.9 e XGBoost 3.4. Não foi rodado nas versões do Colab.
