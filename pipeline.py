import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
    r2_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_DADOS = PASTA_PROJETO / "data"

SEMENTE = 42
CLASSES = ["A", "D", "H"]
CODIGO_CLASSE = {classe: indice for indice, classe in enumerate(CLASSES)}
TEMPORADAS_AGRUPAMENTO = ["2014/15", "2015/16", "2016/17", "2017/18"]
TEMPORADAS_VALIDACAO = [
    "2018/19", "2019/20", "2020/21", "2021/22", "2022/23", "2023/24", "2024/25",
]
TEMPORADA_TESTE = "2025/26"
TEMPORADA_FUTURA = "2026/27"

COLUNAS_BRUTAS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
]

ESTATISTICAS_JOGO = [
    "gols_pro", "gols_contra", "chutes_pro", "chutes_contra",
    "no_alvo_pro", "no_alvo_contra", "escanteios_pro", "escanteios_contra",
    "faltas_pro", "amarelos_pro", "vermelhos_pro", "pontos",
]
COLUNAS_ESTILO = [coluna for coluna in ESTATISTICAS_JOGO if coluna != "pontos"]

JANELA_FORMA = 5
JANELA_ESTILO = 10
MINIMO_JOGOS_ESTILO = 5

ELO_INICIAL = 1500.0
ELO_PROMOVIDO = 1450.0
ELO_K = 20.0
ELO_VANTAGEM_CASA = 65.0
ELO_REGRESSAO_TEMPORADA = 0.25


def carregar_partidas():
    quadros = []
    for caminho in sorted(PASTA_DADOS.glob("E0_*.csv")):
        quadro = pd.read_csv(
            caminho, encoding="utf-8-sig", usecols=lambda coluna: coluna in COLUNAS_BRUTAS
        ).dropna(subset=["HomeTeam"])
        codigo = caminho.stem.split("_")[1]
        quadro["temporada"] = f"20{codigo[:2]}/{codigo[2:]}"
        quadros.append(quadro)
    partidas = pd.concat(quadros, ignore_index=True)
    partidas["data"] = pd.to_datetime(partidas["Date"], dayfirst=True, format="mixed")
    partidas = partidas.sort_values(["data", "HomeTeam"], kind="stable").reset_index(drop=True)
    return partidas.rename(columns={"HomeTeam": "casa", "AwayTeam": "fora", "FTR": "resultado"})


def pontos_do_resultado(resultado, e_mandante):
    if resultado == "D":
        return 1
    return 3 if (resultado == "H") == e_mandante else 0


def montar_formato_longo(partidas):
    visoes = []
    for e_mandante in (True, False):
        lado, outro = ("H", "A") if e_mandante else ("A", "H")
        visao = pd.DataFrame({
            "id_partida": partidas.index,
            "time": partidas["casa" if e_mandante else "fora"],
            "data": partidas["data"],
            "temporada": partidas["temporada"],
            "mandante": e_mandante,
            "gols_pro": partidas[f"FT{lado}G"],
            "gols_contra": partidas[f"FT{outro}G"],
            "chutes_pro": partidas[f"{lado}S"],
            "chutes_contra": partidas[f"{outro}S"],
            "no_alvo_pro": partidas[f"{lado}ST"],
            "no_alvo_contra": partidas[f"{outro}ST"],
            "escanteios_pro": partidas[f"{lado}C"],
            "escanteios_contra": partidas[f"{outro}C"],
            "faltas_pro": partidas[f"{lado}F"],
            "amarelos_pro": partidas[f"{lado}Y"],
            "vermelhos_pro": partidas[f"{lado}R"],
            "pontos": [pontos_do_resultado(r, e_mandante) for r in partidas["resultado"]],
        })
        visoes.append(visao)
    longo = pd.concat(visoes, ignore_index=True)
    return longo.sort_values(["time", "data"], kind="stable").reset_index(drop=True)


def media_movel_anterior(longo, colunas, janela, minimo):
    anterior = longo.groupby("time")[colunas].shift(1)
    anterior["time"] = longo["time"]
    medias = anterior.groupby("time")[colunas].rolling(janela, min_periods=minimo).mean()
    return medias.reset_index(level=0, drop=True).sort_index()


def acrescentar_contexto_de_temporada(longo):
    por_time_temporada = longo.groupby(["time", "temporada"])
    pontos_antes = por_time_temporada["pontos"].cumsum() - longo["pontos"]
    jogos_antes = por_time_temporada.cumcount()
    longo["pontos_por_jogo_temporada"] = (pontos_antes / jogos_antes.replace(0, np.nan))
    longo["jogos_na_temporada"] = jogos_antes
    longo["descanso_dias"] = longo.groupby("time")["data"].diff().dt.days.clip(upper=30)
    return longo


def calcular_variaveis_por_time(partidas):
    longo = montar_formato_longo(partidas)
    forma = media_movel_anterior(longo, ESTATISTICAS_JOGO, JANELA_FORMA, 1).add_prefix("forma_")
    estilo = media_movel_anterior(
        longo, COLUNAS_ESTILO, JANELA_ESTILO, MINIMO_JOGOS_ESTILO
    ).add_prefix("estilo_")
    longo = acrescentar_contexto_de_temporada(longo)
    return pd.concat([longo, forma, estilo], axis=1)


def fator_de_gols(diferenca):
    if diferenca <= 1:
        return 1.0
    if diferenca == 2:
        return 1.5
    return (11 + diferenca) / 8


def calcular_elo(partidas):
    notas = {}
    temporada_atual = None
    elo_casa, elo_fora = [], []
    for linha in partidas.itertuples():
        if linha.temporada != temporada_atual:
            temporada_atual = linha.temporada
            for time in notas:
                notas[time] += ELO_REGRESSAO_TEMPORADA * (ELO_INICIAL - notas[time])
        nota_casa = notas.setdefault(linha.casa, ELO_PROMOVIDO)
        nota_fora = notas.setdefault(linha.fora, ELO_PROMOVIDO)
        elo_casa.append(nota_casa)
        elo_fora.append(nota_fora)
        esperado = 1 / (1 + 10 ** (-(nota_casa + ELO_VANTAGEM_CASA - nota_fora) / 400))
        real = {"H": 1.0, "D": 0.5, "A": 0.0}[linha.resultado]
        ajuste = ELO_K * fator_de_gols(abs(linha.FTHG - linha.FTAG)) * (real - esperado)
        notas[linha.casa] = nota_casa + ajuste
        notas[linha.fora] = nota_fora - ajuste
    return np.array(elo_casa), np.array(elo_fora)


def montar_tabela_de_partidas():
    partidas = carregar_partidas()
    variaveis = calcular_variaveis_por_time(partidas)
    colunas_variaveis = [
        coluna for coluna in variaveis.columns
        if coluna.startswith(("forma_", "estilo_"))
        or coluna in ("pontos_por_jogo_temporada", "jogos_na_temporada", "descanso_dias")
    ]
    casa = variaveis[variaveis["mandante"]].set_index("id_partida")[colunas_variaveis].add_prefix("casa_")
    fora = variaveis[~variaveis["mandante"]].set_index("id_partida")[colunas_variaveis].add_prefix("fora_")
    tabela = partidas.join(casa).join(fora)
    tabela["elo_casa"], tabela["elo_fora"] = calcular_elo(partidas)
    tabela["elo_diferenca"] = tabela["elo_casa"] - tabela["elo_fora"]
    tabela["saldo_gols"] = tabela["FTHG"] - tabela["FTAG"]
    return tabela


def separar_conjuntos(tabela):
    desenvolvimento = tabela[tabela["temporada"] < TEMPORADA_TESTE].reset_index(drop=True)
    teste = tabela[tabela["temporada"] == TEMPORADA_TESTE].reset_index(drop=True)
    futura = tabela[tabela["temporada"] == TEMPORADA_FUTURA].reset_index(drop=True)
    return desenvolvimento, teste, futura


def montar_divisoes_temporais(desenvolvimento):
    temporadas = desenvolvimento["temporada"].to_numpy()
    divisoes = []
    for temporada in TEMPORADAS_VALIDACAO:
        treino = np.flatnonzero(temporadas < temporada)
        validacao = np.flatnonzero(temporadas == temporada)
        divisoes.append((treino, validacao))
    return divisoes


def empilhar_perfis(tabela):
    perfis = []
    for lado in ("casa", "fora"):
        colunas = [f"{lado}_estilo_{coluna}" for coluna in COLUNAS_ESTILO]
        perfis.append(tabela[colunas].set_axis(COLUNAS_ESTILO, axis=1))
    return pd.concat(perfis, ignore_index=True).dropna()


def montar_agrupador(quantidade_clusters):
    return Pipeline([
        ("escala", StandardScaler()),
        ("pca", PCA(n_components=0.90, random_state=SEMENTE)),
        ("kmeans", KMeans(n_clusters=quantidade_clusters, n_init=20, random_state=SEMENTE)),
    ])


def avaliar_valores_de_k(perfis):
    matriz = perfis.to_numpy()
    resultados = []
    for quantidade in range(2, 9):
        agrupador = montar_agrupador(quantidade).fit(matriz)
        projetado = agrupador[:-1].transform(matriz)
        resultados.append({
            "k": quantidade,
            "inercia": float(agrupador[-1].inertia_),
            "silhueta": float(silhouette_score(projetado, agrupador[-1].labels_)),
        })
    return resultados


def escolher_k(resultados):
    candidatos = [item for item in resultados if 3 <= item["k"] <= 6]
    return max(candidatos, key=lambda item: item["silhueta"])["k"]


def atribuir_clusters(tabela, agrupador, quantidade_clusters):
    colunas_novas = {}
    for lado in ("casa", "fora"):
        perfil = tabela[[f"{lado}_estilo_{coluna}" for coluna in COLUNAS_ESTILO]]
        completas = perfil.notna().all(axis=1).to_numpy()
        rotulos = np.full(len(tabela), -1)
        rotulos[completas] = agrupador.predict(perfil.to_numpy()[completas])
        for rotulo in range(-1, quantidade_clusters):
            nome = "sem_historico" if rotulo == -1 else str(rotulo)
            colunas_novas[f"{lado}_cluster_{nome}"] = (rotulos == rotulo).astype(float)
    return pd.DataFrame(colunas_novas, index=tabela.index)


def listar_colunas_base(tabela):
    colunas = ["elo_casa", "elo_fora", "elo_diferenca"]
    for lado in ("casa", "fora"):
        colunas += [c for c in tabela.columns if c.startswith(f"{lado}_forma_")]
        colunas += [f"{lado}_pontos_por_jogo_temporada", f"{lado}_jogos_na_temporada", f"{lado}_descanso_dias"]
    return colunas


def preprocessamento(escalar):
    passos = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        passos.append(("escala", StandardScaler()))
    return passos


def criar_classificadores():
    return {
        "Regressão Logística": (
            Pipeline(preprocessamento(True) + [("modelo", LogisticRegression(max_iter=3000))]),
            {"modelo__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
        ),
        "Árvore de Decisão": (
            Pipeline(preprocessamento(False) + [("modelo", DecisionTreeClassifier(random_state=SEMENTE))]),
            {"modelo__max_depth": [1, 2, 3, 4, 5, 6], "modelo__min_samples_leaf": [5, 10, 20, 50, 100]},
        ),
        "Random Forest": (
            Pipeline(preprocessamento(False) + [
                ("modelo", RandomForestClassifier(n_estimators=300, random_state=SEMENTE, n_jobs=1))
            ]),
            {
                "modelo__max_depth": [4, 6, 8, 12],
                "modelo__min_samples_leaf": [10, 25, 50, 100],
                "modelo__max_features": ["sqrt", 0.3],
            },
        ),
        "XGBoost": (
            Pipeline(preprocessamento(False) + [
                ("modelo", XGBClassifier(
                    objective="multi:softprob", eval_metric="mlogloss", subsample=0.8,
                    colsample_bytree=0.8, random_state=SEMENTE, n_jobs=1,
                ))
            ]),
            {
                "modelo__n_estimators": [100, 200, 400, 800, 1600],
                "modelo__max_depth": [1, 2, 3, 4],
                "modelo__learning_rate": [0.003, 0.01, 0.03],
            },
        ),
        "SVM (RBF)": (
            Pipeline(preprocessamento(True) + [
                ("modelo", CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=SEMENTE), method="sigmoid", ensemble=False, cv=5,
                ))
            ]),
            {"modelo__estimator__C": [0.01, 0.03, 0.1, 0.3, 1.0, 10.0], "modelo__estimator__gamma": ["scale", 0.00003, 0.0001, 0.0003, 0.001]},
        ),
    }


def criar_regressores():
    return {
        "Regressão Linear Múltipla": (
            Pipeline(preprocessamento(True) + [("modelo", LinearRegression())]), {},
        ),
        "Ridge": (
            Pipeline(preprocessamento(True) + [("modelo", Ridge())]),
            {"modelo__alpha": [1, 10, 100, 300, 1000, 3000]},
        ),
        "Random Forest": (
            Pipeline(preprocessamento(False) + [
                ("modelo", RandomForestRegressor(n_estimators=300, random_state=SEMENTE, n_jobs=1))
            ]),
            {"modelo__max_depth": [2, 3, 4, 6, 8], "modelo__min_samples_leaf": [5, 10, 25, 50]},
        ),
        "SVR (RBF)": (
            Pipeline(preprocessamento(True) + [("modelo", SVR(kernel="rbf"))]),
            {"modelo__C": [0.1, 1.0, 10.0, 100.0], "modelo__gamma": ["scale", 0.0003, 0.001, 0.003]},
        ),
    }


def probabilidades_das_odds(tabela):
    bruto = 1 / tabela[["B365A", "B365D", "B365H"]].to_numpy()
    return bruto / bruto.sum(axis=1, keepdims=True)


def perdas_por_partida(y, probabilidades):
    escolhidas = probabilidades[np.arange(len(y)), y]
    return -np.log(np.clip(escolhidas, 1e-15, 1.0))


def calcular_metricas_classificacao(y, probabilidades):
    previsto = probabilidades.argmax(axis=1)
    reais = np.eye(3)[y]
    acumulada_prevista = np.cumsum(probabilidades, axis=1)[:, :2]
    acumulada_real = np.cumsum(reais, axis=1)[:, :2]
    return {
        "acuracia": float(accuracy_score(y, previsto)),
        "f1_macro": float(f1_score(y, previsto, average="macro", labels=[0, 1, 2], zero_division=0)),
        "recall_empate": float(recall_score(y, previsto, labels=[1], average=None, zero_division=0)[0]),
        "log_loss": float(log_loss(y, probabilidades, labels=[0, 1, 2])),
        "auc_ovr": float(roc_auc_score(y, probabilidades, multi_class="ovr", average="macro", labels=[0, 1, 2])),
        "brier": float(((probabilidades - reais) ** 2).sum(axis=1).mean()),
        "rps": float(((acumulada_prevista - acumulada_real) ** 2).sum(axis=1).mean() / 2),
    }


def calcular_metricas_regressao(y, previsto):
    return {
        "mae": float(mean_absolute_error(y, previsto)),
        "rmse": float(np.sqrt(mean_squared_error(y, previsto))),
        "r2": float(r2_score(y, previsto)),
    }


def perda_media_por_temporada(estimador, x, y, divisoes):
    perdas = []
    for treino, validacao in divisoes:
        modelo = clone(estimador).fit(x.iloc[treino], y[treino])
        perdas.append(log_loss(y[validacao], modelo.predict_proba(x.iloc[validacao]), labels=[0, 1, 2]))
    return perdas


def buscar_melhor_modelo(estimador, grade, x, y, divisoes, pontuacao):
    busca = GridSearchCV(estimador, grade, cv=divisoes, scoring=pontuacao, n_jobs=-1)
    return busca.fit(x, y)


def resumir_busca(busca):
    melhor = busca.best_index_
    por_temporada = [-float(busca.cv_results_[f"split{i}_test_score"][melhor]) for i in range(len(TEMPORADAS_VALIDACAO))]
    return {
        "melhores_parametros": {chave: valor for chave, valor in busca.best_params_.items()},
        "cv_por_temporada": por_temporada,
        "cv_media": float(np.mean(por_temporada)),
    }


def rodar_classificacao(desenvolvimento, teste, futura, conjuntos, divisoes):
    y_dev = desenvolvimento["resultado"].map(CODIGO_CLASSE).to_numpy()
    y_teste = teste["resultado"].map(CODIGO_CLASSE).to_numpy()
    y_futura = futura["resultado"].map(CODIGO_CLASSE).to_numpy()
    resultados, previsoes_teste, previsoes_futura = {}, {}, {}
    for nome_conjunto, colunas in conjuntos.items():
        for nome, (estimador, grade) in criar_classificadores().items():
            busca = buscar_melhor_modelo(estimador, grade, desenvolvimento[colunas], y_dev, divisoes, "neg_log_loss")
            chave = f"{nome} | {nome_conjunto}"
            probabilidades_teste = busca.predict_proba(teste[colunas])
            probabilidades_futura = busca.predict_proba(futura[colunas])
            resultados[chave] = {
                **resumir_busca(busca),
                "teste": calcular_metricas_classificacao(y_teste, probabilidades_teste),
                "futura": calcular_metricas_classificacao(y_futura, probabilidades_futura),
            }
            previsoes_teste[chave] = probabilidades_teste
            previsoes_futura[chave] = probabilidades_futura
            print(chave, round(resultados[chave]["cv_media"], 4), round(resultados[chave]["teste"]["log_loss"], 4), flush=True)
    return resultados, previsoes_teste, previsoes_futura


def rodar_referencias(desenvolvimento, teste, futura, divisoes):
    y_dev = desenvolvimento["resultado"].map(CODIGO_CLASSE).to_numpy()
    y_teste = teste["resultado"].map(CODIGO_CLASSE).to_numpy()
    y_futura = futura["resultado"].map(CODIGO_CLASSE).to_numpy()
    probabilidades_odds_dev = probabilidades_das_odds(desenvolvimento)
    cv_odds = [
        float(log_loss(y_dev[val], probabilidades_odds_dev[val], labels=[0, 1, 2]))
        for _, val in divisoes
    ]
    frequencias = DummyClassifier(strategy="prior")
    so_elo = Pipeline([("escala", StandardScaler()), ("modelo", LogisticRegression(max_iter=3000))])
    resultados, previsoes_teste, previsoes_futura = {}, {}, {}
    montagens = {
        "Sempre mandante (frequências do treino)": (frequencias, ["elo_diferenca"]),
        "Só Elo (Regressão Logística)": (so_elo, ["elo_diferenca"]),
    }
    for nome, (estimador, colunas) in montagens.items():
        por_temporada = perda_media_por_temporada(estimador, desenvolvimento[colunas], y_dev, divisoes)
        modelo = clone(estimador).fit(desenvolvimento[colunas], y_dev)
        prob_teste = modelo.predict_proba(teste[colunas])
        prob_futura = modelo.predict_proba(futura[colunas])
        resultados[nome] = {
            "cv_por_temporada": por_temporada,
            "cv_media": float(np.mean(por_temporada)),
            "teste": calcular_metricas_classificacao(y_teste, prob_teste),
            "futura": calcular_metricas_classificacao(y_futura, prob_futura),
        }
        previsoes_teste[nome], previsoes_futura[nome] = prob_teste, prob_futura
    nome_odds = "Odds Bet365 (referência)"
    prob_teste, prob_futura = probabilidades_das_odds(teste), probabilidades_das_odds(futura)
    resultados[nome_odds] = {
        "cv_por_temporada": cv_odds,
        "cv_media": float(np.mean(cv_odds)),
        "teste": calcular_metricas_classificacao(y_teste, prob_teste),
        "futura": calcular_metricas_classificacao(y_futura, prob_futura),
    }
    previsoes_teste[nome_odds], previsoes_futura[nome_odds] = prob_teste, prob_futura
    return resultados, previsoes_teste, previsoes_futura


def rodar_regressao(desenvolvimento, teste, futura, conjuntos, divisoes):
    y_dev = desenvolvimento["saldo_gols"].to_numpy(dtype=float)
    y_teste = teste["saldo_gols"].to_numpy(dtype=float)
    y_futura = futura["saldo_gols"].to_numpy(dtype=float)
    resultados, previsoes_teste = {}, {}
    media_treino = Pipeline([("modelo", DummyRegressor(strategy="mean"))])
    for nome_conjunto, colunas in conjuntos.items():
        modelos = dict(criar_regressores())
        for nome, (estimador, grade) in modelos.items():
            busca = buscar_melhor_modelo(
                estimador, grade or {"modelo__fit_intercept": [True]}, desenvolvimento[colunas],
                y_dev, divisoes, "neg_root_mean_squared_error",
            )
            chave = f"{nome} | {nome_conjunto}"
            previsto_teste = busca.predict(teste[colunas])
            resultados[chave] = {
                "melhores_parametros": dict(busca.best_params_),
                "cv_rmse_media": float(-busca.best_score_),
                "teste": calcular_metricas_regressao(y_teste, previsto_teste),
                "futura": calcular_metricas_regressao(y_futura, busca.predict(futura[colunas])),
            }
            previsoes_teste[chave] = previsto_teste
            print(chave, round(resultados[chave]["cv_rmse_media"], 4), round(resultados[chave]["teste"]["rmse"], 4), flush=True)
    referencia = media_treino.fit(desenvolvimento[["elo_diferenca"]], y_dev)
    previsto_referencia = referencia.predict(teste[["elo_diferenca"]])
    resultados["Média do treino (referência)"] = {
        "teste": calcular_metricas_regressao(y_teste, previsto_referencia),
        "futura": calcular_metricas_regressao(y_futura, referencia.predict(futura[["elo_diferenca"]])),
    }
    previsoes_teste["Média do treino (referência)"] = previsto_referencia
    return resultados, previsoes_teste


def bootstrap_da_diferenca(perdas_a, perdas_b, repeticoes=2000):
    gerador = np.random.default_rng(SEMENTE)
    diferenca = perdas_a - perdas_b
    sorteios = gerador.integers(0, len(diferenca), size=(repeticoes, len(diferenca)))
    medias = diferenca[sorteios].mean(axis=1)
    return {
        "media": float(diferenca.mean()),
        "ic95_inferior": float(np.percentile(medias, 2.5)),
        "ic95_superior": float(np.percentile(medias, 97.5)),
    }


def escolher_modelo_final(resultados_modelos):
    return min(resultados_modelos, key=lambda chave: resultados_modelos[chave]["cv_media"])


def comparar_no_teste(teste, previsoes, final, comparados):
    y = teste["resultado"].map(CODIGO_CLASSE).to_numpy()
    perdas_final = perdas_por_partida(y, previsoes[final])
    return {
        nome: bootstrap_da_diferenca(perdas_final, perdas_por_partida(y, previsoes[nome]))
        for nome in comparados
    }


def salvar_previsoes(teste, previsoes, previsoes_regressao, caminho):
    saida = teste[["temporada", "data", "casa", "fora", "resultado", "saldo_gols"]].copy()
    for nome, probabilidades in previsoes.items():
        for indice, classe in enumerate(CLASSES):
            saida[f"{nome} :: {classe}"] = probabilidades[:, indice]
    for nome, previsto in previsoes_regressao.items():
        saida[f"{nome} :: saldo"] = previsto
    saida.to_csv(caminho, index=False)


def escolher_regressor_final(resultados_regressao):
    com_validacao = {n: m for n, m in resultados_regressao.items() if "cv_rmse_media" in m}
    return min(com_validacao, key=lambda chave: com_validacao[chave]["cv_rmse_media"])


def descrever_agrupamento(perfis, agrupador, quantidade_clusters, avaliacao_k):
    rotulos = agrupador[-1].labels_
    perfil_medio = perfis.groupby(rotulos).mean().round(3)
    return {
        "k_escolhido": quantidade_clusters,
        "avaliacao_k": avaliacao_k,
        "tamanho_por_cluster": {int(k): int(v) for k, v in pd.Series(rotulos).value_counts().sort_index().items()},
        "perfil_medio_por_cluster": {int(k): linha.to_dict() for k, linha in perfil_medio.iterrows()},
        "variancia_explicada_pca": float(agrupador["pca"].explained_variance_ratio_.sum()),
        "componentes_pca": int(agrupador["pca"].n_components_),
    }


def executar():
    inicio = time.time()
    tabela = montar_tabela_de_partidas()
    desenvolvimento, teste, futura = separar_conjuntos(tabela)
    divisoes = montar_divisoes_temporais(desenvolvimento)

    perfis = empilhar_perfis(desenvolvimento[desenvolvimento["temporada"].isin(TEMPORADAS_AGRUPAMENTO)])
    avaliacao_k = avaliar_valores_de_k(perfis)
    quantidade_clusters = escolher_k(avaliacao_k)
    agrupador = montar_agrupador(quantidade_clusters).fit(perfis.to_numpy())
    agrupamento = descrever_agrupamento(perfis, agrupador, quantidade_clusters, avaliacao_k)
    print("k escolhido:", quantidade_clusters, flush=True)

    conjuntos_de_tabelas = []
    for parte in (desenvolvimento, teste, futura):
        conjuntos_de_tabelas.append(pd.concat([parte, atribuir_clusters(parte, agrupador, quantidade_clusters)], axis=1))
    desenvolvimento, teste, futura = conjuntos_de_tabelas

    colunas_base = listar_colunas_base(desenvolvimento)
    colunas_cluster = [c for c in desenvolvimento.columns if "_cluster_" in c]
    conjuntos = {"sem K-Means": colunas_base, "com K-Means": colunas_base + colunas_cluster}

    resultados_classificacao, previsoes_teste, previsoes_futura = rodar_classificacao(
        desenvolvimento, teste, futura, conjuntos, divisoes
    )
    referencias, previsoes_referencia_teste, previsoes_referencia_futura = rodar_referencias(
        desenvolvimento, teste, futura, divisoes
    )
    previsoes_teste.update(previsoes_referencia_teste)
    resultados_regressao, previsoes_regressao = rodar_regressao(
        desenvolvimento, teste, futura, conjuntos, divisoes
    )

    final = escolher_modelo_final(resultados_classificacao)
    sem_kmeans = final.replace("com K-Means", "sem K-Means")
    comparacoes = comparar_no_teste(
        teste, previsoes_teste, final,
        ["Só Elo (Regressão Logística)", "Odds Bet365 (referência)", sem_kmeans],
    )

    resultados = {
        "particao": {
            "treino_e_validacao": f"{desenvolvimento['temporada'].min()} a {desenvolvimento['temporada'].max()}",
            "jogos_desenvolvimento": int(len(desenvolvimento)),
            "temporada_teste": TEMPORADA_TESTE,
            "jogos_teste": int(len(teste)),
            "temporada_futura_parcial": TEMPORADA_FUTURA,
            "jogos_futura": int(len(futura)),
            "temporadas_validacao_cruzada": TEMPORADAS_VALIDACAO,
            "temporadas_agrupamento": TEMPORADAS_AGRUPAMENTO,
            "distribuicao_resultado_teste": teste["resultado"].value_counts(normalize=True).round(4).to_dict(),
            "distribuicao_resultado_desenvolvimento": desenvolvimento["resultado"].value_counts(normalize=True).round(4).to_dict(),
        },
        "agrupamento": agrupamento,
        "colunas": {"base": colunas_base, "cluster": colunas_cluster},
        "classificacao": {**resultados_classificacao, **referencias},
        "modelo_final_classificacao": final,
        "comparacoes_bootstrap_log_loss_teste": comparacoes,
        "regressao": resultados_regressao,
        "modelo_final_regressao": escolher_regressor_final(resultados_regressao),
        "versoes": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "segundos": round(time.time() - inicio, 1),
    }
    with open(PASTA_PROJETO / "resultados.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=2)
    salvar_previsoes(teste, previsoes_teste, previsoes_regressao, PASTA_PROJETO / "previsoes_teste.csv")
    print("final:", final, "| segundos:", resultados["segundos"], flush=True)


if __name__ == "__main__":
    executar()
