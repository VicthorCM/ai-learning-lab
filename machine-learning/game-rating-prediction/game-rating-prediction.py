
# 1. Importações e Configurações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os



df = pd.read_csv('datasets/bestSelling_games.csv')

df.drop(['game_name','reviews_like_rate','all_reviews_number','release_date','user_defined_tags','supported_os','supported_languages','other_features'],axis=1,inplace=True)


"""
Análise exploratória dos dados
"""

print(df.isna().sum())

for col in df.keys():
    print(f"{col}: {df[col].dtype}")

import matplotlib.pyplot as plt
import seaborn as sns

# Define o estilo
sns.set(style="whitegrid")

# Para cada coluna numérica, plotar histograma
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribuição de {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(f"machine-learning/game-rating-prediction/plots/ditribuicao-{col}.png")

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"machine-learning/game-rating-prediction/plots//boxsplot-{col}.png")

for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
    plt.figure(figsize=(10, 4))
    df[col].value_counts().head(15).plot(kind='bar', color='lightgreen')
    plt.title(f'Contagem dos Top 15 valores em {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"machine-learning/game-rating-prediction/plots//top15-{col}.png")

plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("machine-learning/game-rating-prediction/plots/matriz-correlacao.png")

sns.pairplot(df.select_dtypes(include=['int64', 'float64']).dropna().sample(n=200))  # limitar amostras para performance
plt.suptitle("Pairplot entre variáveis numéricas", y=1.02)
plt.savefig(f"machine-learning/game-rating-prediction/plots//pairplot.png")

for i in df.keys():
    print(f'----------------{i}---------------')
    print(df[i].value_counts())
    print(df[i].describe())
    print(df[i].info())



"""
Remoção de Outliers com IsolationFlorest

"""

import pandas as pd
from sklearn.ensemble import IsolationForest


modelo = IsolationForest(contamination=0.05)
df['outlier'] = modelo.fit_predict(df[['price','length']])

df_filtrado = df[df["outlier"] == 1]

print(df['outlier'].value_counts())

print(df[df['outlier']==-1]['price'].describe())

for col in df_filtrado.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_filtrado[col], color='orange')
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"machine-learning/game-rating-prediction/plots//boxplot-{col}-sem-outliers.png")


"""

Codificação de variáveis categóricas

Foi utilizada a técnica de TargetEncoder para feature 'developer' devido a grande quantidade de valores possíveis
"""

df_filtrado['developer_encoded']= df_filtrado.groupby('developer')['rating'].transform('mean')

df_filtrado.drop(['developer','outlier'],axis=1,inplace=True)

# df_filtrado.head()

"""
Codificação com OneHotEncoder para a variável 'age_restriction' 

"""

from sklearn.preprocessing import OneHotEncoder


# Criando o codificador com o nome da coluna
encoder = OneHotEncoder(sparse_output=False)
# Aplicando o encoder
encoded = encoder.fit_transform(df_filtrado[["age_restriction"]])
col_names = encoder.get_feature_names_out(["age_restriction"])

df_encoded = pd.DataFrame(encoded, columns=col_names)

df_final = pd.concat([df_filtrado, df_encoded], axis=1)

df_final.dropna(inplace=True)

"""
Separando o target das demais feature

Também foi retirada 'age_restriction' após a codificação
"""

target = df_final['rating']

df_final.drop(['rating','age_restriction'],axis=1,inplace=True)

"""
Escalonamento das variáveis quantitativas
"""

from sklearn.preprocessing import StandardScaler


col_num = df_final.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

df_scaler = df_final.copy()

df_scaler[col_num] = scaler.fit_transform(df_final[col_num])

df_scaler.head()

target

"""
Seleção de variáveis com método Filter
"""

# Calcular a matriz de correlação absoluta para seleção de variaveis - Metodo Filter
correlation_matrix = df_scaler.corr().abs()
# Selecionar apenas a parte superior da matriz de correlação
upper = correlation_matrix.where(
 np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
# Encontrar colunas com correlação > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
# Remover as colunas com alta correlação
df_reduced = df_scaler.drop(columns=to_drop)
# Mostrar colunas removidas
print("\nColunas removidas por alta correlação:", to_drop)

"""
Definindo X e Y
"""

#define X como as variaveis para treino e Y a variavel alvo
X = df_reduced
y = target

"""
Separar entre Treino e teste
"""

from sklearn.model_selection import  train_test_split
# Dividir em treino (70%) e teste (30%) - método Holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Treinamento do modelo
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

modelos = {
    "MLPRegressor": MLPRegressor(solver='adam', max_iter=500, random_state=42),
    "SVR": SVR(gamma='auto'),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42)
}

for nome, modelo in modelos.items():
    print(f"\n{nome}:")

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Coeficiente de Determinação (R2): {r2:.4f}")
    # 7. Visualizar previsões
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([0, 5], [0, 5], color='red', linestyle='--')
    plt.xlabel("Nota real")
    plt.ylabel("Nota previsto")
    plt.title(f"Previsão das notas dos jogos -{nome} ")
    plt.grid(True)
    plt.savefig(f"machine-learning/game-rating-prediction/plots/{nome}.png")

"""
Avaliação do modelo
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Avaliação
print("\n--- Avaliação ---")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

"""
Visualizar Previsões
"""

# 7. Visualizar previsões
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.xlabel("Nota real")
plt.ylabel("Nota previsto")
plt.title("Previsão das notas dos jogos - Random Forest Regressor")
plt.grid(True)
plt.show()

"""
Geração de arquivo com as previsões
"""

df_y_pred = pd.DataFrame(y_pred, columns=['rating'])

# Salvar o DataFrame em um arquivo CSV
df_y_pred.to_csv(f'machine-learning/game-rating-prediction/exports/rating.csv', index=False)
