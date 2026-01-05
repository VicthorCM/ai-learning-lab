import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.ensemble import IsolationForest


# 1 - escolha da base de dados 

"""
A base de dados escolhida foi a Students Performance Dataset (https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)

colunas:
        ['StudentID,'Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
        'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
        'Volunteering', 'GPA', 'GradeClass']

"""

df = pd.read_csv('datasets/Student_performance_data _.csv')

df.drop(['StudentID'],axis=1,inplace=True)



# Análise inicial dos dados

# for i in df.keys():
#     print(f'----------------{i}---------------')
#     print(df[i].value_counts())
#     print(df[i].describe())
#     print(df[i].info())



# 2 - Codificação das variáveis categóricas

""" As variáveis categóricas já estão codificadas"""

# 3- Remoção de outliers (variáveis numéricas)

df_filtrado = df.copy()


# Modelo Isolation Forest para identificação de outlier

modelo = IsolationForest(contamination=0.005, random_state=42)
df_filtrado["outlier"] = modelo.fit_predict(df)
df_sem_outliers = df_filtrado[df_filtrado["outlier"] == 1].drop(columns=["outlier"])

# print(df_filtrado[df_filtrado["outlier"] == -1]) #     ver os outliers


# divisão do dataframe e features e targets
x = pd.DataFrame(df_sem_outliers[['Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
                                'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
                                'Volunteering', 'GPA']],
                        columns=  ['Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
                                'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
                                'Volunteering', 'GPA'])

y = pd.Series(df_sem_outliers['GradeClass'], name = 'target')


# 4 - Escalonamento de dados

scaler = MinMaxScaler()

x_normalizado = scaler.fit_transform(x)



# 5 -  Divisão entre treino e teste

x_train,x_test, y_train, y_test = train_test_split(x_normalizado,y, test_size=0.3,stratify=y) 

# 5.1 - Balanceamento de classes

from sklearn.datasets import make_classification 
# from imblearn.over_sampling import SMOTE 
from collections import Counter 


# # Aplicar SMOTE 
# smote = SMOTE(random_state=42)

# x_res, y_res = smote.fit_resample(x_train, y_train) 



# 6 - Treinamento da rede neural MLP

mlp = MLPClassifier(hidden_layer_sizes=(20,10), #(10,10,10,10)
                     max_iter=1000,
                     activation='relu',
                     batch_size=16,
                     solver='adam')

mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)

# 6.1 - Plotando o gráfico de loss
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.title('Curva de Loss durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("machine-learning/student-performance-mlp-classifier/plots/loss.png")

# 7 - Avaliação da rede neural
import seaborn as sns
from sklearn.metrics import classification_report

# 1. Gerando o Relatório de Classificação em texto
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred))

# 2. Criando uma Matriz de Confusão Visual com Seaborn
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)

# Plotando o heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'],
            yticklabels=['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'])

plt.title('Matriz de Confusão - Desempenho Estudantil')
plt.xlabel('Predição do Modelo')
plt.ylabel('Valor Real (Ground Truth)')
plt.savefig("machine-learning/student-performance-mlp-classifier/plots/predicao.png")