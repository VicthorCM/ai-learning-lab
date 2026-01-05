#  AI & Machine Learning Lab

Repositório dedicado ao armazenamento de projetos e experimentos desenvolvidos durante a graduação nas disciplinas de **Inteligência Artificial**, **Machine Learning** e **Deep Learning**.

---

##  Trabalhos em Destaque

### 1. Predição de Preços do Bitcoin (LSTM - Finance)
- **Objetivo:** Prever o preço de fechamento do Bitcoin (BTC) utilizando dados históricos para auxiliar na análise de tendências de mercado.
- **Tecnologias:** Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib.
- **Técnicas:** - **Séries Temporais Financeiras:** Normalização de dados com `MinMaxScaler` para acelerar a convergência do modelo.
    - **Arquitetura de Rede Neural:** Uso de camadas **LSTM** (Long Short-Term Memory) empilhadas para capturar padrões de volatilidade temporal.
    - **Validação:** Divisão temporal dos dados (train/test split) para garantir que o modelo seja testado em dados futuros em relação ao treinamento.
- **Resultado:** Modelo capaz de seguir a tendência de preço do ativo, validado através da comparação visual entre valores reais e preditos.
- **Status:** Concluído ✅

### 2. Previsão de Casos de Catapora (LSTM - Epidemiology)
- **Objetivo:** Prever o número semanal de casos de catapora na Hungria utilizando janelas deslizantes de 40 semanas.
- **Tecnologias:** Python, TensorFlow, Scikit-Learn.
- **Técnicas:** LSTM, Huber Loss para tratamento de outliers e análise sazonal.
- **Resultado:** Coeficiente de determinação ($R^2$) de ~0.67.
- **Status:** Concluído ✅

### 3. Predição de Notas de Jogos (Regressão Multi-Modelo)
- **Objetivo:** Prever o `rating` de jogos eletrônicos com base em variáveis como preço e tempo de jogo.
- **Tecnologias:** Scikit-Learn, XGBoost, Pandas.
- **Técnicas:** Target Encoding, Isolation Forest e benchmarking entre modelos clássicos e modernos.
- **Status:** Concluído ✅

### 4. Classificação de Performance Estudantil (MLP Classifier)
- **Objetivo:** Classificar o desempenho acadêmico de alunos.
- **Tecnologias:** Scikit-Learn, Pandas.
- **Técnicas:** Normalização e Redes Neurais densas (MLP).
- **Status:** Concluído ✅

---

##  Tecnologias e Ferramentas
- **Linguagem:** Python 3.x
- **Deep Learning:** TensorFlow, Keras (LSTMs, Redes Neurais).
- **Machine Learning:** Scikit-Learn (Regressão, Classificação, Pré-processamento), XGBoost.
- **Ambientes:** VS Code, Jupyter Notebook, Google Colab.

---

## 📁 Estrutura do Repositório

```text
/
├── machine-learning/
│   ├── student-performance/        # Classificação de Notas Estudantis
│   │   └──student-performance-mlp-classifier.py
│   │
│   └── game-rating-prediction/     # Regressão de Notas de Jogos (Steam)
│       ├── game-rating-prediction.py
│       ├── plots/                  # Matriz de correlação e gráficos de resíduos
│       └── exports/                # CSVs com predições dos modelos
│
├── deep-learning/
│   ├── chickenpox-cases-lstm/      # Séries Temporais: Casos de Catapora na Hungria
│   │   ├── plots/                  # Gráficos de Sazonalidade e Real vs. Predito
│   │   └── chickenpox-cases-lstm.ipynb
│   │
│   └── bitcoin-price-prediction/   # Séries Temporais: Predição de Preços de Cripto
│       └── LSTM_BTC.ipynb
│
├── datasets/                       # Armazenamento centralizado das bases de dados (CSV)
│
└── .gitignore                      # Configuração para ignorar venv e caches de modelos