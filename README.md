![status](https://img.shields.io/badge/status-stable-brightgreen)  ![python](https://img.shields.io/badge/python-3.10%2B-blue)
 
![tests](https://img.shields.io/badge/tests-pytest-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)
  
![build](https://github.com/treino258/Predict-Health-Risk-IA/actions/workflows/test.yml/badge.svg)


# 🫀 Predict Health Risk IA

> Ferramenta de triagem de risco cardíaco com Machine Learning — do dado clínico à predição em tempo real.

📌 Este projeto tem caráter educacional e demonstra boas práticas de Machine Learning, incluindo validação adequada, ajuste de threshold e calibração de probabilidades. Para uso em produção clínica, seriam necessários validação externa, aprovação regulatória e integração com especialistas da área da saúde.

![Interface](src/img/image.png)

---

## 🎯 Business Goal

Doenças cardiovasculares são a principal causa de morte no mundo. O diagnóstico precoce depende de profissionais especializados e exames que nem sempre são acessíveis. Este projeto cria uma ferramenta de triagem inteligente que auxilia na identificação de pacientes em alto risco cardíaco com base em dados clínicos simples — democratizando um primeiro nível de avaliação preventiva.

> ⚠️ **Aviso:** Esta ferramenta é um apoio à triagem, não substitui diagnóstico médico profissional.

---

## ⚙️ Solution & Impact

Modelo de classificação binária (**Alto Risco / Baixo Risco**) treinado com dados clínicos reais, validado com cross-validation de 5 folds e otimizado via `RandomizedSearchCV` com foco em **recall** — minimizando falsos negativos, que representam o erro de maior impacto clínico na saúde.

Entregue como aplicação web via **Streamlit**, permitindo que qualquer pessoa insira dados clínicos e receba uma predição instantânea, sem necessidade de infraestrutura técnica.

---

## 🔍 Análise Exploratória dos Dados (EDA)

Antes de modelar, os dados foram investigados em profundidade para garantir qualidade e extrair insights clínicos.

**Qualidade dos dados**
172 pacientes apresentavam Colesterol = 0 — biologicamente impossível. Esses valores foram tratados como ausentes e substituídos pela mediana do grupo (doente/saudável), preservando o padrão clínico de cada classe.

**Principais descobertas**

| Variável | Saudável (mediana) | Doença (mediana) | Diferença |
|---|---|---|---|
| Idade | 51 anos | 57 anos | +11.8% |
| MaxHR | 150 bpm | 126 bpm | **-16.0%** |
| Colesterol | 231 mg/dL | 246 mg/dL | +6.3% |
| Oldpeak | 0.0 | 1.2 | **↑ forte** |
| RestingBP | 130 mmHg | 132 mmHg | +1.5% |

**Insights clínicos extraídos:**

- **Oldpeak** é o indicador mais discriminante — pacientes doentes têm depressão do segmento ST significativamente maior, consistente com a literatura cardiológica
- **MaxHR** inversamente associado à doença — frequência cardíaca máxima mais baixa indica menor capacidade funcional cardíaca
- **FastingBS** (glicemia em jejum > 120) aparece quase exclusivamente em pacientes doentes, sugerindo forte associação com risco metabólico-cardiovascular
- **Colesterol** apresenta alta variabilidade e outliers expressivos — variável ruidosa que exigiu tratamento cuidadoso
- **Idade** média dos doentes é 6 anos superior, reforçando o risco crescente com o envelhecimento

---

## 📊 Resultados do Modelo

| Métrica | Valor |
|---|---|
| Acurácia | 88.0% |
| Recall — Alto Risco (classe 1) | **89,47%** |
| Specificity — Baixo Risco (classe 0) | **87,10%** |
| Precision — Alto Risco (classe 1) | 89,47% |
| F1-Score — Alto Risco (classe 1) | 89,47% |


---

**Confusion Matrix:**
```
[[54  8]   → 54 saudáveis corretos | 8 falsos positivos
 [ 8  68]]  → 8 falsos negativos    | 68 doentes corretos
```

O modelo identifica corretamente **89,47% dos pacientes com risco cardíaco real** — o falso negativo (doente classificado como saudável) é o erro mais perigoso na saúde e foi o critério principal de otimização. A Specificity de 87.10% indica que o modelo também evita alarmar pacientes saudáveis desnecessariamente.

---

## 🧠 Decisões Técnicas

**Por que separar os dados em três conjuntos?**

A divisão clássica treino/teste em duas partes cria um problema sutil: se o threshold é ajustado usando o conjunto de teste, os resultados reportados são otimistas e não refletem o desempenho real em dados novos — isso se chama data leakage.

Neste projeto, os dados foram divididos em três partes:

```
918 pacientes (100%)
├── Treino  (70%) → treina o modelo e os hiperparâmetros
├── Validação (15%) → ajusta o threshold aqui, sem tocar no teste
└── Teste (15%) → reporta o resultado final, tocado uma única vez
```

Cada conjunto tem uma única responsabilidade. O threshold foi escolhido na validação, e o teste foi usado apenas para reportar o número final — garantindo uma estimativa honesta do desempenho real.

**Por que Recall como métrica principal?**

Em triagem cardíaca, os dois tipos de erro têm custos assimétricos:

- Falso Negativo (alto risco classificado como baixo): paciente não recebe atenção médica → consequência potencialmente fatal
- Falso Positivo (baixo risco classificado como alto): paciente faz exames desnecessários → custo controlável

Por isso, minimizar Falsos Negativos é prioridade. O Recall mede exatamente isso: dos pacientes que realmente têm alto risco, quantos o modelo identifica corretamente?

Acurácia seria uma métrica enganosa aqui — com dataset desbalanceado, um modelo que classifica tudo como "baixo risco" pode ter acurácia alta e Recall zero.

**Por que RandomForestClassifier?**

Três razões práticas para esse problema:

- **Interpretabilidade:** fornece `feature_importances_`, permitindo entender quais variáveis clínicas mais influenciam o risco — essencial em contexto médico
- **Robustez a outliers:** dados clínicos têm ruído natural (ex: 172 registros de colesterol inconsistentes tratados via imputação por mediana)
- **Desempenho sem escalonamento:** diferente de SVM ou KNN, não exige normalização das features, simplificando o pipeline de inferência

**Por que imputação por mediana no colesterol?**

172 registros apresentavam colesterol = 0, valor clinicamente impossível. Duas opções foram consideradas:

- Remover os registros: perderia ~18% do dataset — impacto significativo para um modelo de saúde
- Imputar pela mediana: preserva os dados e é robusto a outliers, ao contrário da média

A mediana foi escolhida por ser menos sensível aos valores extremos presentes na distribuição de colesterol.

**Por que validação cruzada (5-fold) além do train_test_split?**

O `train_test_split` avalia o modelo em uma única divisão — resultado dependente do acaso da partição. A validação cruzada com 5 folds garante que todo dado foi usado tanto para treino quanto para teste, produzindo uma estimativa mais confiável da capacidade de generalização do modelo.

**Por que calibrar probabilidades (CalibratedClassifierCV)?**

Modelos baseados em árvores, como RandomForest, frequentemente produzem probabilidades mal calibradas — ou seja, o valor retornado por `predict_proba` não representa corretamente a probabilidade real do evento. Isso impacta diretamente o uso de threshold, pois decisões baseadas em probabilidades imprecisas podem ser inconsistentes.

**Por que ajuste de threshold na decisão final?**

Por padrão, modelos de classificação utilizam um threshold fixo de 0.5 para converter probabilidades em classes. No entanto, esse valor é arbitrário e nem sempre adequado para problemas reais.

Neste projeto, o threshold foi ajustado buscando o maior valor de Specificity que ainda mantivesse Recall ≥ 95% — feito exclusivamente no conjunto de validação, sem qualquer contato com o conjunto de teste.

---

## ⚠️ Limitações do Modelo

Apesar dos bons resultados, este modelo possui limitações importantes que devem ser consideradas antes de uso em ambiente real:

**1. Dataset limitado e potencialmente enviesado**

O modelo foi treinado em um dataset específico, que pode não representar toda a população. Pode haver viés em relação a idade, gênero e perfil clínico. O modelo pode não generalizar bem para outros contextos (hospitais, países, etc.)

**2. Não substitui diagnóstico médico**

O modelo é uma ferramenta de triagem, não de diagnóstico. Não considera contexto clínico completo, não substitui exames ou avaliação profissional, e deve ser usado apenas como apoio à decisão.

**3. Sensibilidade vs Precisão (trade-off)**

O modelo foi otimizado para recall, priorizando a detecção de pacientes de alto risco. Isso reduz falsos negativos (casos graves não detectados), mas pode gerar maior carga operacional em um ambiente real por aumentar falsos positivos.

**4. Dependência de qualidade dos dados de entrada**

O desempenho do modelo depende diretamente da qualidade dos dados fornecidos. Erros de input ou dados incompletos podem gerar previsões incorretas.

**5. Modelo estático (não aprende em produção)**

O modelo atual não se atualiza automaticamente nem aprende com novos dados. Para uso real, seria necessário monitoramento contínuo e re-treinamento periódico.

**6. Interpretabilidade limitada**

Apesar do uso de Random Forest, o modelo não explica diretamente o "porquê" de cada decisão individual. Em aplicações críticas, técnicas como SHAP poderiam complementar a interpretabilidade.

---

## 🧠 Como funciona

1. Usuário insere dados clínicos do paciente na interface
2. Os dados passam por um pipeline de preprocessamento (normalização + encoding)
3. O modelo `RandomForestClassifier` calibrado realiza a predição
4. A interface retorna **Alto Risco** ou **Baixo Risco** em tempo real

---

## 🛠️ Tecnologias

`Python` · `Scikit-learn` · `RandomForestClassifier` · `Pandas` · `NumPy` · `Streamlit` · `Joblib` · `Matplotlib`

---

## 📁 Estrutura do Projeto
```
├── train.py                 # Treina o modelo do zero (execute antes do app)
├── model.ipynb              # Notebook de treinamento e avaliação
├── launcher.py              # Inicializador da aplicação
├── requirements.txt         # Dependências
├── models/                  # Gerado pelo train.py — ignorado pelo Git
│   └── modelo_cardiaco.pkl
└── src/
    ├── app/
    │    └── app.py
    ├── DataSet/
    │    └── heart_cleaned.csv        # Dataset
    └── img/
        
```

---

## 📦 Dataset

Dataset público do Kaggle com 918 registros de pacientes e 11 variáveis clínicas:

`Age` · `Sex` · `ChestPainType` · `RestingBP` · `Cholesterol` · `FastingBS` · `RestingECG` · `MaxHR` · `ExerciseAngina` · `Oldpeak` · `ST_Slope`

🔗 [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## ▶️ Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/treino258/Predict-Health-Risk-IA.git
cd Predict-Health-Risk-IA

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Treine o modelo
python train.py

# 4. Execute a aplicação
python launcher.py
```

---

## 👨‍💻 Autor

**Vitor Albuquerque** — Machine Learning & AI