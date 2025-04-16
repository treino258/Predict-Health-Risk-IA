# 🫀 Predict Health Risk IA

Este projeto utiliza aprendizado de máquina para prever o risco de um paciente desenvolver **doença cardíaca**, com base em dados clínicos como idade, colesterol, pressão arterial, tipo de dor no peito, entre outros.

O modelo foi treinado com dados reais de pacientes (dataset público do Kaggle) e alcançou bons resultados de acurácia. A interface foi criada com Streamlit para que qualquer pessoa possa interagir com a IA de forma fácil e intuitiva.

---

## 💻 Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## 🧠 O que o modelo faz?

1. Carrega os dados de entrada fornecidos pelo usuário (paciente)
2. Codifica automaticamente os dados categóricos
3. Usa um modelo `RandomForestClassifier` previamente treinado
4. Retorna o resultado da previsão: **Alto Risco** ou **Baixo Risco**

---

## 📊 Resultados do Modelo

- Acurácia com **train_test_split**: `86.41%`
- Acurácias por rodada (**Cross Validation**):
  - `[0.8804, 0.8260, 0.8423, 0.8360, 0.7431]`
- **Média de acurácia (cross-validation)**: `82.56%`

---

## ▶️ Como executar o projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/predict-health-risk-ia.git
