![status](https://img.shields.io/badge/status-stable-brightgreen)  ![python](https://img.shields.io/badge/python-3.11-blue)
 
![tests](https://img.shields.io/badge/tests-pytest-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)
  
![build](https://github.com/treino258/Predict-Health-Risk-IA/actions/workflows/test.yml/badge.svg)


# 🫀 Predict Health Risk IA

> A cardiac risk screening tool using Machine Learning—from clinical data to real-time prediction.

📌 This project is for educational purposes and demonstrates Machine Learning best practices, including proper validation, threshold adjustment, and probability calibration. For use in clinical production, external validation, regulatory approval, and integration with healthcare professionals would be required.

![Interface](src/img/image.png)

---

## 🚀 Live Demo

🔗 https://predict-health-risk-ia-vitor-albuquerque.streamlit.app/

## 🎯 Business Goal

Cardiovascular diseases are the leading cause of death worldwide. Early diagnosis depends on specialized professionals and tests that are not always accessible. This project creates an intelligent screening tool that assists in identifying patients at high cardiac risk based on simple clinical data—democratizing a first level of preventive assessment.

> ⚠️ **Warning:** This tool is a screening aid and does not replace professional medical diagnosis.

---

## 💡 Solution & Impact

A binary classification model (**High Risk / Low Risk**) trained with real clinical data, validated with 5-fold cross-validation, and optimized via `RandomizedSearchCV` with a focus on **recall** — minimizing false negatives, which represent the error with the highest clinical impact in healthcare.

Delivered as a web application via **Streamlit**, allowing anyone to enter clinical data and receive an instant prediction, without the need for technical infrastructure.

---

## 🔍 Exploratory Data Analysis (EDA)

Before modeling, the data was thoroughly investigated to ensure quality and extract clinical insights.

**Data Quality**
172 patients had Cholesterol = 0—biologically impossible. These values were treated as missing and replaced by the group median (sick/healthy), preserving the clinical pattern of each class.

**Key Findings**
               (Median)    (Median)
| Variable   | Healthy  |    Disease   |  Difference |
|------------|-----------|-------------|------------|
| Age      | 51 years   | 57 years     | +11.8%     |
| MaxHR      | 150 bpm   | 126 bpm     | **-16.0%** |
| Cholesterol | 231 mg/dL | 246 mg/dL   | +6.3%      |
| Oldpeak    | 0.0 | 1.2 | **↑ forte** |            
| RestingBP  | 130 mmHg  | 132 mmHg    | +1.5%      |

**Clinical insights extracted:**

- **Oldpeak** is the most discriminating indicator—sick patients have significantly higher ST-segment depression, consistent with cardiological literature.
- **MaxHR** is inversely associated with the disease—a lower maximum heart rate indicates lower cardiac functional capacity.
- **FastingBS** (> 120) appears almost exclusively in sick patients, suggesting a strong association with metabolic-cardiovascular risk.
- **Cholesterol** shows high variability and significant outliers—noisy variable that required careful treatment.
- **Idade** is 6 years higher on average for sick patients, reinforcing the increasing risk with aging.

---

## 📊 Model Results

| Metric | Value |
|---------|-------|
| Accuracy | 88.0% |
| Recall — Alto Risco (classe 1) | **89,47%** |
| Specificity — Baixo Risco (classe 0) | **87,10%** |
| Precision — Alto Risco (classe 1) | 89,47% |
| F1-Score — Alto Risco (classe 1) | 89,47% |


---

**Confusion Matrix:**
```
[[54  8]   → 54 correct healthy | 8 false positives
 [ 8  68]]  → 8 false negatives    | 68 correct disease
```

The model correctly identifies **89.47% of patients with real cardiac risk—false negatives (sick classified as healthy)** are the most dangerous error in healthcare and were the main optimization criterion. The Specificity of 87.10% indicates that the model also avoids unnecessarily alarming healthy patients.

---
## 📐 Estimated Impact Simulation

> Each claim is labeled by evidence strength: **[Data-Supported]** = directly tied to a cited study | **[Possible]** = plausible extrapolation from cited evidence |

---

### 🔢 Projected Impact — 10,000 Patients Screened

> Conservative simulation based on the model's validated metrics and published epidemiological studies.  
> Assumes a screened adult population with ~20% true high-risk prevalence [Possible].

| Outcome | Value | Label |
|---|---|---|
| High-risk patients correctly identified (Recall 89.47%) | **~1,789 of 2,000** | [Data-Supported] |
| High-risk patients missed (False Negatives 10.53%) | ~211 | [Data-Supported] |
| Healthy patients correctly cleared (Specificity 87.10%) | ~6,968 of 8,000 | [Data-Supported] |
| Additional cases caught vs. traditional ACC/AHA algorithm | **+7.6%** | [Data-Supported] |

**Deaths potentially prevented over 5–10 years — three real-world scenarios:**

| Scenario | Treatment Adherence | Estimated Deaths Prevented | Label |
|---|---|---|---|
| Pessimistic | 37% *(US primary prevention registry)* | **~99** | [Possible] |
| Realistic | 57% *(pooled meta-analysis, primary prevention)* | **~153** | [Possible] |
| Optimistic | 100% *(full adherence, theoretical ceiling)* | **~268** | [Possible] |

*Calculated as: 1,789 flagged patients × adherence rate × 15% mortality reduction (RCT meta-analysis, n=1,053,772). The 15% figure is the most conservative reduction reported — other meta-analyses found up to 25%.*

---

### 📖 What Grounds These Numbers

**Recall 89.47% and Specificity 87.10%** are the model's validated metrics on the held-out test set, derived directly from the confusion matrix (54 true negatives, 68 true positives, 8 false negatives, 8 false positives out of 138 test patients).

**+7.6% more cases caught than traditional screening** **[Data-Supported]** comes from a prospective cohort study across 378,256 UK primary care patients comparing ML algorithms against the ACC/AHA guidelines. Neural networks caught 355 additional CVD cases (+7.6%) and Random Forest caught 191 more — using the same type of clinical variables this model uses (age, cholesterol, blood pressure, fasting glucose).
> *Weng et al. (2017). PLOS ONE. doi: 10.1371/journal.pone.0174944*

**15% mortality reduction** **[Data-Supported]** is the conservative lower bound from a meta-analysis of large statin RCTs. Early preventive treatment reduces major coronary events by 27%, stroke by 18%, and all-cause mortality by 15%. A broader network meta-analysis of 139 RCTs (n=1,053,772) confirmed statins (RR 0.81), BP-lowering agents (RR 0.82), and lifestyle interventions (RR 0.75) all significantly reduce cardiovascular mortality.
> *Thavendiranathan et al. (2006). PMC1884492 | NCBI Bookshelf (2025). NBK617276*

**37–57% real-world adherence range** **[Data-Supported]** is grounded in two published sources: a US registry study found only 37% of primary prevention patients were adherent to guideline-recommended statins; a 2025 systematic review and meta-analysis reported a pooled adherence rate of 57.5% specifically in primary prevention settings (vs. 64.4% in secondary prevention), with adherence declining further over time.
> *Martinez et al. (2023). PMC12354097 | Khunti et al. (2025). European Journal of Preventive Cardiology. doi: 10.1093/eurjpc/zwaf769*

**20% high-risk prevalence** **[Possible]** is a conservative estimate consistent with the Brazilian epidemiological context — CVD accounts for 31% of all deaths in Brazil, and a 2019 National Health Survey found only 0.5% of Brazilian adults present ideal cardiovascular health across all AHA metrics.
> *Ribeiro et al. (2016). Circulation. doi: 10.1161/CIRCULATIONAHA.114.008727 | Malta et al. (2023). PMC10069666*

---

### ⚠️ Simulation Assumptions & Limitations

This is a proof-of-concept estimate, not a clinical trial. The projected impact depends on assumptions that must be validated before any real-world deployment:

- **Population mismatch:** The model was trained on 918 patients from a Kaggle dataset. Performance on a different demographic profile may vary from the reported 89.47% recall.
- **Prevalence assumption:** The 20% high-risk prevalence is an estimate. A different base rate directly shifts all downstream numbers.
- **Treatment access gap:** Correctly flagging a patient only creates value if they can access follow-up care — a significant barrier in low-income Brazilian regions.
- **RCT-to-real-world gap:** The 15% mortality reduction comes from controlled trials; real-world impact is typically lower.

External validation on prospective Brazilian clinical data would be required to confirm these estimates.

---

## 🧠 Technical Decisions

**Why separate data into three sets?**

The classic train/test split creates a subtle problem: if the threshold is adjusted using the test set, the reported results are optimistic and do not reflect real performance on new data—this is called data leakage.

In this project, data was divided into three parts:

```
918 patients (100%)
├── Train  (70%) → trains the model and hyperparameters
├── Validation (15%) → adjusts the threshold here, without touching the test set
└── Test (15%) → reports the final result, touched only once
```

Each set has a single responsibility. The threshold was chosen in validation, and the test was used only to report the final number—ensuring an honest estimate of real performance.

**Why Recall as the main metric

In cardiac screening, the two types of errors have asymmetric costs:

- False Negative (high risk classified as low): patient does not receive medical attention → potentially fatal consequence.
- False Positive (low risk classified as high): patient undergoes unnecessary tests → controllable cost.

Therefore, minimizing False Negatives is a priority. Recall measures exactly this: of the patients who are actually at high risk, how many does the model correctly identify?

Accuracy would be a misleading metric here—with an imbalanced dataset, a model that classifies everything as "low risk" could have high accuracy and zero Recall.

**Why RandomForestClassifier?**

Three practical reasons for this problem:

- **Interpretability:** provides `feature_importances_`, allowing us to understand which clinical variables most influence risk—essential in a medical context.
- **Robustness to outliers:** clinical data has natural noise (e.g., 172 inconsistent cholesterol records treated via median imputation).
- **Performance without scaling:** unlike SVM or KNN, it does not require feature normalization, simplifying the inference pipeline.

**Why median imputation for cholesterol?**

172 records had cholesterol = 0, a clinically impossible value. Two options were considered:

- Remove the records: ~18% of the dataset would be lost—significant impact for a health model.

- Impute by median: preserves data and is robust to outliers, unlike the mean.

The median was chosen because it is less sensitive to extreme values present in the cholesterol distribution.

**Why 5-fold cross-validation in addition to train_test_split?**

`train_test_split` evaluates the model on a single partition—a result dependent on the luck of the draw. 5-fold cross-validation ensures that every data point was used for both training and testing, producing a more reliable estimate of the model's generalization capacity.

**Why calibrate probabilities (CalibratedClassifierCV)?**

Tree-based models, like RandomForest, often produce poorly calibrated probabilities—meaning the value returned by `predict_proba` does not correctly represent the true probability of the event. This directly impacts the use of thresholds, as decisions based on inaccurate probabilities can be inconsistent.

**Why adjust the threshold for the final decision?**

By default, classification models use a fixed threshold of 0.5 to convert probabilities into classes. However, this value is arbitrary and not always suitable for real-world problems.

In this project, the threshold was adjusted seeking the highest Specificity value that still maintained Recall ≥ 95%—done exclusively in the validation set, without any contact with the test set.

---

## ⚠️ Model Limitations

Despite the good results, this model has important limitations that must be considered before use in a real environment:

**1. Limited and potentially biased dataset**

The model was trained on a specific dataset, which may not represent the entire population. There may be bias regarding age, gender, and clinical profile. The model may not generalize well to other contexts (hospitals, countries, etc.).

**2. Does not replace medical diagnosis**

The model is a screening tool, not a diagnostic one. It does not consider the complete clinical context, does not replace exams or professional evaluation, and should be used only as decision support.

**3. Sensitivity vs. Precision (trade-off)**

The model was optimized for recall, prioritizing the detection of high-risk patients. This reduces false negatives (serious cases not detected), but may create a higher operational burden in a real environment by increasing false positives.

**4. Dependency on input data quality**

Model performance depends directly on the quality of the data provided. Input errors or incomplete data can generate incorrect predictions.

**5. Static model (does not learn in production)**

The current model does not update automatically or learn from new data. For real use, continuous monitoring and periodic re-training would be necessary.

**6. Limited interpretability**

Despite the use of Random Forest, the model does not directly explain the "why" of each individual decision. In critical applications, techniques like SHAP could complement interpretability.

---

## 🤖 AI Assistant

The application includes a Gemini-powered assistant capable of:

- Explaining model performance metrics
- Identifying the most important risk factors
- Running patient risk assessments through tool calling
- Answering questions about the model in natural language

The assistant uses Google's Gemini API and integrates directly with the trained machine learning model.

## 🧠 How It Works

### Risk Prediction

0. Acess 🔗 https://predict-health-risk-ia-vitor-albuquerque.streamlit.app/
1. User enters clinical data
2. Data is preprocessed
3. Calibrated Random Forest generates probabilities
4. Custom threshold converts probabilities into risk classes
5. Results are displayed in Streamlit

### AI Assistant

1. User asks a question
2. Gemini receives the request
3. Gemini calls available tools when necessary
4. The assistant returns explanations based on real model outputs
---

## 🛠️ Tecnologias

`Python` · `Scikit-learn` · `RandomForestClassifier` · `Pandas` · `NumPy` · `Streamlit` · `Joblib` · `Matplotlib`

---

## 📁 Estrutura do Projeto
```
├── train.py                     # Treina o modelo do zero (execute antes do app)
├── model.ipynb                  # Notebook de treinamento e avaliação
├── launcher.py                  # Inicializador da aplicação
├── requirements.txt             # Dependências
├── pyproject.toml            
├── models/                      # Gerado pelo train.py — ignorado pelo Git
│   └── modelo_cardiaco.pkl
└── src/
    ├── Data-Analysis.ipynb  
    ├── model.ipynb
    ├── app/
    │    └── app.py              # Streamlit app
    ├── DataSet/
    │    └── heart_cleaned.csv   # Dataset 
    ├── Configs/
    │    ├──  paths.py
    │    └──  train.yaml
    ├── img/                     # Images for documentation
    ├── llm/                     # Gemini integration
    │    ├── chat_tab.py
    │    ├── client.py
    │    └── tools.py
    ├── models/
         └── modelo_cardiaco.pkl

```    

---

## 📦 Dataset

Public Kaggle dataset with 918 patient records and 11 clinical variables:

`Age` · `Sex` · `ChestPainType` · `RestingBP` · `Cholesterol` · `FastingBS` · `RestingECG` · `MaxHR` · `ExerciseAngina` · `Oldpeak` · `ST_Slope`

🔗 [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/treino258/Predict-Health-Risk-IA.git
cd Predict-Health-Risk-IA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Run the application
python launcher.py
```
---
### 📚 References

| # | Citation |
|---|---|
| 1 | Weng et al. (2017). Can machine-learning improve cardiovascular risk prediction using routine clinical data? *PLOS ONE*. doi: 10.1371/journal.pone.0174944 |
| 2 | Thavendiranathan et al. (2006). Meta-analysis of large RCTs evaluating statin impact on cardiovascular outcomes. *PMC1884492* |
| 3 | NCBI Bookshelf (2025). Optimal primary prevention interventions for CVD: network meta-analysis of 139 RCTs. *NBK617276* |
| 4 | Khunti et al. (2025). Prevalence and determinants of adherence to statin therapy: systematic review and meta-analysis. *European Journal of Preventive Cardiology*. doi: 10.1093/eurjpc/zwaf769 |
| 5 | Martinez et al. (2023). Association between Statin Choice and adherence to statins. *PMC12354097* |
| 6 | Ribeiro et al. (2016). Cardiovascular Health in Brazil. *Circulation*. doi: 10.1161/CIRCULATIONAHA.114.008727 |
| 7 | Malta et al. (2023). Prevalence of ideal cardiovascular health — Brazilian National Health Survey 2019. *PMC10069666* |
| 8 | Souza et al. (2024). ML-based risk prediction for MACE in a Brazilian hospital. *PLOS ONE*. doi: 10.1371/journal.pone.0311719 |
| 9 | Roth et al. (2025). Global, Regional, and National Burden of CVDs, 1990–2023. *JACC*. doi: 10.1016/j.jacc.2025.08.015 |

---

## 👨‍💻 Autor

**Vitor Albuquerque** — Machine Learning & AI