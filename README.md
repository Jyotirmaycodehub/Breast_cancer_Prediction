# 🔬 Breast Cancer Prediction using Machine Learning

Welcome to our **Team ML Project** where we tackle one of the most critical medical challenges—**breast cancer detection**. This predictive model uses **Random Forest Classifier** on clinical data to classify tumors as **benign** or **malignant**.

> 🎯 **Goal:** Empower early diagnosis through machine learning and data analysis.

---

## 👥 Meet the Team

| Name             | Role              |
|------------------|-------------------|
| Jyotirmay Das      | **Team Leader & Data Preprocessing & Visualization** 🧠✨ |
| Ipsita Maji         |Data modeling and ML pipelining |
| Jannutul Bushra        | Evaluation & Metrics |
| Shamindra Das        | Documentation & Reporting |
| Sayan Kumer Chowbey        | Deployment Planning / GUI Ideas |

---

## 🧠 Tools & Technologies

- 🐍 Python 3.x  
- 🧺 Pandas, NumPy – Data handling  
- 📊 Matplotlib, Seaborn – Visualizations  
- 🤖 Scikit-learn – ML modeling & evaluation  
- 📁 Jupyter Notebook – Workflow presentation

---

## 📂 Dataset

- **Source:** [Breast Cancer Dataset](21239_breast_cancer_prediction_data.csv)
- **Target Variable:**  
  - `0`: Malignant (cancerous)  
  - `1`: Benign (non-cancerous)

---

## 🔍 Workflow Summary

### 🔹 Data Preprocessing
- Checked for nulls & duplicates
- Outlier detection on `mean perimeter` using IQR
- Cleaned data for optimal modeling

### 🔹 Feature Engineering
- Feature scaling via `StandardScaler`
- Train/Test split (80/20)

### 🔹 Model Training
- Model: `RandomForestClassifier(n_estimators=100, random_state=42)`
- Trained on scaled data
- Prediction & probability scoring

### 🔹 Model Evaluation
- ✅ Accuracy: ~95%  
- 📉 Confusion Matrix  
- 📈 ROC-AUC Curve  
- 🧾 Classification Report

---

## 📊 Visual Insights

- 📦 Boxplot to detect outliers
- 🧬 ROC Curve: Shows model sensitivity
- 💡 Classification report for precision, recall & F1-score

---

👨‍💻 Developer: Jyotirmay
📧 Email: [jyotirmay1999das@gmail.com]
