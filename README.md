# 🌸 Iris Flower Classification - Machine Learning Project

An end-to-end Machine Learning project that classifies Iris flowers into species based on their measurements using Logistic Regression.

---

## 🚀 Project Overview

This project uses the famous Iris dataset to build a machine learning model capable of predicting flower species based on:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model is trained, saved, and deployed with a simple web interface using Streamlit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Scaling using StandardScaler
4. Model Training using Logistic Regression
5. Model Evaluation
6. Model Saving using Joblib
7. Web App Deployment using Streamlit

---

## 📁 Project Structure

```
Iris-Flower-Classification/
│
├── model/
│   ├── iris_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── iris_analysis.ipynb
│
├── app.py
├── ui.py
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd Iris-Flower-Classification
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

---

### 3️⃣ Run Console Version

```bash
python app.py
```

---

### 4️⃣ Run Web Application

```bash
python -m streamlit run ui.py
```

The app will open automatically in your browser.

---

## 🌺 Sample Prediction

Input:
```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

Output:
```
Iris-setosa
```

---

## 🎯 Key Learnings

- Data preprocessing and visualization
- Feature scaling importance
- Logistic Regression implementation
- Model serialization using Joblib
- Building ML web apps using Streamlit

---

## 📌 Future Improvements

- Add model accuracy display in UI
- Deploy the app online
- Add more ML models for comparison
- Improve UI styling

---

## 👩‍💻 Author

Komal  
Aspiring Data Scientist & Machine Learning Enthusiast

---

## ⭐ If you like this project, give it a star!
