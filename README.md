# Email Spam Detection using Decision Tree

This project implements a **text classification pipeline** to detect **spam vs ham emails** using Natural Language Processing (NLP) techniques and a **Decision Tree classifier**. It includes text preprocessing, feature extraction, exploratory analysis, model training, and evaluation.

---

## 📌 Project Objective

To classify emails into:

* **Ham (0)** – legitimate emails
* **Spam (1)** – unwanted or promotional emails

The project demonstrates how raw text data can be converted into numerical features and used in a supervised ML model.

---

## 🛠️ Tools & Libraries

* **Python**
* **Pandas** – data handling
* **Matplotlib** – visualization
* **Seaborn** – statistical plots
* **Scikit-learn** – NLP, model training, evaluation

---

## 📂 Dataset

**Input File:** `email.csv`

### Key Columns

* `Message` – email text content
* `Category` – spam or ham label

---

## 🔄 Project Workflow

### 1. Data Loading & Encoding

* Loaded dataset and inspected unique categories
* Encoded target labels using **LabelEncoder**

  * `0 → Ham`
  * `1 → Spam`

---

### 2. Feature Engineering

* Created a new feature: **Message Length**
* Helps analyze structural differences between spam and ham emails

---

### 3. Exploratory Data Analysis (EDA)

* **Category distribution** bar plot
* **Message length distribution** by category

---

### 4. Text Vectorization

* Used **CountVectorizer** with:

  * English stopwords removal
  * Maximum 5000 features
* Converted text into Bag-of-Words representation

---

### 5. Word Frequency Analysis

* Extracted and visualized **Top 20 most frequent words** in training data
* Helps understand dominant terms in emails

---

### 6. Train-Test Split

* 80% training data
* 20% testing data
* Fixed random state for reproducibility

---

### 7. Model Training

* Algorithm: **Decision Tree Classifier**
* Trained on vectorized text features

---

### 8. Model Evaluation

Metrics used:

* **Accuracy Score**
* **Classification Report** (Precision, Recall, F1-score)
* **Confusion Matrix** visualization

---

## 📈 Visual Outputs

* Category distribution
* Message length histogram by class
* Top frequent words bar chart
* Confusion matrix heatmap

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Place `email.csv` in the project directory
4. Run the Python script

---

## 📌 Use Cases

* Email spam filtering systems
* NLP text classification practice
* Binary classification projects
* Portfolio project for Data Science / AI roles

---

## 👤 Author

**Khubaib**
Aspiring AI Engineer | NLP & Machine Learning

---

## ⭐ Notes

* Decision Trees are easy to interpret but may overfit
* Performance can be improved using:

  * TF-IDF Vectorization
  * Naive Bayes or Logistic Regression
  * Hyperparameter tuning

---

If this project helped you, feel free to ⭐ the repository and extend it with more advanced NLP techniques.
