# 🌸 Iris Flower Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Level](https://img.shields.io/badge/Level-Beginner-lightgrey)

A beginner-friendly **machine learning project** that trains and compares three classification algorithms on the classic Iris dataset to predict a flower's species from its measurements.

---

## 👤 Project Details

| | |
|---|---|
| **Author** | Anshuman Kapoor |
| **Course** | B.Tech CSE (AI & ML), 1st Year, 1st Semester |
| **University** | COER University |
| **GitHub** | [Anshu0629](https://github.com/Anshu0629) |

---

## 🎯 Objective

Predict the species of an Iris flower (`setosa`, `versicolor` or `virginica`) from four measurements, and find out which algorithm performs best.

**Input features**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Output:** Flower species

---

## 📊 Dataset

The Iris dataset ships with scikit-learn, so **no download is needed**.

- **Samples:** 150 (50 per species)
- **Features:** 4
- **Classes:** 3
- **Missing values:** 0

---

## 🤖 Algorithms Used

| Algorithm | Accuracy |
|---|---|
| Logistic Regression | 96.67% |
| K-Nearest Neighbors (k=5) | 100.00% |
| Decision Tree (max depth 3) | 96.67% |

**Best model:** K-Nearest Neighbors

> Results come from an 80/20 train-test split with `random_state=42`.

---

## 🛠️ Technologies

Python · pandas · NumPy · matplotlib · scikit-learn

---

## 🗂️ Project Structure

```
iris-flower-classifier/
├── iris_classifier.py   # Main script
├── requirements.txt     # Python libraries needed
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

1. **Load** the dataset into a pandas DataFrame
2. **Explore** shape, class counts and missing values
3. **Split** data: 80% training, 20% testing
4. **Train** three different models
5. **Compare** their accuracy and pick the best
6. **Evaluate** the best model with a classification report and confusion matrix
7. **Predict** the species of a new flower

---

## 🚀 How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/Anshu0629/iris-flower-classifier.git
   cd iris-flower-classifier
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the project**
```bash
   python iris_classifier.py
```

---

## 💻 Sample Output

```
STEP 2: TRAINING AND COMPARING MODELS
Logistic Regression    Accuracy: 96.67%
K-Nearest Neighbors    Accuracy: 100.00%
Decision Tree          Accuracy: 96.67%

STEP 3: BEST MODEL -> K-Nearest Neighbors
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

STEP 4: PREDICTING A NEW FLOWER
Measurements (cm): [5.1, 3.5, 1.4, 0.2]
Predicted species: setosa
```

The script also saves three charts in the project folder: `iris_scatter.png`, `model_comparison.png` and `confusion_matrix.png`.

---

## 🔮 Future Improvements

- [ ] Add more models (SVM, Random Forest)
- [ ] Use **cross-validation** instead of a single train/test split
- [ ] Take flower measurements as user input
- [ ] Build a small web app using **Streamlit**

---

## 🙋 Author

**Anshuman Kapoor**
B.Tech CSE (AI & ML), COER University
[GitHub: Anshu0629](https://github.com/Anshu0629)

⭐ If you found this project helpful, consider giving it a star!
