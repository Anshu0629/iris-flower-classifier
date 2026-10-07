"""
Iris Flower Classifier
----------------------
A beginner-friendly machine learning project that compares three
classification algorithms on the famous Iris dataset.

Run:  python iris_classifier.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():
    """Load the Iris dataset as a pandas DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris


def explore_data(df):
    """Print a quick summary of the dataset."""
    print("=" * 50)
    print("STEP 1: EXPLORING THE DATA")
    print("=" * 50)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nShape (rows, columns):", df.shape)
    print("\nSamples per species:")
    print(df["species"].value_counts())
    print("\nMissing values:", df.isnull().sum().sum())


def plot_data(df):
    """Save a scatter plot of petal length vs petal width."""
    plt.figure(figsize=(7, 5))
    for species, group in df.groupby("species"):
        plt.scatter(
            group["petal length (cm)"],
            group["petal width (cm)"],
            label=species,
        )
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Iris Dataset: Petal Length vs Petal Width")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iris_scatter.png", dpi=120)
    plt.close()
    print("\nSaved plot: iris_scatter.png")


def train_and_compare(X_train, X_test, y_train, y_test):
    """Train three models and return their accuracies and fitted objects."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    }

    print("\n" + "=" * 50)
    print("STEP 2: TRAINING AND COMPARING MODELS")
    print("=" * 50)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name:<22} Accuracy: {accuracy * 100:.2f}%")

    return models, results


def plot_comparison(results):
    """Save a bar chart comparing model accuracies."""
    plt.figure(figsize=(7, 4))
    plt.bar(results.keys(), [v * 100 for v in results.values()],
            color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.close()
    print("Saved plot: model_comparison.png")


def evaluate_best(models, results, X_test, y_test, class_names):
    """Show a detailed report and confusion matrix for the best model."""
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    predictions = best_model.predict(X_test)

    print("\n" + "=" * 50)
    print(f"STEP 3: BEST MODEL -> {best_name}")
    print("=" * 50)
    print(classification_report(y_test, predictions, target_names=class_names))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=class_names, cmap="Blues"
    )
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved plot: confusion_matrix.png")
    return best_model


def predict_new_flower(model, class_names):
    """Predict the species of a brand-new flower."""
    # sepal length, sepal width, petal length, petal width (all in cm)
    new_flower = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    prediction = model.predict(new_flower)[0]
    print("\n" + "=" * 50)
    print("STEP 4: PREDICTING A NEW FLOWER")
    print("=" * 50)
    print("Measurements (cm):", new_flower.values.tolist()[0])
    print("Predicted species:", class_names[prediction])


def main():
    df, iris = load_data()
    explore_data(df)
    plot_data(df)

    X = df[iris.feature_names]
    y = df["target"]

    # 80% of data for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models, results = train_and_compare(X_train, X_test, y_train, y_test)
    plot_comparison(results)
    best_model = evaluate_best(models, results, X_test, y_test, iris.target_names)
    predict_new_flower(best_model, iris.target_names)

    print("\nDone! Check the saved .png files in this folder.")


if __name__ == "__main__":
    main()
