import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import cv2
import mlflow
import mlflow.sklearn

import argparse

class AutoKNN:
    def __init__(self, k, data_path, mlflow_tracking_uri="./mlruns", mlflow_experiment_name="hand_signe_knn_experiment"):
        print("Initializing AutoKNN...")
        self.init_constants()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        self.k = k
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(self.data_path)

    def init_constants(self):
        print("Initializing constants...")
        self.TARGET_SIZE = (28, 28)
        self.ALPHABET_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


    def split_data(self, X, y):
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def load_data(self, path):
        print("Loading data...")
        image_arr = []
        image_name_file = []
        image_name_value = []
        counter = 0
        for letter in self.ALPHABET_LIST:
            letter_path = os.path.join(path, letter)
            for image_name in os.listdir(letter_path):
                image_path = os.path.join(letter_path, image_name)
                image = plt.imread(image_path)
                image = np.array(image)
                image = cv2.resize(image, self.TARGET_SIZE)
                image_arr.append(image)
                image_name_file.append(image)
                image_name_value.append(letter)
                counter += 1
                if counter % 100 == 0:
                    print(f"Loaded {counter} images")

        X = np.array(image_arr)
        y = np.array(image_name_value)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        return X_train, X_test, y_train, y_test

    def fit(self):
        print("Fitting model...")
        with mlflow.start_run():
            mlflow.log_param("n_neighbors", self.k)
            mlflow.log_param("test_size", 0.2)

            self.model = KNeighborsClassifier(n_neighbors=self.k)
            print("Training model...")

            self.model.fit(self.X_train, self.y_train)
            self.predict(self.X_test)
            accuracy, recall, f1, precision = self.evaluate()

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            classification_report = self.classification_repport()
            mlflow.log_text(classification_report, "classification_report.txt")

            self.plot_confusion_matrix()
            mlflow.log_artifact("confusion_matrix.png")

            self.roc_curve()
            mlflow.log_artifact("roc_curve.png")

    def predict(self, X):
        print("Predicting...")
        self.y_pred = self.model.predict(X)

    def evaluate(self):
        print("Evaluating...")
        accuracy = accuracy_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        return accuracy, recall, f1, precision

    def classification_repport(self):
        print("Generating classification report...")
        print(classification_report(self.y_test, self.y_pred))
        return classification_report(self.y_test, self.y_pred)
      
    def plot_confusion_matrix(self):
        print("Plotting confusion matrix...")
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig("confusion_matrix.png")
        plt.close()
    
    def roc_curve(self):
        print("Generating ROC curve...")
        
        # Binarize the labels for multi-class ROC
        y_test_binarized = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_test_binarized.shape[1]

        # Predict probabilities
        y_proba = self.model.predict_proba(self.X_test)

        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
            )

        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("ROC Curve for Multi-class Classification", fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

    def save_model(self, path="./knn_model.joblib"):
        print("Saving model...")
        self.model.save(path)


parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--data-path", type=str, default="data")
parser.add_argument("--mlflow-tracking-uri", type=str, default="./mlruns")
parser.add_argument("--experiment-name", type=str, default="hand_signe_knn_experiment")


args = parser.parse_args()
knn = AutoKNN(args.k, args.data_path, args.mlflow_tracking_uri, args.experiment_name)
knn.fit()
# knn.evaluate()

