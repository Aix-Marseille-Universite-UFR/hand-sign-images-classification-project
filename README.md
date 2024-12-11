
# Hand Sign Classifier with KNN

## Overview
This project implements a Hand Sign Classifier using the K-Nearest Neighbors (KNN) algorithm. The model is trained on a dataset of American Sign Language (ASL) hand signs, and aims to classify images of hand signs into their respective categories (letters from A to Y). The project also uses MLflow for tracking experiments and logging models. Additionally, it includes functionalities for hyperparameter tuning via GridSearchCV, and evaluation through performance metrics like accuracy, precision, and a confusion matrix.

## Project Structure
1. **Data Loading**: The dataset consists of grayscale images of ASL hand signs. Each image is resized to 28x28 pixels and flattened for use in machine learning.
2. **Model Training**: The model is trained using the KNN classifier. Hyperparameters are optimized using GridSearchCV with options for the number of neighbors (n_neighbors), weights (weights), and distance metric (metric).
3. **Model Evaluation**: The model's performance is evaluated using accuracy, precision, and a confusion matrix.
4. **MLflow Tracking**: The project uses MLflow for experiment tracking, model logging, and registering the best-performing model.
5. **Model Deployment**: The trained model is registered with MLflow’s model registry and can be used for inference on new images.

## Requirements
- Python 3.x
- MLflow
- Scikit-learn
- OpenCV
- Matplotlib
- Joblib
- Seaborn
- Requests

You can install the required libraries using pip:

```bash
pip install mlflow scikit-learn opencv-python matplotlib joblib seaborn requests lime
```
or  using python virtual environment
```bash
python -m venv env
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is the **ASL Hand Sign Dataset**, which can be found at:

[ASL Hand Sign Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Make sure to download and store the dataset locally on your machine.

### Dataset Path
```python
dataset_path = "/path/to/dataset/asl-dataset/asl-dataset"
```

## Steps

### 1. Data Loading and Preprocessing
Images from the dataset are loaded, resized to 28x28 pixels, and flattened into a 1D array to be used in the KNN model.

```python
for letter in alphabet_array:
    path_letter = os.path.join(dataset_path, "train", letter)
    for image_name in os.listdir(path_letter):
        image_path = os.path.join(path_letter, image_name)
        image = plt.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        resized_image = cv2.resize(image, target_size)
        image_arr.append(resized_image.flatten())
        image_name_value.append(letter)
```

### 2. Model Training with KNN and Hyperparameter Tuning
The model is trained using KNN with hyperparameter tuning using GridSearchCV. Several combinations of parameters such as the number of neighbors (n_neighbors), distance metric (metric), and weights are tested to find the best configuration.

```python
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

### 3. MLflow Experiment Tracking
MLflow is used to log all experiments, including model parameters, metrics, and the best model. Each run's configuration and results are saved to help track the performance of different hyperparameter settings.

```python
with mlflow.start_run(run_name=f"Version_{i}") as run:
    mlflow.log_params({'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric})
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.sklearn.log_model(model, "KNN_Model")
```

### 4. Model Evaluation
After training, the model’s performance is evaluated using accuracy, precision, and a confusion matrix. The confusion matrix provides insight into how well the model classifies each of the ASL hand signs.

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f"Confusion Matrix")
plt.show()
```

### 5. Model Registration and Deployment
Once the best model is found, it is registered in MLflow’s Model Registry for easy management and deployment.

```python
mlflow.register_model(model_uri, model_name)
```

### 6. Model Inference (Prediction)
The model can be used for inference on new images, where predictions are made based on the model registered in MLflow.

```python
def predict_img(image, reel_value):
    input_data = {"inputs": [image.tolist()]}
    response = requests.post(url, headers=headers, data=json.dumps(input_data))
    if response.status_code == 200:
        prediction = response.json()
        predicted_letter = alphabet_array[prediction["predictions"][0]]
        print(f"Real value: {alphabet_array[reel_value]}, Predicted: {predicted_letter}")
```

## Experiment and Model Tracking
All experiment details, including hyperparameters, performance metrics, and models, are logged in MLflow. After the best model is determined, it is saved and registered, and the logs are available for analysis.

## LIME Explanation for Model Predictions
To interpret model predictions, we use LIME (Local Interpretable Model-agnostic Explanations). LIME helps identify the most influential features (pixels) for a given prediction.
from lime.lime_tabular import LimeTabularExplainer

```python
# Initialize LIME explainer
lime_explainer = LimeTabularExplainer(
    X_train,
    feature_names=[f"Pixel {i}" for i in range(X_train.shape[1])],
    class_names=[str(label) for label in np.unique(y_train)],
    discretize_continuous=True
)

# Explain a test instance
instance = X_test[0]
lime_exp = lime_explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10
)

# Display explanation
lime_exp.show_in_notebook(show_table=True)
```
## Conclusion
This project showcases the use of the K-Nearest Neighbors (KNN) algorithm for classifying hand signs from the American Sign Language (ASL) dataset. It incorporates best practices for model training, evaluation, and tracking using MLflow. The model can be used for real-time predictions on hand sign images, providing a strong foundation for building sign language recognition systems.

## Authors
* **NEDDAY ANAS**
* **BELHOCINE MEHDI**
* **BOUDRA Ayoub**
* **KENZEDDINE Mohamed Amine**
 