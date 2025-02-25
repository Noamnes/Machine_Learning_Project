import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import tensorflow as tf
from tensorflow import keras


def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    errors = {
                    "train_error": 1 - accuracy_score(y_train, y_pred_train),
                    "test_error": 1 - accuracy_score(y_test, y_pred_test)
                }
    return errors
    
def fit_evaluate_models(models_list, X_train, X_test, y_train, y_test):
    acc_dict = {}
    for i, model in enumerate(models_list):
        model.fit(X_train, y_train)
        acc_dict[f'{model.__class__.__name__} - model {i+1}'] = evaluate_model(model, X_train, X_test, y_train, y_test)
    return acc_dict

def display_metrics_and_confusion_matrix(model_name, y_true, y_pred):
    """
    Displays accuracy, classification report (as a DataFrame), and a confusion matrix heatmap.
    """
    acc = accuracy_score(y_true, y_pred)

    # Convert classification report to DataFrame
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T

    # Print accuracy
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")

    # Display the classification report as a DataFrame
    display(report_df.style.background_gradient(cmap="Blues").format("{:.2f}"))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def load_preprocess_mnist_data(scale_features=True, flatten_image=True,  use_pca=False, pca_variance=0.95):
    """
    Loads and preprocesses the MNIST dataset with optional scaling, flattening, and PCA reduction.

    Parameters
    ----------
    scale_features : bool, default=True
        If True, scales pixel values to range [0,1] by dividing by 255
    flatten_image : bool, default=True
        If True, flattens 28x28 images into 784-dimensional vectors
    use_pca : bool, default=False
        If True, applies PCA dimensionality reduction
    pca_variance : float, default=0.95
        The desired explained variance ratio when using PCA

    Returns
    -------
    X_train : ndarray
        Training data, shape (n_samples, n_features)
    X_test : ndarray
        Test data, shape (n_samples, n_features)
    y_train : ndarray
        Training labels, shape (n_samples,)
    y_test : ndarray
        Test labels, shape (n_samples,)
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    if flatten_image:
        X_train = X_train.reshape((X_train.shape[0], 28 * 28))
        X_test = X_test.reshape((X_test.shape[0], 28 * 28))
    
    if scale_features:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    if use_pca:
        pca = PCA(n_components=pca_variance)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
    return X_train, X_test, y_train, y_test

def build_baseline_neural_network(input_dim, num_classes=10):
    """
    Builds a simple feed-forward neural network using Keras Sequential API.

    Arguments:
        input_dim (int): Dimensionality of input features (784 for raw MNIST, or fewer if PCA is used).
        num_classes (int): Number of output classes. Default is 10 for digits [0..9].

    Returns:
        A compiled tf.keras Sequential model.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def train_baseline_nn(X_train, y_train, X_test, y_test, epochs=5, batch_size=128):
    """
    Trains the baseline neural network on the training data. Evaluates on the test set.

    Arguments:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        model: The trained neural network.
        test_accuracy (float): Accuracy on the test set.
    """
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    display_metrics_and_confusion_matrix("Baseline NN", y_test, y_pred)
    return model

# return options:
    # 1. train|test accuracy
    # 2. train&test accuracy
    # 3. confusion_matrix
    # 4. precision, recall, f1-score

# display options: all the return options and more.

# 3.1 KNN
def train_knn(X_train, y_train, X_test, y_test, k=3):
    """
    Trains a K-Nearest Neighbors model and evaluates on the test set.

    Arguments:
        k (int): Number of neighbors.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    display_metrics_and_confusion_matrix("KNN", y_test, y_pred)
    return knn


# 3.2 SVM
def train_svm(X_train, y_train, X_test, y_test, kernel='rbf'):
    """
    Trains a Support Vector Machine model with the specified kernel.

    Arguments:
        kernel (str): Kernel type ('linear', 'rbf', 'poly', etc.).
    """
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    display_metrics_and_confusion_matrix("SVM", y_test, y_pred)
    return svm


# 3.3 Decision Tree
def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=None):
    """
    Trains a Decision Tree classifier.

    Arguments:
        max_depth (int or None): The maximum depth of the tree. If None, no maximum.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    display_metrics_and_confusion_matrix("Decision Tree", y_test, y_pred)
    return dt

# 3.4 AdaBoost
def train_adaboost(X_train, y_train, X_test, y_test, n_estimators=50):
    """
    Trains an AdaBoost classifier with decision trees as base estimators.

    Arguments:
        n_estimators (int): Number of weak learners.
    """
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    display_metrics_and_confusion_matrix("AdaBoost", y_test, y_pred)
    return ada


# 3.5 Random Forest
def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
    """
    Trains a Random Forest classifier.

    Arguments:
        n_estimators (int): Number of trees in the forest.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    display_metrics_and_confusion_matrix("Random Forest", y_test, y_pred)
    return rf

###############################################################################
# 4. OPTIONAL ENSEMBLE
###############################################################################
def train_ensemble(models, X_test, y_test):
    """
    Simple demonstration of a majority-voting ensemble.
    `models` is a list of trained (model_name, model) tuples.
    We'll do a simple majority vote on predictions.
    """
    predictions = [model.predict(X_test).reshape(-1, 1) for _, model in models]
    predictions = np.concatenate(predictions, axis=1)
    final_preds = [np.bincount(row).argmax() for row in predictions]
    display_metrics_and_confusion_matrix("Ensemble Majority Voting", y_test, final_preds)
    return np.array(final_preds)


###############################################################################
# 5. EVALUATION FUNCTIONS
###############################################################################
def print_metrics_and_confusion_matrix(model_name, y_true, y_pred):
    """
    Prints accuracy, classification report, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
