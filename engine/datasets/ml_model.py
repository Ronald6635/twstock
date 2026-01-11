"""
Machine Learning Model Module

This module implements machine learning models for stock price prediction using scikit-learn.
It provides functions to train, evaluate, and visualize traditional ML models on financial data.

Key features:
- Polynomial linear regression with hyperparameter tuning
- Support Vector Regression (SVR) with RandomizedSearchCV
- Model performance evaluation using R² score
- Visualization of predictions vs actual values

Architecture notes:
- Uses scikit-learn for ML implementations
- Designed for tabular financial features and regression tasks
- Includes data preprocessing with StandardScaler and PolynomialFeatures
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from scipy.spatial.distance import cdist
from typing import Any, Dict, Optional


__all__ = ["regression_models", "classification_models", "unsupervised_model"]

def regression_models(
     X_train: np.ndarray,
     y_train: np.ndarray,
     X_test: np.ndarray,
     y_test: np.ndarray,
     show_plots: bool = True
) -> Dict[str, Any]:
    """
    Fit machine learning models (linear regression and SVR) to the data, evaluate performance, and visualize results.
    
    This function trains a polynomial linear regression model and an SVR model on the provided features and target,
    evaluates their performance using R² score, and generates scatter plots and time series plots for comparison.
    The models are trained on the provided training data and tested on the provided test data.
    
    Args:
        X_train: Training feature matrix as a numpy array of shape (n_train_samples, n_features).
        y_train: Training target vector as a numpy array of shape (n_train_samples,).
        X_test: Test feature matrix as a numpy array of shape (n_test_samples, n_features).
        y_test: Test target vector as a numpy array of shape (n_test_samples,).
        
    Returns:
        None: This function prints results and displays plots but does not return any values.
        
    Raises:
        ValueError: If X_train, y_train, X_test, or y_test have incompatible shapes or insufficient data.
               
    Note:
        - Requires sklearn, matplotlib, and numpy to be installed.
        - The polynomial degree is fixed at 1 for simplicity; adjust as needed.
        - SVR uses RandomizedSearchCV for hyperparameter tuning with limited iterations.
        - Plots are displayed using matplotlib; ensure a display environment is available.
    """
    # Validate input shapes
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test must have the same number of samples")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and test sets must contain at least one sample")
    
    # Polynomial Linear Regression

    pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),StandardScaler())
    X_train_prep = pipeline.fit_transform(X_train)
    X_test_prep = pipeline.transform(X_test)
    
    lr_model = sklearn.linear_model.LinearRegression()
    lr_model.fit(X_train_prep, y_train)
    r2_score_lr = lr_model.score(X_test_prep, y_test)
    print(f"R² score of the linear regression model: {r2_score_lr:.4f}")
    
    # Visualization for Linear Regression
    y_pred_lr = lr_model.predict(X_test_prep)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.7, label='Predicted vs Actual')
    plt.xlabel('Actual Close Prices')
    plt.ylabel('Predicted Close Prices')
    plt.title('Actual vs Predicted Close Prices (Polynomial Linear Regression)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(y_pred_lr, label='Predicted', alpha=0.7)
        plt.plot(y_test, '.-', label='Actual', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Close Price')
        plt.title('Predicted vs Actual Close Prices Over Samples (Polynomial Linear Regression)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Support Vector Regression with hyperparameter tuning
    param_distributions = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'epsilon': [0.01, 0.1, 0.5, 1.0],
        'kernel': ['rbf', 'linear', 'poly']
    }
    svr_model = SVR()
    random_search = RandomizedSearchCV(
        svr_model, 
        param_distributions, 
        n_iter=10, 
        cv=3, 
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    random_search.fit(X_train_prep, y_train)
    r2_score_svr = random_search.score(X_test_prep, y_test)
    print(f"R² score of the SVR model: {r2_score_svr:.4f}")
    
    # Visualization for SVR
    y_pred_svr = random_search.predict(X_test_prep)
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(y_pred_svr, label='SVR Predicted', alpha=0.7)
        plt.plot(y_test, '.-', label='Actual', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Close Price')
        plt.title('SVR Predicted vs Actual Close Prices Over Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        "linear_model": lr_model,
        "svr_model": random_search,
        "r2_lr": r2_score_lr,
        "r2_svr": r2_score_svr,
        "y_pred_lr": y_pred_lr,
        "y_pred_svr": y_pred_svr,
    }

def classification_models(
    X_train: np.ndarray,
    y_train_cls: np.ndarray,
    X_test: np.ndarray,
    y_test_cls: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Implement and evaluate widely-used classification models for stock price movement prediction.
    
    This function trains Logistic Regression, Random Forest, and SVC models on the provided
    features and target, evaluates their performance using accuracy score and classification reports,
    and visualizes the best model's predictions. It handles both binary and multiclass targets by
    automatically discretizing continuous targets and using multiclass-compatible solvers.
    
    Args:
        X_train: Training feature matrix as a numpy array of shape (n_train_samples, n_features).
        y_train: Training target vector as a numpy array of shape (n_train_samples,).
        X_test: Test feature matrix as a numpy array of shape (n_test_samples, n_features).
        y_test: Test target vector as a numpy array of shape (n_test_samples,).
        show_plots: If True, displays a plot comparing predicted and actual values for the best model.
    
    Returns:
        Dict[str, Any]: A dictionary containing the trained models, accuracy scores,
                       and classification reports for each model. Keys are model names
                       ("logistic_regression", "random_forest", "svc").
    
    Raises:
        ValueError: If X_train, y_train, X_test, or y_test have incompatible shapes or insufficient data.
    
    Note:
        - Continuous targets are automatically discretized into binary classes (1 for positive movement, 0 otherwise).
        - Logistic Regression uses the 'lbfgs' solver to support multiclass classification.
        - Hyperparameter tuning is performed using RandomizedSearchCV for Random Forest and SVC.
        - Classification reports handle undefined metrics (e.g., no predicted samples for a class) by setting them to 0.0.
    """
    # Validate input shapes
    if X_train.shape[0] != y_train_cls.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    if X_test.shape[0] != y_test_cls.shape[0]:
        raise ValueError("X_test and y_test must have the same number of samples")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and test sets must contain at least one sample")
    
    
    print("unique train clusters:", np.unique(y_train_cls))
    print("unique test clusters:", np.unique(y_test_cls))
    # Ensure discrete labels: Convert continuous values to binary (1 for price increase, 0 otherwise)
    if np.issubdtype(y_train_cls.dtype, np.floating):
        y_train_cls = (y_train_cls > 0).astype(int)
        y_test_cls = (y_test_cls > 0).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Logistic Regression Model (updated solver for multiclass support)
    lr_classifier = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
    lr_classifier.fit(X_train_scaled, y_train_cls)
    y_pred_lr = lr_classifier.predict(X_test_scaled)
    results["logistic_regression"] = {
        "model": lr_classifier,
        "accuracy": accuracy_score(y_test_cls, y_pred_lr),
        "report": classification_report(y_test_cls, y_pred_lr, zero_division=0)
    }
    
    # 2. Random Forest Classifier
    rf_params = {
        'n_estimators': [10, 15, 20, 25],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [10, 15, 20]
    }
    rf_classifier = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train_scaled, y_train_cls)
    y_pred_rf = rf_classifier.predict(X_test_scaled)
    results["random_forest"] = {
        "model": rf_classifier.best_estimator_,
        "accuracy": accuracy_score(y_test_cls, y_pred_rf),
        "report": classification_report(y_test_cls, y_pred_rf, zero_division=0)
    }
    
    # 3. Support Vector Classification (SVC)
    svc_params = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svc_classifier = RandomizedSearchCV(
        SVC(random_state=42),
        svc_params,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    svc_classifier.fit(X_train_scaled, y_train_cls)
    y_pred_svc = svc_classifier.predict(X_test_scaled)
    results["svc"] = {
        "model": svc_classifier.best_estimator_,
        "accuracy": accuracy_score(y_test_cls, y_pred_svc),
        "report": classification_report(y_test_cls, y_pred_svc, zero_division=0)
    }
    
    # Print summary
    for model_name, data in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {data['accuracy']:.4f}")
        print(f"Classification Report:\n{data['report']}")
    
    # Plot of the model with highest accuracy
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model_data = results[best_model_name]
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_data['accuracy']:.4f}")
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.title(f'Predictions by {best_model_name}')
        plt.plot(best_model_data['model'].predict(X_test_scaled), label='Predicted', alpha=0.7)
        plt.plot(y_test_cls, '.-', label='Actual', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Class Label')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
        plt.figure(figsize=(10, 6))
        plt.title(f'Predictions by {best_model_name}')
        plt.plot(y_train[best_model_data['model'].predict(X_test_scaled)], label='Predicted', alpha=0.7)
        plt.plot(y_test, '.-', label='Actual', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return results


def clustering_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int = 3,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Perform unsupervised learning (clustering and PCA) on the stock dataset.

    This function applies Gaussian Mixture Model (GMM) clustering to discover 
    hidden patterns and PCA for dimensionality reduction and visualization.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (used only for evaluation/visualization).
        X_test: Test feature matrix.
        y_test: Test target vector (used only for evaluation/visualization).
        n_clusters: Number of components (clusters) for GMM (default: 3).
        show_plots: Whether to display clustering and PCA results.

    Returns:
        Dict[str, Any]: A dictionary containing clustering results, PCA components,
                       silhouette scores, centroids, and representative samples.
    """
    print("\n--- Running Unsupervised Models (Clustering & PCA) ---")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Gaussian Mixture Model (GMM) Clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    train_clusters = gmm.fit_predict(X_train_scaled)
    test_clusters = gmm.predict(X_test_scaled)

    # Get centroids (means of each component)
    centroids = gmm.means_

    # Find representative samples (closest to each centroid)
    distances = cdist(X_train_scaled, centroids)
    representative_indices = np.argmin(distances, axis=0)
    representative_samples = X_train[representative_indices]
    representative_targets = y_train[representative_indices]    

    # Evaluate using Silhouette Score
    sil_score = silhouette_score(X_train_scaled, train_clusters)
    print(f"GMM Silhouette Score: {sil_score:.4f}")

    # 2. PCA for Visualization
    if X_train_scaled.shape[1] < 2:
        print("PCA visualization requires at least 2 clusters. Skipping PCA.")
        return {
            "gmm_model": gmm,
            "silhouette_score": sil_score,
            "train_clusters": train_clusters,
            "test_clusters": test_clusters,
            "centroids": centroids,
            "representative_samples": representative_samples,
            "representative_indices": representative_indices,
            "representative_targets": representative_targets
        }
    else:
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
    
    if show_plots and n_clusters >= 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('GMM Clustering Visualization (PCA Projection)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        "gmm_model": gmm,
        "pca_model": pca,
        "silhouette_score": sil_score,
        "train_clusters": train_clusters,
        "test_clusters": test_clusters,
        "pca_components": X_train_pca,
        "centroids": centroids,
        "representative_samples": representative_samples,
        "representative_indices": representative_indices,
        "representative_targets": representative_targets
    }

# Example usage and testing
if __name__ == '__main__':
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Simulate stock features and target
    X = np.random.randn(n_samples, n_features)
    # Create some correlation with target
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5 + 100

    # Create a binary classification target for demonstration (e.g., price movement up/down)
    y_class = (y > np.median(y)).astype(int)

    # Split data (80% train, 20% test)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_train_class, y_test_class = y_class[:split_idx], y_class[split_idx:]

    # Train regression models
    print("\n--- Running Regression Models ---")
    results_reg = regression_models(X_train, y_train, X_test, y_test, show_plots=False)
    print("\nRegression model training completed successfully!")
    print(f"Final R² (LR): {results_reg['r2_lr']:.4f}, R² (SVR): {results_reg['r2_svr']:.4f}")

    # Semi-supervised Learning (Using clusters as additional features)
    results_unsup = clustering_model(X_train, y_train, X_test, y_test, show_plots=False)
    
    # Augment feature matrices with cluster labels to potentially improve classification
    X_train_augmented = np.column_stack((X_train, results_unsup["train_clusters"]))
    X_test_augmented = np.column_stack((X_test, results_unsup["test_clusters"]))

    # Train classification models with augmented features
    print("\n--- Running Classification Models (Augmented with Clusters) ---")
    # Pass both classification labels and original continuous targets for plotting
    results_clf = classification_models(
        X_train_augmented,
        y_train_class,
        X_test_augmented,
        y_test_class,
        y_train,
        y_test,
        show_plots=True
    )
    print("\nClassification model training completed successfully!")
    for model_name, data in results_clf.items():
        print(f"{model_name} Accuracy: {data['accuracy']:.4f}")