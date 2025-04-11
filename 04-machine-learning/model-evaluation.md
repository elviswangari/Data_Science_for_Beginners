# Model Evaluation and Optimization

[⬅️ Previous: Feature Engineering](feature-engineering.md) | [Next: Git and GitHub ➡️](../00-Github/Git-Github.md)

## Learning Objectives

By the end of this section, you will:

1. Understand different model evaluation metrics
2. Learn cross-validation techniques
3. Master hyperparameter tuning
4. Identify and handle overfitting/underfitting
5. Create model validation pipelines

## 1. Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import numpy as np

def evaluate_classification(y_true, y_pred, y_prob=None):
    """Comprehensive evaluation of classification model"""
    
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # ROC-AUC if probability predictions are available
    if y_prob is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    # Confusion Matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return results
```

### Regression Metrics

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                           r2_score, explained_variance_score)

def evaluate_regression(y_true, y_pred):
    """Comprehensive evaluation of regression model"""
    
    results = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    return results
```

## 2. Cross-Validation Techniques

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score

def perform_kfold_cv(model, X, y, n_splits=5, metric='accuracy'):
    """Perform k-fold cross-validation"""
    
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=metric)
    
    # Print results
    print(f"\n{n_splits}-Fold Cross-Validation Results:")
    print(f"Mean {metric}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

def perform_stratified_cv(model, X, y, n_splits=5, metric='accuracy'):
    """Perform stratified k-fold cross-validation"""
    
    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
    
    # Print results
    print(f"\nStratified {n_splits}-Fold Cross-Validation Results:")
    print(f"Mean {metric}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores
```

## 3. Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

def perform_grid_search(model, param_grid, X, y, cv=5):
    """Perform grid search for hyperparameter tuning"""
    
    # Create grid search object
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Print results
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.3f}")
    
    return grid_search
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def perform_random_search(model, param_distributions, X, y, n_iter=100, cv=5):
    """Perform random search for hyperparameter tuning"""
    
    # Create random search object
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit random search
    random_search.fit(X, y)
    
    # Print results
    print("\nRandom Search Results:")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.3f}")
    
    return random_search
```

## 4. Learning Curves and Validation Curves

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves to diagnose bias-variance tradeoff"""
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    
    # Plot std bands
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

def plot_validation_curves(model, X, y, param_name, param_range, cv=5):
    """Plot validation curves for a specific parameter"""
    
    # Calculate validation curves
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name,
        param_range=param_range, cv=cv, n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label='Training score')
    plt.plot(param_range, val_mean, label='Cross-validation score')
    
    # Plot std bands
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title('Validation Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

## 5. Model Comparison

### Compare Multiple Models

```python
def compare_models(models, X, y, cv=5):
    """Compare multiple models using cross-validation"""
    
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results[name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"\n{name}:")
        print(f"Mean accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return results
```

## 6. Model Calibration

### Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def calibrate_model(model, X, y, cv=5):
    """Calibrate model probabilities"""
    
    # Create calibrated model
    calibrated_model = CalibratedClassifierCV(model, cv=cv)
    calibrated_model.fit(X, y)
    
    # Plot calibration curves
    plt.figure(figsize=(10, 6))
    
    # Original model
    prob_pos = model.fit(X, y).predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, prob_pos, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives,
             "s-", label='Original')
    
    # Calibrated model
    prob_pos_calibrated = calibrated_model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, prob_pos_calibrated, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives,
             "s-", label='Calibrated')
    
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves")
    plt.legend(loc="best")
    plt.show()
    
    return calibrated_model
```

## 7. Error Analysis

### Analyze Predictions

```python
def analyze_errors(y_true, y_pred, X, feature_names=None):
    """Analyze prediction errors"""
    
    # Find incorrect predictions
    incorrect_mask = y_true != y_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # Create error analysis DataFrame
    error_analysis = pd.DataFrame({
        'true_label': y_true[incorrect_mask],
        'predicted_label': y_pred[incorrect_mask],
        'index': incorrect_indices
    })
    
    # Add feature values if names are provided
    if feature_names is not None:
        for i, feature in enumerate(feature_names):
            error_analysis[feature] = X[incorrect_mask, i]
    
    return error_analysis
```

## 8. Practice Exercise

Create a complete model evaluation pipeline:

```python
def complete_evaluation_pipeline(model, X, y, param_grid=None):
    """Complete model evaluation pipeline"""
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Perform cross-validation
    cv_scores = perform_stratified_cv(model, X_train, y_train)
    
    # 3. Perform hyperparameter tuning if param_grid is provided
    if param_grid is not None:
        model = perform_grid_search(
            model, param_grid, X_train, y_train
        ).best_estimator_
    
    # 4. Train final model
    model.fit(X_train, y_train)
    
    # 5. Make predictions
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None
    
    # 6. Evaluate model
    evaluation = evaluate_classification(y_test, y_pred, y_prob)
    
    # 7. Plot learning curves
    plot_learning_curves(model, X, y)
    
    # 8. Analyze errors
    error_analysis = analyze_errors(y_test, y_pred, X_test)
    
    return model, evaluation, error_analysis
```

---

## Navigation

[⬅️ Previous: Feature Engineering](feature-engineering.md) | [Next: Git and GitHub ➡️](../00-Github/Git-Github.md)

[🔝 Back to Top](#model-evaluation-and-optimization)
