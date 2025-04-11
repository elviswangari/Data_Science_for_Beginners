# Machine Learning Fundamentals

[⬅️ Previous: Advanced Visualization](../03-data-visualization/advanced-visualization.md) | [Next: Feature Engineering ➡️](feature-engineering.md)

## Learning Objectives

By the end of this section, you will:

1. Understand core machine learning concepts
2. Learn different types of machine learning algorithms
3. Master the basic machine learning workflow
4. Implement simple ML models using scikit-learn
5. Evaluate model performance effectively

## 1. Introduction to Machine Learning

### Types of Machine Learning

- Supervised Learning
  - Classification
  - Regression
- Unsupervised Learning
  - Clustering
  - Dimensionality Reduction
- Reinforcement Learning

### Basic Terminology

- Features (X): Input variables
- Target (y): Output variable
- Training Data: Data used to train the model
- Test Data: Data used to evaluate the model
- Model: Algorithm that learns patterns from data
- Predictions: Model's output for new data

## 2. The Machine Learning Workflow

### Basic ML Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def basic_ml_pipeline(X, y):
    """Basic machine learning pipeline"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler
```

## 3. Supervised Learning

### Classification

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_classes=2, random_state=42)

def compare_classifiers(X, y):
    """Compare different classification algorithms"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    # Train and evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
        
    return results
```

### Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def regression_example():
    """Example of regression analysis"""
    
    # Generate sample regression data
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    
    return model
```

## 4. Unsupervised Learning

### Clustering

```python
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def clustering_example(X, n_clusters=3):
    """Example of clustering analysis"""
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    
    return {
        'kmeans': kmeans_labels,
        'gmm': gmm_labels
    }
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_dimensions(X, n_components=2):
    """Reduce dimensionality of data"""
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    return {
        'pca': X_pca,
        'tsne': X_tsne,
        'pca_explained_variance': pca.explained_variance_ratio_
    }
```

## 5. Model Evaluation

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def perform_cross_validation(model, X, y, cv=5):
    """Perform k-fold cross-validation"""
    
    # Create cross-validation object
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=kf)
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return scores
```

### Model Selection

```python
from sklearn.model_selection import GridSearchCV

def perform_grid_search(model, param_grid, X, y):
    """Perform grid search for hyperparameter tuning"""
    
    # Create grid search object
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    return grid_search.best_estimator_
```

## 6. Feature Engineering

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def select_features(X, y, k=5):
    """Select k best features"""
    
    # Create feature selector
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_features = selector.get_support()
    
    return X_selected, selected_features
```

### Feature Creation

```python
def create_polynomial_features(X, degree=2):
    """Create polynomial features"""
    
    from sklearn.preprocessing import PolynomialFeatures
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    return X_poly, poly.get_feature_names_out()
```

## 7. Best Practices

### Data Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline for mixed data types"""
    
    # Numeric preprocessing
    numeric_transformer = StandardScaler()
    
    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(drop='first')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
```

## 8. Practice Exercise

Create a complete machine learning pipeline:

```python
def complete_ml_pipeline(X, y, numeric_features, categorical_features):
    """Create and evaluate complete ML pipeline"""
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numeric_features, categorical_features
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline
```

---

## Navigation

[⬅️ Previous: Advanced Visualization](../03-data-visualization/advanced-visualization.md) | [Next: Feature Engineering ➡️](feature-engineering.md)

[🔝 Back to Top](#machine-learning-fundamentals)
