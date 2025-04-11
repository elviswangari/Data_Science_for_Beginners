# Feature Engineering for Machine Learning

[⬅️ Previous: ML Fundamentals](ml-fundamentals.md) | [Next: Model Evaluation ➡️](model-evaluation.md)

## Learning Objectives

By the end of this section, you will:

1. Understand the importance of feature engineering
2. Master various feature transformation techniques
3. Learn feature selection methods
4. Create new features from existing data
5. Handle different types of features effectively

## 1. Introduction to Feature Engineering

### Why Feature Engineering?

- Improve model performance
- Extract meaningful information from raw data
- Handle different data types appropriately
- Reduce dimensionality
- Create more informative features

## 2. Feature Scaling and Transformation

### Numerical Feature Scaling

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_features(X, method='standard'):
    """Scale numerical features using different methods"""
    
    if method == 'standard':
        # Standardization (Z-score normalization)
        scaler = StandardScaler()
    elif method == 'minmax':
        # Min-Max scaling
        scaler = MinMaxScaler()
    elif method == 'robust':
        # Robust scaling (using statistics that are robust to outliers)
        scaler = RobustScaler()
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Example usage
X = np.random.randn(100, 4)
X_scaled, scaler = scale_features(X, method='standard')
```

### Non-Linear Transformations

```python
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def transform_features(X, method='yeo-johnson'):
    """Apply non-linear transformations to features"""
    
    if method == 'yeo-johnson':
        # Yeo-Johnson transformation
        transformer = PowerTransformer(method='yeo-johnson')
    elif method == 'box-cox':
        # Box-Cox transformation (only for positive values)
        transformer = PowerTransformer(method='box-cox')
    elif method == 'quantile':
        # Transform to normal distribution using quantiles
        transformer = QuantileTransformer(output_distribution='normal')
    
    X_transformed = transformer.fit_transform(X)
    return X_transformed, transformer
```

## 3. Categorical Feature Engineering

### Encoding Techniques

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

def encode_categorical(X, method='onehot', target=None):
    """Encode categorical variables using different methods"""
    
    if method == 'label':
        # Label encoding
        encoder = LabelEncoder()
        X_encoded = encoder.fit_transform(X)
    
    elif method == 'onehot':
        # One-hot encoding
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)
    
    elif method == 'target':
        # Target encoding
        encoder = TargetEncoder()
        X_encoded = encoder.fit_transform(X, target)
    
    return X_encoded, encoder
```

### Handling High Cardinality

```python
def handle_high_cardinality(X, threshold=10):
    """Handle categorical features with high cardinality"""
    
    # Count frequency of each category
    value_counts = X.value_counts()
    
    # Create 'Other' category for less frequent values
    mask = value_counts < threshold
    mapping = {k: 'Other' if mask[k] else k for k in value_counts.index}
    
    # Apply mapping
    X_processed = X.map(mapping)
    
    return X_processed, mapping
```

## 4. Feature Creation

### Time-Based Features

```python
def create_datetime_features(df, date_column):
    """Create features from datetime column"""
    
    df = df.copy()
    
    # Extract basic components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['hour'] = df[date_column].dt.hour
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df
```

### Interaction Features

```python
def create_interactions(X, degree=2, include_bias=False):
    """Create interaction features between columns"""
    
    from sklearn.preprocessing import PolynomialFeatures
    
    # Create polynomial features (includes interactions)
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_interact = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out()
    
    return X_interact, feature_names
```

## 5. Feature Selection

### Statistical Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_features_statistical(X, y, method='f_test', k=10):
    """Select features using statistical tests"""
    
    if method == 'f_test':
        # F-test for classification
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        # Mutual information for classification
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support()
    
    return X_selected, selected_features
```

### Model-Based Selection

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def select_features_model_based(X, y, method='random_forest'):
    """Select features using model-based importance"""
    
    if method == 'random_forest':
        # Random Forest feature importance
        model = RandomForestClassifier(n_estimators=100)
        selector = SelectFromModel(model, prefit=False)
    
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support()
    
    return X_selected, selected_features, selector.estimator_.feature_importances_
```

## 6. Text Feature Engineering

### Text Preprocessing

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess_text(texts, method='tfidf'):
    """Preprocess text data and convert to features"""
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    
    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        # Remove stop words and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words]
        return ' '.join(tokens)
    
    # Preprocess all texts
    processed_texts = [preprocess(text) for text in texts]
    
    if method == 'tfidf':
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
    else:
        # Bag of words
        vectorizer = CountVectorizer()
    
    X_text = vectorizer.fit_transform(processed_texts)
    
    return X_text, vectorizer
```

## 7. Advanced Feature Engineering

### Automated Feature Engineering

```python
def automated_feature_engineering(df, target_col):
    """Automated feature engineering pipeline"""
    
    from featuretools import EntitySet, dfs
    
    # Create an entity set
    es = EntitySet(id='data')
    
    # Add the dataframe as an entity
    es.entity_from_dataframe(entity_id='data',
                            dataframe=df,
                            index='index',  # Assuming df has an index
                            target=target_col)
    
    # Run deep feature synthesis
    feature_matrix, feature_defs = dfs(entityset=es,
                                     target_entity='data',
                                     max_depth=2)
    
    return feature_matrix, feature_defs
```

## 8. Practice Exercise

Create a complete feature engineering pipeline:

```python
def complete_feature_pipeline(df, categorical_cols, numerical_cols, 
                            date_cols=None, text_cols=None):
    """Complete feature engineering pipeline"""
    
    df_processed = df.copy()
    
    # Handle numerical features
    for col in numerical_cols:
        df_processed[col], _ = scale_features(
            df_processed[col].values.reshape(-1, 1)
        )
    
    # Handle categorical features
    for col in categorical_cols:
        df_processed[col], _ = handle_high_cardinality(df_processed[col])
        df_processed[col], _ = encode_categorical(df_processed[col])
    
    # Handle date features
    if date_cols:
        for col in date_cols:
            df_processed = create_datetime_features(df_processed, col)
    
    # Handle text features
    if text_cols:
        for col in text_cols:
            text_features, _ = preprocess_text(df_processed[col])
            # Convert sparse matrix to dense and add as new features
            text_features = pd.DataFrame(
                text_features.toarray(),
                columns=[f'{col}_text_{i}' for i in range(text_features.shape[1])]
            )
            df_processed = pd.concat([df_processed, text_features], axis=1)
    
    return df_processed
```

---

## Navigation

[⬅️ Previous: ML Fundamentals](ml-fundamentals.md) | [Next: Model Evaluation ➡️](model-evaluation.md)

[🔝 Back to Top](#feature-engineering-for-machine-learning)
