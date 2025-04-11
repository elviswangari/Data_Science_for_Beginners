# Data Cleaning Techniques

[⬅️ Previous: Pandas Guide](pandas-guide.md) | [Next: Matplotlib and Seaborn ➡️](../03-data-visualization/matplotlib-seaborn.md)

## Learning Objectives

By the end of this section, you will:

1. Understand common data quality issues
2. Master essential data cleaning techniques
3. Learn how to handle missing data effectively
4. Identify and handle outliers
5. Validate and verify cleaned data

## 1. Understanding Data Quality

### Common Data Quality Issues

- Missing values
- Duplicate records
- Inconsistent formats
- Outliers
- Invalid data
- Typos and errors
- Structural errors

## 2. Data Assessment

### Initial Data Exploration

```python
import pandas as pd
import numpy as np

def assess_data_quality(df):
    """Perform initial data quality assessment"""
    
    # Basic information
    print("\nBasic Information:")
    print(df.info())
    
    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Duplicate rows
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    # Basic statistics
    print("\nNumerical Column Statistics:")
    print(df.describe())
    
    # Value counts for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())
```

## 3. Handling Missing Data

### Detection and Analysis

```python
def analyze_missing_data(df):
    """Analyze patterns in missing data"""
    
    # Calculate percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Missing value patterns
    missing_pattern = pd.DataFrame({
        'Column': df.columns,
        'Percent_Missing': missing_percentage,
        'Data_Type': df.dtypes
    })
    
    return missing_pattern.sort_values('Percent_Missing', ascending=False)
```

### Missing Data Strategies

```python
def handle_missing_values(df, strategy='advanced'):
    """Handle missing values using different strategies"""
    
    if strategy == 'simple':
        # Simple imputation
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    elif strategy == 'advanced':
        # Advanced imputation
        from sklearn.impute import SimpleImputer, KNNImputer
        
        # KNN imputation for numeric data
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical data
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    return df
```

## 4. Handling Duplicates

### Identifying Duplicates

```python
def identify_duplicates(df):
    """Identify and analyze duplicate records"""
    
    # Find duplicate rows
    duplicates = df[df.duplicated(keep=False)]
    
    # Group duplicates
    if len(duplicates) > 0:
        grouped = duplicates.groupby(list(duplicates.columns)).size().reset_index(name='count')
        return grouped.sort_values('count', ascending=False)
    
    return pd.DataFrame()
```

### Removing Duplicates

```python
def remove_duplicates(df, subset=None):
    """Remove duplicate records with options"""
    
    # Keep first occurrence and mark duplicates
    df['is_duplicate'] = df.duplicated(subset=subset, keep='first')
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    
    # Report cleaning results
    n_duplicates = df['is_duplicate'].sum()
    print(f"Removed {n_duplicates} duplicate rows")
    
    return df_cleaned.drop('is_duplicate', axis=1)
```

## 5. Handling Outliers

### Detecting Outliers

```python
def detect_outliers(df, columns, method='zscore'):
    """Detect outliers using different methods"""
    
    outliers = {}
    
    if method == 'zscore':
        # Z-score method
        for column in columns:
            zscore = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers[column] = df[zscore > 3]
    
    elif method == 'iqr':
        # IQR method
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers[column] = df[(df[column] < (Q1 - 1.5 * IQR)) | 
                                (df[column] > (Q3 + 1.5 * IQR))]
    
    return outliers
```

### Handling Outliers

```python
def handle_outliers(df, columns, method='clip'):
    """Handle outliers using different methods"""
    
    if method == 'clip':
        # Clip values at 3 standard deviations
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = df[column].clip(mean - 3*std, mean + 3*std)
    
    elif method == 'winsorize':
        # Winsorize at 5th and 95th percentiles
        for column in columns:
            low = df[column].quantile(0.05)
            high = df[column].quantile(0.95)
            df[column] = df[column].clip(low, high)
    
    return df
```

## 6. Data Validation

### Data Type Validation

```python
def validate_data_types(df, type_dict):
    """Validate and convert data types"""
    
    errors = []
    
    for column, dtype in type_dict.items():
        try:
            df[column] = df[column].astype(dtype)
        except Exception as e:
            errors.append(f"Error converting {column} to {dtype}: {str(e)}")
    
    return df, errors
```

### Value Range Validation

```python
def validate_value_ranges(df, range_dict):
    """Validate values are within specified ranges"""
    
    violations = {}
    
    for column, (min_val, max_val) in range_dict.items():
        invalid = df[(df[column] < min_val) | (df[column] > max_val)]
        if len(invalid) > 0:
            violations[column] = invalid
    
    return violations
```

## 7. Complete Data Cleaning Pipeline

```python
def complete_cleaning_pipeline(df):
    """Execute complete data cleaning pipeline"""
    
    print("1. Initial Assessment")
    assess_data_quality(df)
    
    print("\n2. Handling Missing Values")
    df = handle_missing_values(df, strategy='advanced')
    
    print("\n3. Removing Duplicates")
    df = remove_duplicates(df)
    
    print("\n4. Handling Outliers")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = handle_outliers(df, numeric_cols, method='clip')
    
    print("\n5. Validating Data Types")
    type_dict = {col: 'category' for col in df.select_dtypes(include=['object']).columns}
    df, type_errors = validate_data_types(df, type_dict)
    
    return df
```

## 8. Practice Exercise

Create a data cleaning function for a specific use case:

```python
def clean_customer_data(df):
    """Clean customer dataset"""
    
    # 1. Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # 2. Clean email addresses
    df['email'] = df['email'].str.lower().str.strip()
    
    # 3. Format phone numbers
    df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)
    
    # 4. Convert dates
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    
    # 5. Handle missing values
    df = handle_missing_values(df, strategy='simple')
    
    # 6. Remove duplicates
    df = remove_duplicates(df, subset=['email', 'phone'])
    
    return df
```

---

## Navigation

[⬅️ Previous: Pandas Guide](pandas-guide.md) | [Next: Matplotlib and Seaborn ➡️](../03-data-visualization/matplotlib-seaborn.md)

[🔝 Back to Top](#data-cleaning-techniques)
