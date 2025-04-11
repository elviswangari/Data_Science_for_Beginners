# Pandas for Data Science

[⬅️ Previous: NumPy Guide](numpy-guide.md) | [Next: Data Cleaning ➡️](data-cleaning-techniques.md)

## Learning Objectives

By the end of this section, you will:

1. Understand Pandas DataFrame and Series objects
2. Master data loading and manipulation techniques
3. Learn data cleaning and preprocessing methods
4. Perform advanced data analysis operations

## 1. Introduction to Pandas

### Why Pandas?

- Efficient data manipulation and analysis
- Built-in data alignment
- Integrated handling of missing data
- Powerful group by and merge operations
- Excel-like functionality in Python

## 2. Pandas Data Structures

### Series

```python
import pandas as pd

# Create Series
s = pd.Series([1, 2, 3, 4, 5], name='numbers')
s_with_index = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# Accessing elements
print(s[0])           # Access by position
print(s_with_index['a'])  # Access by label
```

### DataFrame

```python
# Create DataFrame
data = {
    'name': ['John', 'Anna', 'Peter'],
    'age': [28, 22, 35],
    'city': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)

# From CSV
df_csv = pd.read_csv('data.csv')

# From Excel
df_excel = pd.read_excel('data.xlsx')
```

## 3. Data Manipulation

### Basic Operations

```python
# View data
print(df.head())        # First 5 rows
print(df.tail())        # Last 5 rows
print(df.info())        # DataFrame info
print(df.describe())    # Statistical summary

# Selecting data
names = df['name']              # Select column
subset = df[['name', 'age']]   # Select multiple columns
row = df.loc[0]                # Select row by label
row_iloc = df.iloc[0]          # Select row by position
```

### Filtering and Sorting

```python
# Filter data
young = df[df['age'] < 30]
in_ny = df[df['city'] == 'New York']

# Multiple conditions
young_in_ny = df[(df['age'] < 30) & (df['city'] == 'New York')]

# Sort data
sorted_by_age = df.sort_values('age')
sorted_multiple = df.sort_values(['age', 'name'])
```

## 4. Data Cleaning

### Handling Missing Data

```python
# Check for missing values
print(df.isnull().sum())

# Drop missing values
df_cleaned = df.dropna()

# Fill missing values
df_filled = df.fillna(0)                    # Fill with zero
df_filled = df.fillna(method='ffill')       # Forward fill
df_filled = df.fillna(df.mean())            # Fill with mean
```

### Data Type Conversion

```python
# Convert types
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])

# Convert categorical data
df['category'] = df['category'].astype('category')
```

## 5. Data Analysis

### Grouping and Aggregation

```python
# Group by single column
by_city = df.groupby('city')
city_stats = by_city.agg({
    'age': ['mean', 'min', 'max'],
    'name': 'count'
})

# Multiple group by
by_city_gender = df.groupby(['city', 'gender'])
stats = by_city_gender.agg('mean')
```

### Merging and Joining

```python
# Example DataFrames
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [1, 2, 4], 'score': [90, 85, 95]})

# Merge
merged = pd.merge(df1, df2, on='id')        # Inner merge
merged_left = pd.merge(df1, df2, on='id', how='left')  # Left merge

# Concatenate
combined = pd.concat([df1, df2])            # Vertical concatenation
combined_horizontal = pd.concat([df1, df2], axis=1)  # Horizontal
```

## 6. Advanced Features

### Time Series

```python
# Create date range
dates = pd.date_range('2023-01-01', periods=5)

# Time series operations
ts = pd.Series(range(5), index=dates)
print(ts.resample('M').mean())    # Monthly resampling
print(ts.shift(1))                # Shift data
```

### Pivot Tables

```python
# Create pivot table
pivot = df.pivot_table(
    values='sales',
    index='date',
    columns='product',
    aggfunc='sum'
)

# Multi-level pivot
multi_pivot = df.pivot_table(
    values='sales',
    index=['date', 'region'],
    columns='product'
)
```

## 7. Data Export

### Saving Data

```python
# Save to CSV
df.to_csv('output.csv', index=False)

# Save to Excel
df.to_excel('output.xlsx', sheet_name='Sheet1')

# Save to JSON
df.to_json('output.json')
```

## 8. Practice Exercises

1. Data Cleaning Pipeline

```python
def clean_dataset(df):
    """Clean a raw dataset."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        'numeric_col': df['numeric_col'].mean(),
        'categorical_col': 'unknown'
    })
    
    # Convert data types
    df['date_col'] = pd.to_datetime(df['date_col'])
    
    return df
```

2. Analysis Function

```python
def analyze_sales(df):
    """Analyze sales data."""
    # Group by product and date
    sales_analysis = df.groupby(['product', pd.Grouper(key='date', freq='M')])
    
    # Calculate statistics
    stats = sales_analysis.agg({
        'quantity': ['sum', 'mean', 'std'],
        'revenue': ['sum', 'mean']
    })
    
    return stats
```

3. Data Transformation

```python
def transform_data(df):
    """Transform data for analysis."""
    # Create new features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate rolling averages
    df['rolling_avg'] = df['value'].rolling(window=7).mean()
    
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df
```

---

## Navigation

[⬅️ Previous: NumPy Guide](numpy-guide.md) | [Next: Data Cleaning ➡️](data-cleaning-techniques.md)

[🔝 Back to Top](#pandas-for-data-science)
