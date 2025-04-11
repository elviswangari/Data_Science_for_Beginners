# Data Visualization with Matplotlib and Seaborn

[⬅️ Previous: Data Cleaning](../02-data-manipulation/data-cleaning-techniques.md) | [Next: Advanced Visualization ➡️](advanced-visualization.md)

## Learning Objectives

By the end of this section, you will:

1. Understand basic plotting with Matplotlib
2. Master statistical visualization with Seaborn
3. Create publication-quality visualizations
4. Learn to customize and style plots effectively

## 1. Introduction to Matplotlib

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Create basic line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('Basic Line Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()
```

### Multiple Plots

```python
# Create subplot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
ax1.plot(x, np.sin(x), 'b-', label='sin(x)')
ax1.set_title('Sine Wave')
ax1.legend()

# Second subplot
ax2.plot(x, np.cos(x), 'r-', label='cos(x)')
ax2.set_title('Cosine Wave')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 2. Common Plot Types with Matplotlib

### Scatter Plots

```python
# Generate random data
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0, 1, 100)

plt.figure(figsize=(8, 6))
plt.scatter(data1, data2, alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Bar Charts

```python
categories = ['A', 'B', 'C', 'D']
values = [4, 3, 2, 5]

plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

### Histograms

```python
data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, alpha=0.7)
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
```

## 3. Introduction to Seaborn

### Basic Seaborn Plots

```python
import seaborn as sns
import pandas as pd

# Create sample dataset
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Basic scatter plot
sns.scatterplot(data=data, x='x', y='y', hue='category')
plt.title('Seaborn Scatter Plot')
plt.show()
```

### Statistical Plots

```python
# Box plot
sns.boxplot(data=data, x='category', y='y')
plt.title('Box Plot')
plt.show()

# Violin plot
sns.violinplot(data=data, x='category', y='y')
plt.title('Violin Plot')
plt.show()
```

## 4. Advanced Seaborn Features

### Distribution Plots

```python
# Single variable distribution
sns.displot(data=data, x='x', kind='kde')
plt.title('KDE Plot')
plt.show()

# Joint distribution
sns.jointplot(data=data, x='x', y='y', kind='hex')
plt.show()
```

### Regression Plots

```python
# Simple regression plot
sns.regplot(data=data, x='x', y='y')
plt.title('Regression Plot')
plt.show()

# Complex regression
sns.lmplot(data=data, x='x', y='y', hue='category')
plt.title('Regression by Category')
plt.show()
```

## 5. Customization and Styling

### Matplotlib Customization

```python
# Set style parameters
plt.style.use('seaborn')  # Use seaborn style

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
ax.fill_between(x, np.sin(x), alpha=0.3)

# Customize
ax.set_title('Customized Plot', fontsize=14, pad=20)
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()
```

### Seaborn Themes and Palettes

```python
# Set theme
sns.set_theme(style="whitegrid", palette="husl")

# Create plot with custom palette
sns.scatterplot(data=data, x='x', y='y', hue='category', 
                palette=sns.color_palette("Set2"))
plt.title('Styled Seaborn Plot')
plt.show()
```

## 6. Multiple Plot Types

### Pair Plots

```python
# Create more complex dataset
df = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100),
    'category': np.random.choice(['A', 'B'], 100)
})

# Create pair plot
sns.pairplot(df, hue='category')
plt.show()
```

### Heat Maps

```python
# Create correlation matrix
correlation = df.corr()

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## 7. Best Practices

### Plot Organization

```python
# Create organized multi-plot figure
fig = plt.figure(figsize=(15, 10))

# Define grid
gs = fig.add_gridspec(2, 3)

# Add subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])
ax3 = fig.add_subplot(gs[1, :])

# Plot in each subplot
ax1.scatter(data['x'], data['y'])
sns.kdeplot(data=data, x='x', ax=ax2)
sns.boxplot(data=data, x='category', y='y', ax=ax3)

# Adjust layout
plt.tight_layout()
plt.show()
```

## 8. Practice Exercise

Create a comprehensive visualization dashboard:

```python
def create_dashboard(df):
    """Create a dashboard with multiple plots"""
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x='x1', y='x2', hue='category', ax=ax1)
    ax1.set_title('Scatter Plot')
    
    # Distribution plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data=df, x='x1', hue='category', ax=ax2)
    ax2.set_title('Distribution Plot')
    
    # Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=df, x='category', y='x1', ax=ax3)
    ax3.set_title('Box Plot')
    
    # Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()
```

---

## Navigation

[⬅️ Previous: Data Cleaning](../02-data-manipulation/data-cleaning-techniques.md) | [Next: Advanced Visualization ➡️](advanced-visualization.md)

[🔝 Back to Top](#data-visualization-with-matplotlib-and-seaborn)
