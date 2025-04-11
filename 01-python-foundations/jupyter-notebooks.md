# Jupyter Notebooks

[⬅️ Previous: Python Fundamentals](python-for-data-science.md) | [Next: NumPy Guide ➡️](../02-data-manipulation/numpy-guide.md)

# Jupyter Notebooks for Data Science

## Learning Objectives

By the end of this section, you will:

1. Understand what Jupyter Notebooks are and their importance in data science
2. Learn how to create and manage Jupyter Notebooks
3. Master key features like markdown cells, code cells, and magic commands
4. Learn best practices for organizing data science projects with Jupyter

## 1. Introduction to Jupyter Notebooks

### What are Jupyter Notebooks?

- Interactive computing environment that combines code, text, mathematics, plots, and media
- Perfect for data exploration and visualization
- Great for sharing analysis and results
- Supports multiple programming languages (Python, R, Julia, etc.)

### Getting Started

```python
# First, install Jupyter if you haven't already
# In terminal/command prompt:
# pip install jupyter

# Launch Jupyter Notebook:
# jupyter notebook
```

## 2. Working with Cells

### Code Cells

```python
# This is a code cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.randn(100)
plt.hist(data)
plt.title('Histogram of Random Data')
plt.show()
```

### Markdown Cells

```markdown
# Main Title
## Subtitle
### Section

- Bullet points
- Support *italic*
- And **bold** text

1. Numbered lists
2. Are also supported

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

## 3. Magic Commands

### Common Magic Commands

```python
# Line magics (start with %)
%time               # Time execution of a single line
%matplotlib inline  # Display plots in notebook
%pwd               # Show current directory

# Cell magics (start with %%)
%%time             # Time execution of entire cell
%%bash             # Run bash commands
%%html             # Render cell as HTML
```

## 4. Interactive Data Analysis

### Example: Loading and Exploring Data

```python
# Load a dataset
import pandas as pd
df = pd.read_csv('data.csv')

# Display first few rows
display(df.head())

# Basic statistics
df.describe()

# Interactive visualization with plotly
import plotly.express as px
fig = px.scatter(df, x='column1', y='column2')
fig.show()
```

## 5. Best Practices

### Notebook Organization

1. Start with import statements
2. Add markdown documentation
3. Keep code cells focused and small
4. Use clear section headers
5. Include output validation

### Example Structure

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Data
df = pd.read_csv('data.csv')

# 3. Data Preprocessing
# Clean missing values
df = df.dropna()

# 4. Analysis
# Calculate statistics
summary_stats = df.describe()

# 5. Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y')
plt.title('Data Visualization')
plt.show()
```

## 6. Advanced Features

### Widgets for Interactivity

```python
from ipywidgets import interact
import ipywidgets as widgets

@interact(x=widgets.IntSlider(min=1, max=100))
def plot_data(x):
    plt.figure(figsize=(10, 6))
    plt.plot(range(x))
    plt.title(f'Line plot with {x} points')
    plt.show()
```

### Debugging in Notebooks

```python
# Use %debug magic after an error occurs
# Or use this at the start of your notebook:
%pdb on

# Set breakpoints in your code
import pdb
pdb.set_trace()  # Code will pause here
```

## 7. Tips and Tricks

1. **Keyboard Shortcuts**
   - `Shift + Enter`: Run cell and move to next
   - `Ctrl + Enter`: Run cell and stay
   - `A`: Insert cell above
   - `B`: Insert cell below
   - `M`: Change to markdown
   - `Y`: Change to code

2. **Auto-reload External Modules**

```python
%load_ext autoreload
%autoreload 2
```

3. **Performance Tips**
   - Use `%%time` to measure cell execution time
   - Clear output of unused cells
   - Restart kernel when memory usage is high

## 8. Practice Exercise

Create a Jupyter notebook that:

1. Loads a dataset
2. Performs basic data cleaning
3. Creates visualizations
4. Includes markdown documentation
5. Uses interactive widgets

Example starter code:

```python
# Load libraries and data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Create interactive plot
@interact(column=df.columns)
def plot_distribution(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column)
    plt.title(f'Distribution of {column}')
    plt.show()
```

---

## Navigation

[⬅️ Previous: Python Fundamentals](python-for-data-science.md) | [Next: NumPy Guide ➡️](../02-data-manipulation/numpy-guide.md)

[🔝 Back to Top](#jupyter-notebooks)
