# NumPy for Data Science

[⬅️ Previous: Jupyter Notebooks](../01-python-foundations/jupyter-notebooks.md) | [Next: Pandas Guide ➡️](pandas-guide.md)

## Learning Objectives

By the end of this section, you will:

1. Understand NumPy arrays and their advantages
2. Master array operations and broadcasting
3. Learn array manipulation techniques
4. Perform numerical computations efficiently

## 1. Introduction to NumPy

### Why NumPy?

- Foundation for scientific computing in Python
- Efficient array operations
- Memory-efficient storage
- Powerful broadcasting capabilities
- Integration with other data science libraries

## 2. NumPy Arrays

### Creating Arrays

```python
import numpy as np

# From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 4))        # Array of zeros
ones = np.ones((2, 3))         # Array of ones
identity = np.eye(3)           # Identity matrix
random = np.random.rand(3, 3)  # Random values
```

### Array Properties

```python
# Shape
print(arr2.shape)      # (2, 3)

# Dimensions
print(arr2.ndim)       # 2

# Data type
print(arr2.dtype)      # int64

# Size
print(arr2.size)       # 6
```

## 3. Array Operations

### Basic Operations

```python
# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)          # Addition
print(a * b)          # Multiplication
print(a ** 2)         # Exponentiation
print(np.sqrt(a))     # Square root
```

### Broadcasting

```python
# Arrays of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([1, 0, 1])

# Broadcasting
result = matrix * vector  # Multiplies each row by vector
```

## 4. Array Manipulation

### Reshaping and Transposing

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape
matrix = arr.reshape(2, 3)
print(matrix)

# Transpose
transposed = matrix.T
print(transposed)
```

### Stacking and Splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack
vertical = np.vstack((a, b))

# Horizontal stack
horizontal = np.hstack((a, b))

# Splitting
arr = np.array([1, 2, 3, 4, 5, 6])
split = np.split(arr, 3)  # Split into 3 equal parts
```

## 5. Advanced Operations

### Linear Algebra

```python
# Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

dot_product = np.dot(a, b)
matrix_product = a @ b  # Python 3.5+

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(a)
```

### Statistical Operations

```python
data = np.array([1, 2, 3, 4, 5])

mean = np.mean(data)
median = np.median(data)
std = np.std(data)
var = np.var(data)
```

## 6. Practical Examples

### Data Processing

```python
# Generate sample data
data = np.random.randn(1000)

# Process data
processed = np.where(data > 0, 1, 0)  # Binary classification
masked = np.ma.masked_where(data < 0, data)  # Mask negative values
```

### Image Processing

```python
# Create simple image (grayscale)
image = np.random.rand(5, 5)

# Image operations
rotated = np.rot90(image)        # Rotate 90 degrees
flipped = np.flip(image, axis=0) # Flip vertically
```

## 7. Performance Tips

### Vectorization

```python
# Slow loop-based approach
def slow_sum(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Fast vectorized approach
def fast_sum(arr):
    return np.sum(arr)
```

### Memory Efficiency

```python
# Create view instead of copy
a = np.array([1, 2, 3])
b = a.view()  # Creates a view
c = a.copy()  # Creates a copy
```

## 8. Practice Exercises

1. Create a function to generate and manipulate matrices

```python
def matrix_operations():
    # Create 3x3 matrix
    matrix = np.random.rand(3, 3)
    
    # Calculate:
    print("Determinant:", np.linalg.det(matrix))
    print("Inverse:", np.linalg.inv(matrix))
    print("Trace:", np.trace(matrix))
    
    return matrix
```

2. Implement data transformation pipeline

```python
def transform_data(data):
    # Normalize data
    normalized = (data - np.mean(data)) / np.std(data)
    
    # Remove outliers (values > 3 std)
    cleaned = normalized[np.abs(normalized) <= 3]
    
    return cleaned
```

3. Create image filters

```python
def apply_filters(image):
    # Apply various filters
    blurred = np.convolve(image, np.ones(3)/3, mode='same')
    edges = np.gradient(image)
    
    return blurred, edges
```

---

## Navigation

[⬅️ Previous: Jupyter Notebooks](../01-python-foundations/jupyter-notebooks.md) | [Next: Pandas Guide ➡️](pandas-guide.md)

[🔝 Back to Top](#numpy-for-data-science)
