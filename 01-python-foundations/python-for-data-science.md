# Python Fundamentals

[⬅️ Back to Table of Contents](../README.md) | [Next: Jupyter Notebooks ➡️](jupyter-notebooks.md) | [↩️ Back](../README.md)

## Learning Objectives

By the end of this section, you will:

1. Understand Python basics and core concepts
2. Learn essential Python data structures
3. Master control flow and functions
4. Handle files and errors effectively
5. Write clean and maintainable code

## 1. Variables and Data Types

### Basic Data Types

```python
# Numbers
integer_num = 42
float_num = 3.14

# Strings
text = "Hello, Python!"
multiline_text = """This is a
multiline string"""

# Booleans
is_active = True
is_complete = False

# None type
empty_value = None
```

### Type Conversion

```python
# Converting between types
str_num = "42"
num = int(str_num)    # String to integer
float_num = float(num)  # Integer to float
text = str(num)       # Number to string
```

## 2. Data Structures

### Lists

```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]

# List operations
fruits.append("grape")     # Add element
fruits.remove("banana")    # Remove element
first_fruit = fruits[0]   # Access element
last_fruit = fruits[-1]   # Access last element
sub_list = fruits[1:3]    # Slicing

# List methods
fruits.sort()             # Sort list
fruits.reverse()          # Reverse list
fruit_count = len(fruits) # List length
```

### Dictionaries

```python
# Creating dictionaries
person = {
    "name": "John Doe",
    "age": 30,
    "city": "New York"
}

# Dictionary operations
person["email"] = "john@example.com"  # Add key-value
del person["age"]                     # Remove key-value
name = person.get("name", "Unknown")  # Safe access

# Dictionary methods
keys = person.keys()      # Get all keys
values = person.values()  # Get all values
items = person.items()    # Get key-value pairs
```

### Tuples and Sets

```python
# Tuples (immutable)
coordinates = (10, 20)
x, y = coordinates    # Tuple unpacking

# Sets (unique elements)
numbers = {1, 2, 3, 3, 4}  # Duplicates removed
numbers.add(5)             # Add element
numbers.remove(2)          # Remove element
```

## 3. Control Flow

### Conditional Statements

```python
# If statements
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"
```

### Loops

```python
# For loops
for i in range(5):
    print(i)

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
for num in range(10):
    if num == 5:
        continue    # Skip rest of iteration
    if num == 8:
        break       # Exit loop entirely
```

## 4. Functions

### Function Basics

```python
def greet(name, greeting="Hello"):
    """
    A simple greeting function.
    Args:
        name: Person's name
        greeting: Type of greeting (default: Hello)
    Returns:
        Formatted greeting string
    """
    return f"{greeting}, {name}!"

# Function calls
message = greet("Alice")
custom_message = greet("Bob", "Hi")
```

### Lambda Functions

```python
# Anonymous functions
square = lambda x: x ** 2
double = lambda x: x * 2

# Using with built-in functions
numbers = [1, 2, 3, 4, 5]
squares = list(map(square, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

## 5. File Handling

### Reading and Writing Files

```python
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, Python!\n")
    file.write("This is a new line.")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()        # Read entire file
    
with open("example.txt", "r") as file:
    lines = file.readlines()     # Read lines into list

# Appending to a file
with open("example.txt", "a") as file:
    file.write("\nAppended line")
```

## 6. Error Handling

### Try-Except Blocks

```python
def safe_operation():
    try:
        # Risky operations
        number = int(input("Enter a number: "))
        result = 10 / number
        return result
    except ValueError:
        return "Please enter a valid number"
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    finally:
        print("Operation completed")
```

### Custom Exceptions

```python
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_age(age):
    if age < 0:
        raise ValidationError("Age cannot be negative")
    if age > 150:
        raise ValidationError("Invalid age value")
    return True
```

## 7. Modules and Packages

### Importing

```python
# Basic imports
import math
from datetime import datetime
from random import randint, choice

# Using imports
pi = math.pi
current_time = datetime.now()
random_number = randint(1, 100)
```

### Creating Modules

```python
# utils.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# main.py
from utils import add, subtract
result = add(10, 5)
```

## 8. Best Practices

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings and comments
- Keep functions small and focused
- Use consistent indentation

### Code Organization

- Group related functionality
- Use modules to organize code
- Keep files focused and manageable
- Use proper package structure

## Practice Exercises

1. Create a phone book program using dictionaries
2. Implement a file-based todo list
3. Build a simple calculator with error handling

Example:

```python
def create_phone_book():
    phone_book = {}
    
    while True:
        name = input("Enter name (or 'quit' to exit): ")
        if name.lower() == 'quit':
            break
            
        try:
            number = input("Enter phone number: ")
            if not number.isdigit():
                raise ValueError("Phone number must contain only digits")
            phone_book[name] = number
        except ValueError as e:
            print(f"Error: {e}")
    
    return phone_book
```

---

## Navigation

[⬅️ Back to Table of Contents](../README.md) | [Next: Jupyter Notebooks ➡️](jupyter-notebooks.md) | [↩️ Back](../README.md)

[🔝 Back to Top](#python-fundamentals)
