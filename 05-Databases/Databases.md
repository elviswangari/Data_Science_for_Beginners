# Introduction to Databases

## What is a Database?

- A database is a collection of data that is organized and stored in a way that can be easily accessed, managed, and updated.
- Think of it like a digital version of a file cabinet where information is categorized and stored in an organized manner.

### Types of Databases

1. **Relational Databases (SQL)**:  
   Data is stored in tables (rows and columns) and uses SQL to manage data.  
   Examples: MySQL, PostgreSQL.

2. **NoSQL Databases**:  
   Data is stored in a non-tabular format, such as documents, key-value pairs, or graphs.  
   Examples: MongoDB, Firebase.

---

# Introduction to Relational Databases and SQL

## What is a Relational Database?

- A relational database organizes data into tables, with rows and columns.
- Tables can be linked to each other using relationships, which help keep the data organized and connected.

## What is SQL?

- SQL stands for Structured Query Language. It is the standard language used to manage and manipulate relational databases.
- SQL allows you to create, read, update, and delete data, commonly referred to as CRUD operations.

### SQL Database Languages

Relational databases typically support four types of languages within SQL, each serving a specific function:

1. **Data Definition Language (DDL)**:
   - Used to define and modify the structure of the database (e.g., creating tables, altering them).
   - Examples:
     - `CREATE TABLE` – creates a new table.
     - `ALTER TABLE` – modifies an existing table.
     - `DROP TABLE` – deletes a table.

2. **Data Manipulation Language (DML)**:
   - Used to manipulate or modify data stored in the database (e.g., inserting, updating, deleting records).
   - Examples:
     - `INSERT INTO` – adds new data.
     - `UPDATE` – modifies existing data.
     - `DELETE` – removes data.
     - `SELECT` – retrieves data.

3. **Data Control Language (DCL)**:
   - Used to control access to data stored in the database, such as who can see or modify data.
   - Examples:
     - `GRANT` – gives specific permissions to users.
     - `REVOKE` – removes those permissions.

4. **Transaction Control Language (TCL)**:
   - Used to manage transactions, ensuring that all operations in a transaction are completed successfully.
   - Examples:
     - `COMMIT` – saves all changes made in the current transaction.
     - `ROLLBACK` – undoes changes made in the current transaction if an error occurs.

---

# Basic SQL Commands

### 1. **Create a Database**

- Create a new database named `school` and use it:

    ```sql
    CREATE DATABASE school;
    USE school;
    ```

### 2. **Create a Table (Create)**

- Create a `students` table with columns for `id`, `name`, `age`, and `email`:

    ```sql
    CREATE TABLE students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        age INT,
        email VARCHAR(100)
    );
    ```

### 3. **Insert Data (Create)**

- Insert sample records into the `students` table:

    ```sql
    INSERT INTO students (name, age, email) VALUES 
    ('John Doe', 20, 'john@example.com'),
    ('Jane Smith', 22, 'jane@example.com'),
    ('Alice Brown', 21, 'alice@example.com');
    ```

### 4. **Read Data (Read)**

- Retrieve all records from the `students` table:

    ```sql
    SELECT * FROM students;
    ```

### 5. **Update Data (Update)**

- Update John Doe’s age to 21:

    ```sql
    UPDATE students SET age = 21 WHERE name = 'John Doe';
    ```

### 6. **Delete Data (Delete)**

- Delete Jane Smith’s record:

    ```sql
    DELETE FROM students WHERE name = 'Jane Smith';
    ```

### 7. **Final Read to Confirm Changes (Read)**

- Retrieve all records again to confirm the updates:

    ```sql
    SELECT * FROM students;
    ```
