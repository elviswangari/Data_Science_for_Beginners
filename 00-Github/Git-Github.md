# Introduction to Git and GitHub for Beginners

## Learning Objectives

By the end of this session, you will:

1. Understand what Git and GitHub are, and why they are important.
2. Learn basic Git commands to manage versions of code locally.
3. Clone an existing repository from GitHub.
4. Create a repository from a local project and push it to GitHub.
5. Learn how to push and pull changes to and from GitHub.
6. Perform basic tasks like initializing a repository, committing changes, and collaborating via GitHub.

---

## 1. Introduction to Git and GitHub

### What is Git?

- **Git** is a distributed version control system that tracks changes in files over time.
- It helps you "save checkpoints" in your project, allowing you to revert to earlier versions.
- Git is commonly used in software development to manage and collaborate on code with multiple developers.
  
### What is GitHub?

- **GitHub** is a cloud-based platform that hosts Git repositories.
- It enables developers to store, manage, and collaborate on Git projects online.
- Think of GitHub as a social media platform for code where people can contribute, share, and collaborate on projects.

### Why Use Git and GitHub?

- **Version control**: Track and manage changes to your project over time.
- **Collaboration**: Work efficiently with team members on the same project without conflicts.
- **Backup**: Safely store your projects in the cloud and access them from anywhere.

---

## 2. Prerequisites

Before starting, make sure you have the following:

1. **Git Installed**: You need Git installed on your machine to follow along. [Download Git here](https://git-scm.com/downloads).
2. **GitHub Account**: Create a GitHub account if you don't already have one. You can do so at [github.com](https://github.com/join).
3. **Basic Git Configuration**: After installing Git, you’ll need to set up your Git username and email, which will be used to track changes.
   - Open Git Bash (Windows) or Terminal (Mac/Linux) and run the following commands:

     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "your.email@example.com"
     ```

   - This ensures that your commits are associated with your identity.

---

## 3. Two Ways to Start a Git Project

### (a) Cloning an Online Repository  

- You can clone an existing repository from GitHub to work on it locally.

### (b) Creating a Local Repository and Pushing it to GitHub  

- You can create a Git repository locally and later push it to GitHub for others to access.

---

## 4. Personal Access Tokens (PAT) in GitHub

### What is a Personal Access Token (PAT)?

- A **Personal Access Token (PAT)** is a secure method of authentication used to interact with GitHub's API.
- It replaces passwords for Git operations like cloning, pushing, or accessing repositories.
- PATs offer more security than traditional username/password logins.

### Why Use Personal Access Tokens?

- **Enhanced Security**: Password-based access for Git operations is deprecated. PATs are a more secure alternative.
- **Fine-Grained Control**: PATs allow you to control specific permissions (e.g., repo access, workflow control).
- **Easy Revocation**: Tokens can be revoked or regenerated without affecting your account security.

### How to Generate a Personal Access Token

1. **Go to GitHub Settings**:
   - Click your profile picture in the top-right corner and select **Settings**.

2. **Access Developer Settings**:
   - Scroll down the sidebar to **Developer settings**.

3. **Generate New Token**:
   - Under **Personal access tokens**, click **Tokens (classic)** or **Fine-grained tokens** (for better security).
   - Click **Generate new token**.

4. **Set Permissions**:
   - Choose the scopes for the token (e.g., `repo` for repository access).
   - Limit the token's scope to specific repositories as needed.

5. **Save the Token**:
   - Copy the token displayed. You will **not** be able to see it again.
   - Store it securely in a password manager.

---

## 5. Practical Part 1: Cloning an Existing Repository

### Scenario: Clone an existing project from GitHub and make some changes

1. **Create a Repository on GitHub**:
    - Go to GitHub, click **New Repository**, name it `my-cloned-repo`, and click **Create**.

2. **Copy the Repository URL**:
    - On the repository page, click the green **Code** button and copy the URL.

3. **Clone the Repository to Your Computer**:

    ```bash
    git clone https://<PAT>@github.com/username/my-cloned-repo.git
    ```

    - Replace `<PAT>` with your Personal Access Token, and `username/my-cloned-repo` with your GitHub username and repo name.
  
4. **Navigate to the Cloned Repository**:

    ```bash
    cd my-cloned-repo
    ```

5. **Create a New File (e.g., `hello.txt`)**:

    ```bash
    echo "Hello from the cloned repo!" > hello.txt
    ```

6. **Stage and Commit the File**:
    - Stage the file:

      ```bash
      git add hello.txt
      ```

    - Commit the changes:

      ```bash
      git commit -m "Add hello.txt"
      ```

7. **Push the Changes to GitHub**:

    ```bash
    git push origin main
    ```

8. **Check GitHub**: The new file will appear in your repository online.

---

## 6. Practical Part 2: Creating a Local Repository with `git init`

### Scenario: Create a new Git repository locally and push it to GitHub

1. **Create a New Folder on Your Computer**:

    ```bash
    mkdir git_project
    cd git_project
    ```

2. **Initialize Git in the Folder**:

    ```bash
    git init
    ```

    - This tells Git to start tracking changes in this folder.

3. **Create a New File (e.g., `readme.txt`)**:

    ```bash
    echo "This is a local repo." > readme.txt
    ```

4. **Stage and Commit the File**:
    - Stage the file:

      ```bash
      git add readme.txt
      ```

    - Commit the changes:

      ```bash
      git commit -m "Add readme.txt"
      ```

5. **Create a New Repository on GitHub**:
    - Go to GitHub, click **New Repository**, name it `github_project`, and click **Create**.

6. **Connect Your Local Repo to GitHub**:

    ```bash
    git remote add origin https://<PAT>@github.com/username/github_project.git
    ```

7. **Push Your Changes to GitHub**:

    ```bash
    git push -u origin main
    ```

8. **Check GitHub**: Your local files will now be uploaded to your GitHub repository.

---

## 7. Practical Recap and Q&A

- **Recap**: We’ve covered how to clone a repository, make changes, and push them to GitHub, as well as how to create a repository locally and link it to GitHub.
- **Q&A**: Open session for any questions or clarification.
