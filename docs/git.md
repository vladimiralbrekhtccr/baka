# How to reassign gitcloned repo to your private repo

To properly connect your existing local folder (which is already a git repository) to your new GitHub repository while preserving the history and clearly highlighting your changes, follow these steps.

We will effectively rename the original source to `upstream` (so you can still get updates later if needed) and set your new repo as `origin`.

### Step 1: Navigate to your directory
Open your terminal and go to the folder:
```bash
cd transformers
```

### Step 2: Commit your current changes locally
To "clearly see what you changed," you need to bundle your modifications into a Git commit. This creates a distinct point in history.

1. Check which files are changed:
```bash
git status
```
2. Add the files you changed (or all modified files):
```bash
git add .
```
3. Commit them with a descriptive message:
```bash
git commit -m "Applied custom changes for Qwen3 VL MoE"
```

### Step 3: Manage the Remotes
Currently, your `origin` points to the repo you originally cloned (likely Hugging Face's transformers). We want to point it to your new repo.

1. **Rename the old origin to 'upstream':**
   This keeps a link to the original code in case you want to pull updates later, but frees up the name `origin` for your new repo.
```bash
git remote rename origin upstream
```

2. **Add your new repository as 'origin':**
```bash
git remote add origin <your_online_created_repo>
```

3. **Verify the connections:**
```bash
git remote -v
```
   *It should look like:*
   *   `origin`: points to your `vladimiralbrekhtccr` repo.
   *   `upstream`: points to the original transformers repo.

### Step 4: Push to your new repository
Now, push the entire history plus your new commit to your empty GitHub repository.

*Note: The default branch is likely `main`, but it could be `master` depending on the age of the original repo. You can check by typing `git branch`.*

```bash
git push -u origin main
```

*(If your current branch is named `master`, use `git push -u origin master` instead).*