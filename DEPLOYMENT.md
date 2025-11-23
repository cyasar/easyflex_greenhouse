# GitHub Deployment Instructions

This guide explains how to deploy the QIM Greenhouse Optimization project to GitHub.

## Prerequisites

1. Git installed on your system
2. GitHub account
3. Project files ready for deployment

## Step-by-Step Deployment

### 1. Initialize Git Repository

Navigate to the project directory:

```bash
cd QIM-Greenhouse-Optimization
```

Initialize a new Git repository:

```bash
git init
```

### 2. Stage All Files

Add all project files to Git:

```bash
git add .
```

**Note:** Files listed in `.gitignore` will be excluded automatically.

### 3. Create Initial Commit

Commit all files with a descriptive message:

```bash
git commit -m "QIM greenhouse optimisation codes uploaded"
```

### 4. Create Main Branch

Ensure you're on the main branch:

```bash
git branch -M main
```

### 5. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `QIM-Greenhouse-Optimization`
3. Description: "Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model"
4. Choose Public or Private
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 6. Add GitHub Remote

Replace `USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/USERNAME/QIM-Greenhouse-Optimization.git
```

**Alternative (SSH):**
```bash
git remote add origin git@github.com:USERNAME/QIM-Greenhouse-Optimization.git
```

### 7. Push to GitHub

Push your code to the remote repository:

```bash
git push -u origin main
```

You may be prompted for GitHub credentials. Use:
- Personal Access Token (recommended) instead of password
- Or SSH key if using SSH URL

## Verifying Deployment

After pushing, verify your repository:

1. Visit: `https://github.com/USERNAME/QIM-Greenhouse-Optimization`
2. Check that all files are present
3. Verify README.md displays correctly

## Updating the Repository

After making changes:

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

## Repository Structure Checklist

Before deploying, ensure your repository has:

- [x] `README.md` - Comprehensive project documentation
- [x] `requirements.txt` - Python dependencies
- [x] `.gitignore` - Git ignore rules
- [x] `CITATION.cff` - Citation information
- [x] `data/` - Dataset directory (may be empty if dataset is large)
- [x] `preprocessing/` - Data preprocessing modules
- [x] `analysis/` - Analysis modules and scripts
- [x] `dwave_annealing/` - D-Wave quantum annealing code
- [x] `qiskit_simulations/` - Qiskit simulations (if applicable)
- [x] `results/` - Output directory (may be empty initially)

## Important Notes

### Data Files

- Large dataset files (`greenhouse_data.xlsx`) may exceed GitHub's file size limits
- Consider using Git LFS for large files:
  ```bash
  git lfs install
  git lfs track "*.xlsx"
  git add .gitattributes
  git add data/greenhouse_data.xlsx
  ```
- Or provide download instructions in README.md

### Sensitive Information

- **Never commit** D-Wave API tokens or credentials
- Ensure `.gitignore` excludes `.env`, `*.token`, and `dwave_config.ini`
- Use environment variables for sensitive data

### Results Directory

- Generated results (`results/figures/`, `results/reports/`) can be committed or ignored
- For reproducibility, consider committing example results
- Add `results/` to `.gitignore` if you want to exclude generated files

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

```bash
# Use Personal Access Token instead of password
# Generate token: GitHub Settings > Developer settings > Personal access tokens

# Or configure SSH
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add public key to GitHub: Settings > SSH and GPG keys
```

### Large File Issues

If files are too large:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.xlsx"
git lfs track "*.png"
git add .gitattributes
```

### Push Rejected

If push is rejected:

```bash
# Pull remote changes first
git pull origin main --allow-unrelated-histories

# Resolve conflicts if any, then push
git push origin main
```

## Next Steps

After successful deployment:

1. Add repository description and topics on GitHub
2. Create releases for major versions
3. Set up GitHub Actions for CI/CD (optional)
4. Add collaborators if working in a team
5. Enable GitHub Pages for documentation (optional)

## Support

For issues with deployment, check:
- [GitHub Documentation](https://docs.github.com/)
- [Git Documentation](https://git-scm.com/doc)
- Project README.md for usage instructions

