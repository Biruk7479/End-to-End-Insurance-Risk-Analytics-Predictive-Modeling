# DVC Setup Guide

## Overview
Data Version Control (DVC) is used to track and version our insurance dataset, ensuring reproducibility and auditability of our analysis.

## Installation

```bash
pip install dvc
```

## Initial Setup

### 1. Initialize DVC
```bash
cd /home/aj7479/Desktop/KAIM/Week-3
dvc init
```

This creates a `.dvc` directory with DVC configuration files.

### 2. Create Local Remote Storage
```bash
# Create a directory for local DVC storage
mkdir -p /home/aj7479/Desktop/KAIM/dvc-storage

# Add it as a DVC remote
dvc remote add -d localstorage /home/aj7479/Desktop/KAIM/dvc-storage
```

### 3. Add Data to DVC
```bash
# Add your data file (update path as needed)
dvc add data/MachineLearningRating_v3.txt

# This creates a data/MachineLearningRating_v3.txt.dvc file
```

### 4. Commit DVC Files to Git
```bash
git add data/MachineLearningRating_v3.txt.dvc data/.gitignore .dvc/config
git commit -m "feat(task-2): add data to DVC tracking"
```

### 5. Push Data to Remote
```bash
dvc push
```

## Data Versioning Workflow

### Creating Data Versions

1. **Initial Version (v1.0 - Raw Data)**
```bash
git add data/*.dvc .dvc/config
git commit -m "data: add raw insurance data v1.0"
git tag -a v1.0-data -m "Raw insurance data from Feb 2014 - Aug 2015"
dvc push
```

2. **Version 2 (v2.0 - Cleaned Data)**
After preprocessing:
```bash
dvc add data/processed/insurance_data_clean.csv
git add data/processed/*.dvc
git commit -m "data: add cleaned insurance data v2.0"
git tag -a v2.0-data -m "Cleaned data with missing values handled"
dvc push
```

3. **Version 3 (v3.0 - Feature Engineered)**
After feature engineering:
```bash
dvc add data/processed/insurance_data_features.csv
git add data/processed/*.dvc
git commit -m "data: add feature-engineered data v3.0"
git tag -a v3.0-data -m "Data with engineered features"
dvc push
```

### Switching Between Data Versions

To retrieve a specific version of the data:

```bash
# Checkout the git tag for the desired data version
git checkout v1.0-data

# Pull the corresponding data from DVC
dvc pull

# Return to main branch
git checkout main
dvc pull
```

## DVC Commands Reference

### Basic Operations
```bash
# Check DVC status
dvc status

# Add file/directory to DVC
dvc add <path>

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Check out specific version
dvc checkout
```

### Remote Management
```bash
# List remotes
dvc remote list

# Add remote
dvc remote add -d <name> <url>

# Modify remote
dvc remote modify <name> <option> <value>

# Remove remote
dvc remote remove <name>
```

### Data Pipeline
```bash
# Run DVC pipeline
dvc repro

# Show pipeline DAG
dvc dag
```

## Best Practices

1. **Commit .dvc files to Git**: Always commit the `.dvc` files and `.dvc/config` to Git
2. **Don't commit actual data**: The actual data files are in `.gitignore`
3. **Tag versions**: Use Git tags to mark data versions
4. **Document changes**: Always document what changed in each data version
5. **Push after changes**: Always run `dvc push` after adding/modifying data

## Troubleshooting

### Issue: DVC not found
```bash
pip install dvc
# or
pip install dvc[all]
```

### Issue: Permission denied on remote
Check that you have write permissions to the remote storage directory.

### Issue: Data not updating
```bash
# Force update
dvc checkout --relink
dvc pull --force
```

## Directory Structure with DVC

```
Week-3/
├── .dvc/
│   ├── config              # DVC configuration
│   └── .gitignore
├── data/
│   ├── .gitignore          # Ignores actual data files
│   ├── raw/
│   │   ├── insurance_data.txt
│   │   └── insurance_data.txt.dvc  # DVC tracking file (in Git)
│   └── processed/
│       ├── cleaned_data.csv
│       └── cleaned_data.csv.dvc    # DVC tracking file (in Git)
└── ...
```

## Integration with CI/CD

In your GitHub Actions workflow:

```yaml
- name: Set up DVC
  run: |
    pip install dvc
    dvc remote add -d localstorage ${{ secrets.DVC_REMOTE }}
    dvc pull
```

## Monitoring Data Changes

Check what data has changed:
```bash
# Show differences
dvc diff

# Show metrics changes
dvc metrics diff
```

## Automated Setup

Use the provided script for automated setup:

```python
from scripts.dvc_setup import setup_dvc_pipeline

setup_dvc_pipeline(
    project_root=".",
    data_file="data/insurance_data.txt",
    local_storage="/home/aj7479/Desktop/KAIM/dvc-storage"
)
```

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- [DVC with Git](https://dvc.org/doc/use-cases/versioning-data-and-model-files)
