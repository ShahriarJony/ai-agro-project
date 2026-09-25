# AI Training Workspace for AgroAI

## Purpose

This workspace provides isolated environment for AI model training and development, keeping ML training work separate from the main AgroAI application.

## Kaggle API Setup

1. **Install required dependencies:**
   ```bash
   pip install kaggle pandas numpy
   ```

2. **Configure Kaggle credentials:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Kaggle credentials:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

3. **Run dataset verification:**
   ```bash
   cd ai_training && KAGGLE_USERNAME=testuser KAGGLE_KEY=testkey1234 python scripts/verify_kaggle_dataset.py
   ```

## Workspace Structure

- `scripts/` - Python scripts for dataset verification and processing
- `config/` - Configuration files
- `data/` - Raw dataset (never committed)
- `models/` - Trained model files (never committed)
- `outputs/` - Training outputs and reports
- `reports/` - Analysis reports

## Phase 1

This workspace is for Phase 1: Kaggle Dataset Access & Verification ONLY. It does NOT include model training.

**Phase 1 intentionally does NOT do:**
- Train CNN models
- Download full multi-GB datasets
- Modify existing application code
- Commit raw datasets

**Phase 1 does:**
- Verify Kaggle API access
- Identify and catalog available plant disease datasets
- Analyze dataset structure and classes
- Create comprehensive dataset reports

## Notes

- Raw datasets should be cached in `data/raw/` for subsequent phases
- The `.gitignore` protects sensitive files and large datasets
- This workspace remains isolated from the main AgroAI application
