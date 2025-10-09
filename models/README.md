# Models Directory

This directory contains trained model checkpoints.

## Model Files (Not Included in Repository)

Due to large file sizes, model checkpoints are not included in this repository.

## To Reproduce Models

1. Phase 1-3: Run baseline and tuning scripts
2. Phase 4: Run TAPT and fine-tuning
3. Phase 5: Load saved models for ensemble

## Model Structure

    models/
    ├── kcbert_tapt_final_fixed/     KcBERT with TAPT (Primary)
    ├── electra_final_v2_fixed/      ELECTRA (Secondary)
    └── ensemble/                     Ensemble predictions

## Download Pre-trained Models

Contact the repository owner for pre-trained model weights.

## WandB Integration

All training runs are logged to Weights & Biases.
