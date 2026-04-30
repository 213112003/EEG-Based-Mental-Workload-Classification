EEG-Based Mental Workload Classification

Overview
This repository presents a deep learning-based approach for classifying Mental Workload (MWL) using EEG signals. The classification is performed into three levels:
- Low
- Medium
- High
The study utilizes the EEGNet model along with hyperparameter tuning and explainability analysis.

The dataset used in this work is open access (online available) and available at: https://doi.org/10.5281/zenodo.5055046

Methodology
The EEGNet model is employed for efficient EEG signal classification. The model is designed to capture both temporal and spatial features of EEG data.
Hyperparameter Tuning
The following EEGNet hyperparameters were fine-tuned:
- F1 (number of temporal filters)
- F2 (number of pointwise filters)
- D (depth multiplier)
The provided implementation uses:  F1 = 64, F2 = 128 and D = 4  
To reproduce results for other configurations reported in the manuscript, users can modify these parameters within the code.
Experimental Setup
Experiments were conducted under two conditions:
1. Without SMOTE
2. With SMOTE 
Two separate scripts are provided:
-code_without_smote.py
-code_with_smote.py
All other parameters and experimental settings were kept identical in both cases to ensure a fair and unbiased comparison between the two conditions.
All experiments were executed on a High-Performance Computing (HPC) system using PuTTY for remote access. The Python version 3.8.5, TensorFlow version  2.13.1, and MNE Version 1.6.1 were used.

