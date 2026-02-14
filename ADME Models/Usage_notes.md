## Inputs to the scripts:

For each script, locate the code section with the file names. For example, in the case of LogD: 

1. LogD_POLARIS_FILE = 'LogD_POLARIS.csv'   # Polaris train dataset

2. LogD_AUGMENTED_FILE = 'LogD_MERGED.csv'  # Augmented train dataset

3. LogD_TEST_FILE = 'polaris-test.csv'      # Test dataset

4. LogD_TARGET_COLUMN = 'LogD'              # Column name for the labelled data.

Similarly, create the {endpoint}_files in the path of the scripts. There are two training dataset options per endpoint. 

- Required columns in the train datasets : SMILES, {TARGET_COLUMN}
- Required column in the test dataset  : SMILES 

- descriptors_55.txt, descriptors_25.txt: Descriptor lists (one per line) # provided in the repository

## Output Files:

- Model Files (10 per experiment):
  
  - model_{config_name}\_split_{0-9}.pth

  - Serialized XGBoost models (one per split)

- Prediction Files (per experiment):

  - Individual split predictions (10 files):

    - predictions_{config_name}\_split_{0-9}\_blind_test.csv

    - Columns: SMILES, Predicted_{TARGET_COLUMN}\_split_{i}

## Ensemble ADME models

### Inputs

#### 1. Ground Truth Data (in unblind_data/ directory)

**Files:** `{ENDPOINT}_unblind.csv` for each endpoint

**Endpoints:** HLM, KSOL, MLM, LogD, MDR1-MDCK2

**Must contain:**
- SMILES column
- Target column matching endpoint name (e.g., HLM, KSOL, etc.)

#### 2. Metrics Files (in results_{ENDPOINT}/ directories)

**Files:** `{ENDPOINT}_HOLDOUT_LOG10.csv`

**Must contain columns:**
- Dataset, Features, Strategy (combined into config)
- Seed (split number)
- MAE, MSE, R2, PEARSON_R, SPEARMAN_R

#### 3. Prediction Files (various patterns searched)

**Pattern:** `predictions_{endpoint}_{dataset}_{features}_{strategy}_split_{split}_test.csv`

**Must contain:**
- SMILES column
- Prediction column (containing "Predicted" or "prediction")

### Outputs

**All outputs are saved in ensemble_strategies_output/ directory**

**For Each Endpoint (ensemble_strategies_output/{ENDPOINT}/)**

#### 1. Prediction CSV Files (3 files per endpoint)

- `Strategy_1_Best_Single_{ENDPOINT}_predictions.csv`
- `Strategy_2_Top3_Ensemble_{ENDPOINT}_predictions.csv`
- `Strategy_3_Best_from_Groups_{ENDPOINT}_predictions.csv`

Each contains:
- SMILES
- prediction (transformed to match ground truth scale)
- ground_truth (transformed)

#### 2. Evaluation Metrics CSV (1 file per endpoint)

`{ENDPOINT}_strategy_evaluation.csv`

Contains for each strategy:
- MAE, MSE, RMSE, R²
- Pearson R & p-value
- Spearman R & p-value
- Kendall Tau & p-value
- N_samples

#### 3. Visualizations (3 plots per endpoint)

- `{ENDPOINT}_strategy_scatter_plots.png` - Predictions vs ground truth
- `{ENDPOINT}_strategy_comparison.png` - Bar charts comparing all metrics
- `{ENDPOINT}_residual_plots.png` - Residual distributions
