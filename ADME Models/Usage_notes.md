## Inputs to the scripts:

For each script, locate the code section with the file names. For example, in the case of LogD: 

LogD_POLARIS_FILE = 'LogD_POLARIS.csv'   # Polaris train dataset

LogD_AUGMENTED_FILE = 'LogD_MERGED.csv'  # Augmented train dataset

LogD_TEST_FILE = 'polaris-test.csv'      # Test dataset

LogD_TARGET_COLUMN = 'LogD'              # Column name for the labelled data; 'SMILES' is the other column name.

Similarly, create the {endpoint}_files in the path of the scripts.

- Required columns in the train datasets : SMILES, {TARGET_COLUMN}
- Required column in the test dataset  : SMILES 

- descriptors_55.txt: List of 55 descriptors (one per line) # provided in the repository
- descriptors_25.txt: List of 25 descriptors (one per line) # provided in the repository

## Output Files:

- Model Files (10 per experiment):
  ● model_{config_name}_split_{0-9}.pth
  ● Serialized XGBoost models (one per split)
- Prediction Files (per experiment):
  ● Individual split predictions (10 files):
    o predictions_{config_name}\_split_{0-9}\_blind_test.csv
    o Columns: SMILES, Predicted_{TARGET_COLUMN}\_split_{i}
