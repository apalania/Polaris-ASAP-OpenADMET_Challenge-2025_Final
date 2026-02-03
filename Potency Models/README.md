## Scripts for training Potency models: a combination of classical ML + deep learning  

1. Pretrained_GNN-transformer_ChEMBL_random-split.py
Script for training the baseline model of potency prediction using ligand-protein pairs discovered from ChEMBL - UniProt.

2. Fine-tuned_GPFT_T-stratified_random.py
3. Fine-tuned_GPFT_T-stratified_scaffold.py
Gradual partial (progressive) fine-tuning of the baseline ChEMBL data using the Polaris potency train data. GPFT was done with scaffold split or random split.  

4. AttFPGNN-transformer_Polaris-only_T-stratified_random.py
5. ATTFPGNN-transformer_Polaris-only_T-stratified_scaffold.py
Script for training _de novo_ AttentiveFingerPrint GNN-transformer architectures for learning the molecule potency against SARS and MERS _M^Pro_ targets. Options to include docking scores in the final model are provided in the script. Both splits are shown - Random & Scaffold.                                                                                                                                                                    6. Ensembles_Early-Late-Stacking_random.py
7. Ensembles_Early-Late-Stacking_scaffold.py

Ensemble methods for the various model classes, to determine the best pair combination when evaluated on unseen blindset. Three techniques were considered: Early fusion, late fusion, and stacking meta-learner.



XGBoost-models_25D-55D_docking_random-scaffold.py
