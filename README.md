# iCMAMP


The repo is organised as follows:

| FILE NAME         | DESCRIPTION                                                  |
| :---------------- | :----------------------------------------------------------- |
| /data             | The folder includes structural and sequence information for positive and negative samples and is divided into training and test sets |
| /feature_data     | The folder contains 7 sets of feature information extracted from structures and sequences |
| evaluation.py     | Evaluation metrics for the model (iCMAMP-2L)                 |
| pdb_to_sequence.py| Input PDB file and converted to natural amino acid sequence  |
| Feature.py        | Python code for extracting features from sample structures and sequences |
| iCMAMP_1L_main.py | Main process code of iCMAMP-1L                               |
| iCMAMP_2L_main.py | Main process code of iCMAMP-2L                               |
| loss.py           | Loss function of iCMAMP-2L                                   |
| model.py          | Monolithic model framework of iCMAMP-2L                      |
| pdb_to_dpc.py     | Input pdb file to obtain the corresponding DPC features      |
| result.txt        | Result of iCMAMP-2L                                          |

## Installation

- Requirement

  OS：

  - `Windows` ：Windows10 or later

  - `Linux`：Ubuntu 16.04 LTS or later

  Python：

  - `Python` == 3.8

- Download `iCMAMP`to your computer

  ```bash
  git clone https://github.com/swicher123/iCMAMP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd iCMAMP
  pip install -r requirement.txt
  ```
## Predicting structures and extracting ESM1-b feature

If you have a peptide sequence containing chemically modified residues, please use the PEPstrMOD tool to predict its three-dimensional structure and extract features using the ESM1-b model.

PEPstrMOD:
```
https://webs.iiitd.edu.in/raghava/pepstrmod/
```

ESM1-b:
```
https://github.com/facebookresearch/esm
```

## Input PDB file and convert to natural amino acid sequence

If you got a PDB file using PEPstrMOD and need to convert the PDB file into a peptide sequence containing natural amino acids, run the following command:

```bash
python pdb_to_sequence.py
```

In the `pdb_to_sequence.py` program:

```python
input_folder = "pdb_folder"  # pdb folder: stores multiple pdb files  
output_csv = "Sequence.csv"  # Sequence.csv: File storing peptide sequences
```

- `input_folder` is the folder where your PDB files are stored.
- `output_csv` is the path where the Sequence CSV file will be saved.

If you have new PDB files and want to convert to sequence, please update the file paths to the correct ones.

## Structural and sequence feature extraction

If you have already obtained the structure and sequence files for the peptide sample, then run the following command:

```bash
python Feature.py
```

In the `Feature.py` program:

```python
    pos_train_inputfilepath = './data/structure/pos_train'  # Positive sample training set PDB file storage path
    neg_train_inputfilepath = './data/structure/neg_train'  # Negative sample training set PDB file storage path
    pos_test_inputfilepath = './data/structure/pos_test'    # Positive sample testing set PDB file storage path
    neg_test_inputfilepath = './data/structure/neg_test'    # Negative sample testing set PDB file storage path
    filenames = ['pos_train_naturalseq.csv', 'neg_train_naturalseq.csv', 'pos_test_naturalseq.csv','neg_test_naturalseq.csv']  # List of sequence file names
```
If you want to extract features from structure and sequence,  please update the file paths to the correct ones.


## Training and test iCMAMP_1L model

After you have extracted features using Feature.py and ESM1-b, run the iCMAMP-1L model using the following command:

```
python iCMAMP_1L_main.py
```

In the `iCMAMP_1L_main.py` program:

```python
pos_feature_path = f'./feature_data/{df_pos_features[i]}_{dataset}.csv'  # Positive sample feature path
neg_feature_path = f'./feature_data/{df_neg_features[i]}_{dataset}.csv'  # Negative sample feature path
```

If you want to run iCMAMP-1L correctly, please update the path correctly.

## Input PDB file to obtain the corresponding DPC feature file

If you have new PDB files and want to obtain the corresponding DPC features, please run the command:

```bash
python pdb_to_dpc.py
```

In the `pdb_to_dpc.py` program:

```python
input_folder = "pdb_folder"  # pdb folder: stores multiple pdb files  
output_csv = "feature_folder/DPC_feature.csv"  # DPC feature save path
```

- `input_folder` is the folder where your PDB files are stored.
- `output_csv` is the path where the DPC feature CSV file will be saved.

If you have new PDB files and want to obtain the corresponding DPC features, please update the file paths to the correct ones.

## Training and test iCMAMP_2L model

```
python iCMAMP_2L_main.py
```

In the `iCMAMP_2L_main.py` program:

```python
train_feature = read_csv_feature('feature_data', 'DPC_train.csv').values  
test_feature = read_csv_feature('feature_data', 'DPC_test.csv').values  
y_train = get_labels(first_dir, 'train_label')  
y_test = get_labels(first_dir, 'test_label')  
```

These correspond to the file paths for the training set DPC features, test set DPC features, training set labels, and test set labels.

- The DPC feature files are in CSV format.
- The label files are in TXT format.

If there are new dpc feature files and corresponding label files, please update the file paths accordingly.
