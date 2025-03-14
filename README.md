# iCMAMP


The repo is organised as follows:

| FILE NAME         | DESCRIPTION                                                  |
| :---------------- | :----------------------------------------------------------- |
| /data             | The folder includes structural and sequence information for positive and negative samples and is divided into training and test sets |
| /feature_data     | The folder contains 7 sets of feature information extracted from structures and sequences |
| evaluation.py     | Evaluation metrics for the model (iCMAMP-2L)                 |
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

## Training and test iCMAMP_1L model

```
python iCMAMP_1L_main.py
```

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
