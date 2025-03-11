# iCMAMP

Prediction of Chemically Modified Antimicrobial Peptides and Their Sub-functional Activities using hybrid features 
Yujie Yao1,#, Daijun Zhang1,#, Henghui Fan1, Junfeng Xia1, Ting Wu2,3,*, Yansen Su4,*, Yannan Bin1,*
1Information Materials and Intelligent Sensing Laboratory of Anhui Province, Institutes of Physical Science and Information Technology, Anhui University, Hefei, Anhui 230601, China.
2Department of Infectious Diseases & Anhui Center for Surveillance of Bacterial Resistance, The First Affiliated Hospital of Anhui Medical University, Hefei, 230022, China
3Anhui Province Key Laboratory of Infectious Diseases & Institute of Bacterial Resistance, Anhui Medical University, Hefei 230022, China
4School of Artificial Intelligence, Anhui University, Hefei, Anhui 230601, China.
#These authors contributed equally to this work.
Corresponding authors: wutingf88945@163.com (T. W.), suyansen@ahu.edu.cn (Y. S.), and ynbin@ahu.edu.cn (Y. B.).


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

## Training and test iCMAMP_2L model

```shell
cd iCMAMP
python iCMAMP_1L_main.py
python iCMAMP_2L_main.py
```

