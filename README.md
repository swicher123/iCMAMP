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

/data: The folder includes structural and sequence information for positive and negative samples and is divided into training and test sets

/feature_data: The folder contains 7 sets of feature information extracted from structures and sequences

count.py：

evaluation.py：

Feature.py： Python code for extracting features from sample structures and sequences

iCMAMP_1L_main.py：Main process code of iCMAMP-1L

iCMAMP_2L_main.py：Main process code of iCMAMP-2L

loss.py：

model.py：

multiple_label.py：

test.py：

train.py：

tSNE.py：

UMap.py：


