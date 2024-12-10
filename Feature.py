import sys
from collections import Counter
sys.path.append(r"E:\Anaconda\Lib\site-packages")
from padelpy import from_mdl, padeldescriptor
import pandas as pd
import openbabel
import os
from mordred import Calculator, descriptors
from rdkit import Chem
import rdkit
import numpy as np



def multi_Convert (file_PATH,input_format,output_format):

    tqdm = os.listdir(file_PATH)   # 文件夹中的文件列表
    for i in range(0, len(tqdm)):  # 逐次遍历文件夹下的文件
        inputfile = os.path.join(file_PATH, tqdm[i])  # 对应文件夹下的某份文件
        if inputfile.endswith(input_format):
            conv = openbabel.OBConversion()           # 调用转换函数
            conv.OpenInAndOutFiles(inputfile, inputfile.split(input_format)[0] + output_format)  # 输入待转换的文件名及定义转换成功后的文件名
            conv.SetInAndOutFormats(input_format.split('.')[1], output_format.split('.')[1])
            conv.Convert()
            conv.CloseOutFile()
        else:
            continue

def _2DMD_Fingerprint_caculation (inputfilepath,_2DMD: bool = False, Fingerprint: bool = False):

    filenames = os.listdir(inputfilepath)
    output_filename = inputfilepath.split('/')[3]
    print(output_filename)
    df = pd.DataFrame()
    for filename in filenames:

        if filename.endswith('.mdl') and Fingerprint == True:

            inputfile = os.path.join(inputfilepath,filename)

            try:
                dep = from_mdl(inputfile,fingerprints = Fingerprint, descriptors = descriptors)
            except:
                print(filename)
                continue

            df1 = pd.DataFrame(list(dep[0].values()),index=list(dep[0].keys())).T
            df1.insert(0,'ID',filename)
            df = pd.concat([df,df1],ignore_index=True)

        elif filename.endswith('.pdb') and _2DMD == True:

            inputfile = os.path.join(inputfilepath, filename)

            # 创建计算器对象
            calc = Calculator(descriptors, ignore_3D=True)

            # 将pdb文件转化为mol
            mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(inputfile)

            # if mol == None:
            #     with open(inputfile.split('.pdb')[0] + '.smi', 'r') as file:
            #         smi = file.read().strip()
            #
            #     # 将SMILES字符串转换为Mol对象
            #     mol = Chem.MolFromSmiles(smi)

            try:
                # 计算多肽的2D修饰符
                result = calc(mol)

            except:

                df_ID = pd.DataFrame({'ID': [filename.split('.pdb')[0]]})
                df = pd.concat([df, df_ID], ignore_index=True)
                continue

            # 将计算出的2D修饰符放到DataFrame中
            df_mordred = pd.DataFrame.from_dict(result, orient='index', columns=['value']).T

            # 插入ID列
            df_mordred.insert(0, 'ID', filename.split('.pdb')[0])

            # 将计算出的数据与原始数据合并
            df = pd.concat([df, df_mordred], ignore_index=True)

        else:
            continue


    if _2DMD == True and Fingerprint == False:
        df.to_csv('./feature_data/2DMD{}.csv'.format(output_filename), index=False)
    else:
        df.to_csv('./feature_data/Fingerprint_{}.csv'.format(output_filename), index=False)
    print('-----------------')

def ATC(inputfilepath):

    ratio = {}

    df = pd.DataFrame()

    output_filename = inputfilepath.split('/')[2]

    filenames = os.listdir(inputfilepath)

    for filename in filenames:

        Atom = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'B': 0}
        total_Atom = 0

        if filename.endswith('.pdb'):

            inputfile = os.path.join(inputfilepath, filename)

            with open(inputfile,'r') as file:

                for line in file:

                    if line.startswith('ATOM'):
                        atom = line[12:16].strip()[0]
                        if not atom.isalpha():
                            atom = line[12:16].strip()[1]

                        if atom in Atom:
                            Atom[atom] += 1
                        total_Atom += 1

                if total_Atom == 0:
                    continue

                for atom_name,count in Atom.items():

                    ratio[atom_name] = count / total_Atom * 100

                df_ratio = pd.DataFrame([ratio])
                df_ratio.insert(0, 'ID', filename.split('.pdb')[0])
                df = pd.concat([df,df_ratio], ignore_index=True)

    df.to_csv('./feature_data/ATC_{}.csv'.format(output_filename),index=False)


def AAC(filepath):

    # 初始化一个空列表，用于存储所有序列的AAC特征
    AA_list = []
    df = pd.read_csv(filepath)


    try:
        for index, row in df.iterrows():
            ID, sequence = row['ID'], row['sequence']
            count = Counter(sequence)
            # 创建一个新的字典来存储每个序列的AAC特征
            AA = {'ID': '', 'C': 0, 'A': 0, 'E': 0, 'F': 0, 'D': 0, 'G': 0, 'H': 0, 'K': 0, 'L': 0, 'M': 0, 'I': 0,
                  'N': 0, 'Q': 0, 'P': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
            for key in count:
                count[key] = count[key] / len(sequence) * 100
                AA[key] = count[key]
            AA['ID'] = ID
            # 将当前序列的AAC特征添加到列表中
            AA_list.append(AA)

    except Exception as e:
        # 打印错误信息和当前处理的ID
        print(f"Error processing row with ID {ID}: {e}")

    # 将AAC特征列表转换为DataFrame
    df_AAC = pd.DataFrame(AA_list)
    df_AAC.to_csv(f'./feature_data/AAC_{filename.split("_")[0] + "_" + filename.split("_")[1]}.csv', index=False)

def DPC(filepath):

    df = pd.read_csv(filepath)
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    df1 = pd.DataFrame(columns = ['ID_'] + diPeptides)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for index, row in df.iterrows():
        try:
            ID, sequence = row['ID'], row['sequence']
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] + 1
            if sum(tmpCode) != 0:
                tmpCode = [i / sum(tmpCode) * 100 for i in tmpCode]
                tmpCode.insert(0, ID.split('.')[0])
            df1.loc[index] = tmpCode
        except:
            df1.loc[index] = np.nan

    df1.to_csv('./feature_data/DPC_' + filename.split('_')[0] + '_' + filename.split('_')[1] + '.csv', index=False)

def one_hot (seq):
    mapping = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
    one_hot_Seq = np.array([mapping[residue] for residue in seq])
    return one_hot_Seq.flatten()

def NTCT5(filepath):

          df = pd.read_csv(filepath)
          df_binary = pd.DataFrame()
          seq_list = []

          for index, row in df.iterrows():

            ID = row['ID'].split('.')[0]
            sequence = row['sequence']

            # 序列处理
            while len(sequence) < 10:
                sequence += sequence
            seq = sequence[: 5] + sequence[-5:]

            # 使用one_hot编码
            one_hot_seq = one_hot(seq)

            # 每个sequence都添加到列表中
            seq_list.append(one_hot_seq)

          df_binary = pd.DataFrame(seq_list)
          df_binary.to_csv(f'./feature_data/NTCT5_{filename.split("_")[0] + "_" + filename.split("_")[1]}.csv', index=False)

# 主程序入口
if __name__ == "__main__":

    # 结构文件路径
    pos_train_inputfilepath = './data/structure/pos_train'
    neg_train_inputfilepath = './data/structure/neg_train'
    pos_test_inputfilepath = './data/structure/pos_test'
    neg_test_inputfilepath = './data/structure/neg_test'
    inputfilepaths = [pos_train_inputfilepath, neg_train_inputfilepath, pos_test_inputfilepath, neg_test_inputfilepath]

    # 序列文件名称
    filenames = ['pos_train_naturalseq.csv', 'neg_train_naturalseq.csv', 'pos_test_naturalseq.csv','neg_test_naturalseq.csv']

    # 提取结构特征
    for inputpath in inputfilepaths:
    #
            # 将pdb格式转化为mdl格式
            # multi_Convert(inputpath, '.pdb', '.mdl')
    #
            # 计算2D化学修饰符特征
            _2DMD_Fingerprint_caculation(inputpath,_2DMD = True)

            # 计算原子组成百分比
            ATC(inputpath)

            # 计算指纹特征
            _2DMD_Fingerprint_caculation(inputpath,_2DMD = False,Fingerprint = True)

    # 提取序列特征
    for filename in filenames:
        filepath = f'./data/sequence/{filename}'

        # 提取AAC特征
        AAC(filepath)

        # 提取DPC特征
        DPC(filepath)

        # 提取NTCT5特征
        NTCT5(filepath)



