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

pos_train_inputfilepath = './positive-example/pos_train/pos_train'
neg_train_inputfilepath = './negative-example/neg_train/neg_train'
pos_test_inputfilepath = './positive-example/pos_test/pos_test'
neg_test_inputfilepath = './negative-example/neg_test/neg_test'
inputfilepaths = [pos_train_inputfilepath,neg_train_inputfilepath,pos_test_inputfilepath,neg_test_inputfilepath]

def multi_Convert (file_PATH,input_format,output_format):

    tqdm = os.listdir(file_PATH)   # 文件夹中的文件列表
    for i in range(0, len(tqdm)):  # 逐次遍历文件夹下的文件
        inputfile = os.path.join(file_PATH, tqdm[i])  # 对应文件夹下的某份文件
        if inputfile.endswith(input_format):
            conv = openbabel.OBConversion()           # 调用转换函数
            conv.OpenInAndOutFiles(inputfile, inputfile.split(input_format)[0] + output_format)  # 输入待转换的文件名及定义转换成功后的文件名
            conv.SetInAndOutFormats(input_format, output_format)
            conv.Convert()
            conv.CloseOutFile()
        else:
            continue

def descriptor_Fingerprints_caculation(inputfilepath,descriptor: bool = False,fingerprints: bool = False):

    filenames = os.listdir(inputfilepath)
    output_filename = inputfilepath.split('/')[2]
    print(output_filename)
    df = pd.DataFrame()
    for filename in filenames:

        if filename.endswith('.mdl') and fingerprints == True:

            inputfile = os.path.join(inputfilepath,filename)

            try:
                dep = from_mdl(inputfile,fingerprints=fingerprints, descriptors=descriptors)
            except:
                print(filename)
                continue

            df1 = pd.DataFrame(list(dep[0].values()),index=list(dep[0].keys())).T
            df1.insert(0,'ID',filename)
            df = pd.concat([df,df1],ignore_index=True)

        elif filename.endswith('.pdb') and  descriptor == True:

            inputfile = os.path.join(inputfilepath, filename)

            # 创建计算器对象
            calc = Calculator(descriptors, ignore_3D=True)

            # 将pdb文件转化为mol
            mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(inputfile)

            if mol == None:
                with open(inputfile.split('.pdb')[0] + '.smi', 'r') as file:
                    smi = file.read().strip()

                # 将SMILES字符串转换为Mol对象
                mol = Chem.MolFromSmiles(smi)

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


    if descriptor == True and fingerprints == False:
        df.to_csv('descriptor_rdkit{}.csv'.format(output_filename), index=False)
    else:
        df.to_csv('fingerprints_{}.csv'.format(output_filename), index=False)
    print('-----------------')

def Atom_percentage(inputfilepath):

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

    df.to_csv('Atom_count_{}.csv'.format(output_filename),index=False)

def extract_atom_pairs(inputfile):

    mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(inputfile)
    mol2 = Chem.AddHs(mol)
    atom_pairs = []

    for bond in mol2.GetBonds():
        atom1 = bond.GetBeginAtom().GetSymbol()
        atom2 = bond.GetEndAtom().GetSymbol()
        pair = f"{atom1}-{atom2}"
        atom_pairs.append(pair)
    return atom_pairs

def AAC(ID,sequence):

    AA = {'ID':'','C':0,'A':0,'E':0,'F':0,'D':0,'G':0,'H':0,'K':0,'L':0,'M':0,'I':0,'N':0,'Q':0,'P':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}

    try:
        count = Counter(sequence)

        for key in count:
            count[key] = count[key] / len(sequence) * 100
            AA[key] = count[key]
        AA['ID'] = ID
        return AA

    except:
        print(ID)

def DPC(filename):

    df = pd.read_csv(filename)
    AA =  'ACDEFGHIKLMNPQRSTVWY'
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

    df1.to_csv('DPC_' + filename.split('_')[0] + '_' + filename.split('_')[1] + '.csv', index=False)

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

for inputpath in inputfilepaths:

    df = pd.DataFrame()
    filenames = os.listdir(inputpath)
    output_filename = inputpath.split('/')[2]
    for filename in filenames:
#
#         # 将pdb格式转化为mdl格式
#         # multi_Convert(inputpath, 'pdb', 'smi')
#
        # 计算2D化学修饰符特征
        descriptor_Fingerprints_caculation(inputpath,descriptor = True)
#
#         # 计算原子组成百分比
#         #Atom_percentage(inputpath)
#
#         # 计算指纹特征
#         #descriptor_Fingerprints_caculation(inputpath,descriptors = False,fingerprints = True)
#
#         # 计算原子对特征
#         if filename.endswith(".pdb"):
#
#             ID = filename.split(".")[0]
#             atom_pair_ratio = {'ID': 0, 'C-C': 0, 'C-O': 0, 'N-C': 0, 'H-C': 0, 'O-N': 0, 'C-H': 0, 'C-N': 0, 'N-H': 0,
#                                'O-C': 0, 'O-O': 0, 'H-N': 0, 'C-S': 0, 'S-C': 0,'O-H': 0,'S-H': 0,'S-S': 0}
#             inputfile = os.path.join(inputpath,filename)
#
#             try:
#                 atom_pairs = extract_atom_pairs(inputfile)
#             except:
#
#                 atom_pair_ratio_error = {'ID': 0, 'C-C': np.nan, 'C-O': np.nan, 'N-C': np.nan, 'H-C': np.nan, 'O-N': np.nan, 'C-H': np.nan, 'C-N': np.nan, 'N-H': np.nan,'O-C': np.nan, 'O-O': np.nan, 'H-N': np.nan, 'C-S': np.nan, 'S-C': np.nan}
#                 atom_pair_ratio_error['ID'] = ID
#                 df = df.append(atom_pair_ratio_error,ignore_index= True)
#                 print(output_filename + '_' + filename)
#                 continue
#
#             # 统计原子对的数量
#             atom_pair_counts = Counter(atom_pairs)
#             total_count = sum(atom_pair_counts.values())
#
#             # 计算原子对的占比特征
#             for pair, count in atom_pair_counts.items():
#                 atom_pair_ratio[pair] = count / total_count * 100
#
#             # 存储计算结果
#             atom_pair_ratio['ID'] = ID
#             df = df.append(atom_pair_ratio,ignore_index= True)

            # except:
            #     df_ID = pd.DataFrame({'ID': ID})
            #     df = pd.concat([df, df_ID], ignore_index=True)
            #     continue

    # id_column = df['ID']
    # df.drop(labels = 'ID',axis = 1,inplace = True)
    # df.insert(0,'ID',id_column)
    # df.to_csv(f'diatom_{output_filename}.csv',index=False)

#filenames = ['pos_train_1_seq.csv','neg_train_1_seq.csv','pos_test_1_seq.csv','neg_test_1_seq.csv']
# k_value = [2,3,4,5]
#
# for k in k_value:
# for filename in filenames:
#           df1 = pd.DataFrame()
# #         df2 = pd.DataFrame()
# #         df3 = pd.DataFrame()
# #         df_binary = pd.DataFrame()
#           df = pd.read_csv(filename)
# #         seq_list = []
#           for index, row in df.iterrows():
# #           #try:
#             ID = row['ID'].split('.')[0]
#             sequence = row['sequence']
# #
#             # 序列处理
#             seq = sequence[:k] + sequence[-k:]
#             while len(seq) < 2 * k:
#                 seq += seq
#
#             # 使用one_hot编码
#             one_hot_seq = one_hot(seq)
#
#             # 每个sequence都添加到列表中
#             seq_list.append(one_hot_seq)
#
#         df_binary = pd.DataFrame(seq_list)
#         df_binary.to_csv(filename.replace('_1_seq.csv',f'_binary_NT_CT_{k}.csv'), index=False)
            # 计算氨基酸组成特征
          #   AA = AAC(ID,sequence)
          #   df1 = df1.append(AA, ignore_index= True)
          #   df1.insert(0, 'ID', df1.pop('ID'))
          # df1.to_csv('AAC_' + filename.split('_')[0] + '_' + filename.split('_')[1] + '.csv', index=False)

            # 计算二肽组成特征
            # DPC(filename)

            # 计算CT-k、NT-k 特征
        #     if len(sequence) < 5:
        #         continue
        #
        #     seq1 = sequence[:k]
        #     AAC_NT5 = AAC(ID,seq1)
        #     df2 = df2.append(AAC_NT5, ignore_index=True)
        #     df2.insert(0, 'ID', df2.pop('ID'))
        #
        #     seq2 = sequence[-k:]
        #     AAC_CT5 = AAC(ID,seq2)
        #     df3 = df3.append(AAC_CT5, ignore_index=True)
        #     df3.insert(0, 'ID', df3.pop('ID'))
        #

          #except:
             # print(filename + ID)
        #
        # df2.to_csv('AAC_NT5_' + filename.split('_')[0] + '_' + filename.split('_')[1] + '.csv', index=False)
        # df3.to_csv('AAC_CT5_' + filename.split('_')[0] + '_' + filename.split('_')[1] + '.csv', index=False)

