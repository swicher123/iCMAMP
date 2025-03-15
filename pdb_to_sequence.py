import os
import csv
from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1

AA_CONVERT = {
    '(AAB)': 'A', '(NLE)': 'L', '(DIF)': 'F', '(NMW)': 'W', '(NMS)': 'S', '(NMA)': 'A',
    '(NVA)': 'V', '(ALC)': 'A', '(IIL)': 'I', '(HYP)': 'P', '(CYX)': 'C', '(DBA)': 'D',
    '(DA2)': 'R', '(HIE)': 'H', '(ORN)': 'K', '(DPP)': 'A', '(ABA)': 'C', '(AIB)': 'C'
}


def extract_sequence(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name in protein_letters_3to1:
                    seq.append(protein_letters_3to1[res_name])
                else:
                    converted = AA_CONVERT.get(f"({res_name})", "")
                    if converted:
                        seq.append(converted)

    return "".join(seq)


def process_pdb(input_folder, output_csv):
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ID", "Sequence"])

        for pdb_file in pdb_files:
            try:
                pdb_path = os.path.join(input_folder, pdb_file)
                sequence = extract_sequence(pdb_path)
                writer.writerow([pdb_file, sequence])

            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")


# 设置输入文件夹和输出文件
input_folder = "pdb_folder"
output_csv = "Sequence.csv"

# 处理 PDB 文件
process_pdb(input_folder, output_csv)