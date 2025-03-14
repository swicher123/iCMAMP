import os
import csv
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1

AA_CONVERT = {
    '(AAB)': 'A', '(NLE)': 'L', '(DIF)': 'F', '(NMW)': 'W', '(NMS)': 'S', '(NMA)': 'A',
    '(NVA)': 'V', '(ALC)': 'A', '(IIL)': 'I', '(HYP)': 'P', '(CYX)': 'C', '(DBA)': 'D',
    '(DA2)': 'R', '(HIE)': 'H', '(ORN)': 'K', '(DPP)': 'A', '(ABA)': 'C', '(AIB)': 'C'
}

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {AA[i]: i for i in range(len(AA))}
DIPEPTIDES = [aa1 + aa2 for aa1 in AA for aa2 in AA]


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
                    seq.append(AA_CONVERT.get(f"({res_name})", ""))

    return "".join(seq)


def compute_dpc(sequence):
    dpc_vector = np.zeros(400, dtype=int)

    for j in range(len(sequence) - 1):
        if sequence[j] in AA_INDEX and sequence[j + 1] in AA_INDEX:
            idx = AA_INDEX[sequence[j]] * 20 + AA_INDEX[sequence[j + 1]]
            dpc_vector[idx] += 1

    total = np.sum(dpc_vector)
    if total > 0:
        dpc_vector = dpc_vector / total * 100

    return dpc_vector.tolist()


def process_pdb_and_compute_dpc(input_folder, output_csv):
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ID"] + DIPEPTIDES)

        for pdb_file in pdb_files:
            try:
                pdb_path = os.path.join(input_folder, pdb_file)
                sequence = extract_sequence(pdb_path)
                dpc_features = compute_dpc(sequence)
                writer.writerow([pdb_file] + dpc_features)
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
    # print(f"DPC 特征已保存至: {output_csv}")
    print(f"DPC traits saved to: {output_csv}")

if __name__ == "__main__":
    input_folder = "pdb_test"
    output_csv = "sequence_test/DPC_feature_1.csv"
    process_pdb_and_compute_dpc(input_folder, output_csv)
