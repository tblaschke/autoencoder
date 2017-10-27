# coding=utf-8
import json
from collections import OrderedDict

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem


def replace_double(smi: str) -> str:
    for s, w in zip(['Br', 'Cl', 'Si'], ["Ö", "Ä", "Å"]):
        smi = smi.replace(s, w)
    return smi


def replace_single(smi: str) -> str:
    for s, w in zip(['Br', 'Cl', 'Si'], ["Ö", "Ä", "Å"]):
        smi = smi.replace(w, s)
    return smi


def sanitize_dataset(input: str, output: str):
    with open("data/alphabet.json", "r") as fd:
        alphabet = json.load(fd)
    mylogger = RDLogger.logger()
    mylogger.setLevel(val=RDLogger.ERROR)

    reader = Chem.SmilesMolSupplier(input, titleLine=False)
    smiles = []
    counter = 0
    single_alphabet = replace_double("".join(alphabet))
    for mol in reader:
        counter += 1
        if (counter % 1000) == 0:
            print("Mol: {}, ({} valid)".format(counter, len(smiles)))
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        single_smi = replace_double(smi)
        if len(single_smi) <= 120:
            if set(single_alphabet).issuperset(set(single_smi)):
                smiles.append(smi)
    unique_smiles = OrderedDict.fromkeys(smiles).keys()
    print("{} valid SMILES. {} unique SMILES".format(len(smiles), len(unique_smiles)))

    with open(output, 'w') as fd:
        fd.writelines(smi + "\n" for smi in unique_smiles)


def remove_celecoxib(input: str, output: str):
    mylogger = RDLogger.logger()
    mylogger.setLevel(val=RDLogger.ERROR)
    reader = Chem.SmilesMolSupplier(input, titleLine=False)

    celecoxib = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"
    cele_mol = Chem.MolFromSmiles(celecoxib)
    cele_finger = AllChem.GetMorganFingerprint(cele_mol, 2, useFeatures=True)
    nb_similar = 0

    with open(output, 'w') as fd:
        for mol in reader:
            fingerprint = AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)
            similarity = DataStructs.TanimotoSimilarity(cele_finger, fingerprint)
            if similarity > 0.5:
                nb_similar += 1
            else:
                fd.writelines(Chem.MolToSmiles(mol, isomericSmiles=False) + "\n")
    print("Filtered {} compounds".format(nb_similar))


def sanitize_drd2_dataset(input: str, output: str):
    with open("data/alphabet.json", "r") as fd:
        alphabet = json.load(fd)
    mylogger = RDLogger.logger()
    mylogger.setLevel(val=RDLogger.ERROR)

    with open(input, "r") as fd:
        lines = fd.readlines()

    splits = [line.split(" ") for line in lines[1:]]
    smiles = []
    cat = []
    id = []
    counter = 0
    single_alphabet = replace_double("".join(alphabet))
    for tripple in splits:
        counter += 1
        if (counter % 1000) == 0:
            print("Mol: {}, ({} valid)".format(counter, len(smiles)))
        mol = Chem.MolFromSmiles(tripple[0])
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        single_smi = replace_double(smi)
        if len(single_smi) <= 120:
            if set(single_alphabet).issuperset(set(single_smi)):
                smiles.append(smi)
                cat.append(tripple[1])
                id.append(tripple[2])
    unique_smiles = OrderedDict.fromkeys(smiles)
    for i, smi in enumerate(smiles):
        unique_smiles[smi] = cat[i].strip() + " " + id[i].strip()
    print("{} valid SMILES. {} unique SMILES".format(len(smiles), len(unique_smiles)))

    with open(output, 'w') as fd:
        fd.writelines(["SMILES ACTIVE CLUSTER\n"])
        fd.writelines(smi + " " + unique_smiles[smi] + "\n" for smi in unique_smiles)


if __name__ == "__main__":
    # sanitize_dataset(input="data/prior_trainingset_DRD2_actives_removed", output='data/prior_trainingset_DRD2_actives_removed.smi')
    remove_celecoxib(input="data/prior_trainingset_DRD2_actives_removed.smi",
                     output='data/trainingset_DRD2_actives_removed_no_celecoxib.smi')
    # sanitize_drd2_dataset(input="data/DRD2_train", output='data/DRD2_train.smi')
    # sanitize_drd2_dataset(input="data/DRD2_test", output='data/DRD2_test.smi')
    # sanitize_drd2_dataset(input="data/DRD2_validation", output='data/DRD2_validation.smi')
