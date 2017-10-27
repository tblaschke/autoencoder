# coding=utf-8
import os
from random import randint

import rdkit.Chem as Chem
import rdkit.RDLogger as RDLogger
import torch
from torch.utils.data import DataLoader

from src.datareader import SMILESReader, OnehotDecoder
from train import getargs, load_autoencoder

if __name__ == "__main__":
    args, use_cuda = getargs()

    dataset_path = os.path.join(os.path.dirname(__file__), "data/prior_trainingset_DRD2_actives_removed.smi")
    alphabet_path = os.path.join(os.path.dirname(__file__), "data/alphabet.json")

    train = SMILESReader(dataset_path, alphabet_path, subset=(0, 1200001))
    val = SMILESReader(dataset_path, alphabet_path, subset=(1200001, None))

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=use_cuda)

    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=use_cuda)

    autoencoder = load_autoencoder(args)

    onehot_decoder = OnehotDecoder(alphabet_path)

    batches = args.batch_size
    repeat = 100
    autoencoder.eval()
    x, y = val[randint(0, 1000)]
    startsmi = onehot_decoder.decode(x)[0]
    x = train.onehotencoder(startsmi)
    x = x[None, :, :]
    encoded_smi = autoencoder.encode(x)
    encoded_noise = torch.randn(encoded_smi.size())


    def validate_point(encoded, compare_with=None, groundTruth=None, print_all=False):
        encoded_stack = [encoded.squeeze() for i in range(batches)]
        encoded_stack = torch.stack(encoded_stack)
        if groundTruth is not None:
            groundTruth_stack = [groundTruth.squeeze() for i in range(batches)]
            groundTruth_stack = torch.stack(groundTruth_stack)
        invalid = 0
        same = 0
        mol_set = []
        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)
        for i in range(repeat):
            sampled = autoencoder.decode(encoded_stack) if groundTruth is None else autoencoder.decode(encoded_stack,
                                                                                                       groundTruth_stack)
            smi = onehot_decoder.decode(sampled.data)
            # if groundTruth is not None:
            #    print(smi)
            mols = [Chem.MolFromSmiles(s) for s in smi]
            if compare_with is not None:

                compare_with = Chem.MolToSmiles(Chem.MolFromSmiles(compare_with))
                for i, mol in enumerate(mols):
                    if mol is not None:
                        s = Chem.MolToSmiles(mol)
                        if print_all:
                            print(smi[i])
                        if s == compare_with:
                            same += 1
            else:
                for i, mol in enumerate(mols):
                    if mol is not None:
                        s = Chem.MolToSmiles(mol)
                        mol_set.append(s)
            for mol in mols:
                if mol is None:
                    invalid += 1
        if compare_with is None:
            return invalid, set(mol_set)
        else:
            return invalid, same


    print("Guess molecule with GroundTruth %s" % startsmi)
    invalid, same = validate_point(encoded_smi, startsmi, groundTruth=train.onehotencoder(startsmi), print_all=False)
    print("{} (out of {}) invalid ".format(invalid, batches * repeat))
    print("{} (out of {}) reconstructed)".format(same, (batches * repeat) - invalid))

    print("Guess molecule in free mode %s" % startsmi)
    invalid, same = validate_point(encoded_smi, startsmi)
    print("{} (out of {}) invalid)".format(invalid, batches * repeat))
    print("{} (out of {}) reconstructed)".format(same, (batches * repeat) - invalid))

    print("Decode noise")
    invalid, molset = validate_point(encoded_noise)
    print("{} (out of {}) invalid)".format(invalid, batches * repeat))
    print(molset)
