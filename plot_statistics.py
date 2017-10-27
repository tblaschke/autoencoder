# coding=utf-8
import os
import random

import numpy as np
import torch
from plot_sample_celecoxib import load_all_autoencoder
from rdkit import Chem
from rdkit import RDLogger
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.datareader import SMILESReader, OnehotDecoder, OnehotEncoder
from src.utils import acc

if (__name__ == "__main__"):
    autoencoders, autoencodernames, args, use_cuda = load_all_autoencoder()

    alphabet_path = os.path.join(os.path.dirname(__file__), "data/alphabet.json")
    onehot_decoder = OnehotDecoder(alphabet_path)
    onehot_encoder = OnehotEncoder(alphabet_path)
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if use_cuda:
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

    dataset_path = os.path.join(os.path.dirname(__file__), "data/prior_trainingset_DRD2_actives_removed.smi")

    train = SMILESReader(dataset_path, alphabet_path, subset=(0, 1200001))
    val = SMILESReader(dataset_path, alphabet_path, subset=(1200001, None))

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=use_cuda)

    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    for i in range(len(autoencoders)):
        autoencoder = autoencoders[i]
        print("Evaluate {}\n".format(autoencodernames[i]))
        autoencoder.eval()
        train_ac = 0
        val_ac = 0
        val_smi = 0
        for t, (x, y) in enumerate(val_loader):
            x_var = Variable(x.type(dtype), requires_grad=False, volatile=True)
            z_var = autoencoder.encoder(x_var)
            second_last = autoencoder.decoder.get_before_lastLayerOutput(z_var).data
            # we sample each point 500 times to get a number of valid SMILES
            valid = 0
            invalid = 0
            print('\rt = %d/%d' % (t + 1, len(val_loader)))
            for i in range(500):
                tmp = Variable(second_last.type(dtype), requires_grad=False, volatile=True)
                log_probs, samples = autoencoder.decoder.get_lastLayerOutput(tmp, groundTruth=None,
                                                                             temperature=args.temperature)
                smiles = onehot_decoder.decode(samples.data)
                for s in smiles:
                    if Chem.MolFromSmiles(s):
                        valid += 1
                    else:
                        invalid += 1
            ac = acc(log_probs.data, x)
            valid_frac = valid / (valid + invalid)
            val_ac += ac
            val_smi += valid_frac
        val_ac /= t
        val_smi /= t
        print("Validation set character reconstruction accuracy: {}".format(val_ac))
        print("Validation set fraction of valid SMILES: {}".format(val_smi))
        for t, (x, y) in enumerate(train_loader):
            print('\rt = %d/%d' % (t + 1, len(train_loader)))
            x_var = Variable(x.type(dtype), requires_grad=False, volatile=True)
            z_var = autoencoder.encoder(x_var)
            log_probs, samples = autoencoder.decoder(z_var, groundTruth=x_var, temperature=args.temperature)
            ac = acc(log_probs.data, x)
            train_ac += ac
        train_ac /= t
        print("Training set character reconstruction accuracy: {}".format(train_ac))
