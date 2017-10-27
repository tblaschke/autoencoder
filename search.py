# -*- coding: utf-8 -*-
import os.path
import pickle
import random

import GPyOpt
import numpy as np
import torch
from torch.multiprocessing import Queue, Process

from src.datareader import DRD2Reader, OnehotDecoder, OnehotEncoder
from train import getargs, load_autoencoder


def create_plot(queue, steps=510, filename="plot.csv"):
    from src.DynamicPlot import DynamicPlot
    myplot = DynamicPlot(steps, 0, 1, filename)
    while True:
        msg = queue.get()  # Read from the queue and do nothing
        if (msg == 'DONE'):
            break
        else:
            score = msg[0]
            mols = msg[1]
            # mols = []
            myplot.update(score, mols)


def plot_data(queue, score, mols):
    queue.put([score, mols])


def create_mollogger(queue, train, test, val):
    import rdkit.Chem as Chem
    import rdkit.RDLogger as RDLogger
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    train_bin = {}
    test_bin = {}
    val_bin = {}
    for s in train:
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        train_bin[mol] = []
    for s in test:
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        test_bin[mol] = []
    for s in val:
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        val_bin[mol] = []

    nb_messages = 0
    while True:
        msg = queue.get()
        if (msg == 'DONE'):
            break
        else:
            smis, point = msg
            nb_messages += 1
            mols = [Chem.MolFromSmiles(s) for s in smis]
            mols_bins = [Chem.MolToSmiles(m) for m in mols if m is not None]
            for mol in mols_bins:
                if mol in train_bin:
                    print("Found mol in training set! Iteration: {}".format(nb_messages))
                if mol in test_bin:
                    print("Found mol in training set! Iteration: {}".format(nb_messages))
                if mol in val_bin:
                    print("Found mol in training set! Iteration: {}".format(nb_messages))


def log_mols(queue, mols, point):
    queue.put([mols, point])


if (__name__ == "__main__"):
    args, use_cuda = getargs()
    if args.seed:
        seed = args.seed
    else:
        seed = random.randint(0, 9999999)
    print("Seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("No CUDA")
    filename = os.path.join(os.path.dirname(__file__), "search_tmp/plot_drd2_search_ecfp6_{}.csv".format(seed))
    pklfilename = os.path.join(os.path.dirname(__file__), "search_tmp/plot_drd2_search_ecfp6_{}.pkl".format(seed))

    queue = Queue()
    maxiter = 500
    nb_init = 100
    plot_p = Process(target=create_plot, args=((queue), maxiter + nb_init, filename))
    plot_p.daemon = True
    plot_p.start()

    autoencoder = load_autoencoder(
        args)  # during the loading of the autoencoder model we continue from the last rng state
    # so to make the following experiment reproducible we have to use the given random seed

    train_path = os.path.join(os.path.dirname(__file__), "data/DRD2_train.smi")
    test_path = os.path.join(os.path.dirname(__file__), "data/DRD2_test.smi")
    val_path = os.path.join(os.path.dirname(__file__), "data/DRD2_validation.smi")
    alphabet_path = os.path.join(os.path.dirname(__file__), "data/alphabet.json")
    with open(os.path.join(os.path.dirname(__file__), "data/clf.pkl"), 'rb') as fd:
        classfier = pickle.load(fd, encoding="latin1")
    onehot_decoder = OnehotDecoder(alphabet_path)
    onehot_encoder = OnehotEncoder(alphabet_path)
    batches = args.batch_size
    train = DRD2Reader(train_path, alphabet_path)
    test = DRD2Reader(test_path, alphabet_path)
    val = DRD2Reader(val_path, alphabet_path)

    smi1 = train.smiles[:4526]
    smi2 = test.smiles[:1405]
    smi3 = val.smiles[:1287]
    logqueue = Queue()
    log_p = Process(target=create_mollogger, args=((logqueue), smi1, smi2, smi3))
    log_p.daemon = True
    log_p.start()

    encoded = autoencoder.encode(train[0][0].unsqueeze(0))

    import rdkit.Chem as Chem
    import rdkit.RDLogger as RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)


    def decode(point, batch=500, n=1):
        shape = point.shape
        point = np.repeat(point, batch)
        point = point.reshape(shape + (batch,)).transpose()
        for i in range(n):
            decoded_list = [onehot_decoder.decode(autoencoder.decode(point).data) for i in range(n)]
        mols = [Chem.MolFromSmiles(d) for decoded in decoded_list for d in decoded]
        mols = [m for m in mols if m is not None]
        return mols  # , smis


    def active_mol(mols, classfier=classfier):
        if len(mols) == 0:
            # return a random number close to 1
            return 1 - random.random() * 1e-2, [], [], 0
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, useFeatures=False) for mol in mols]
        probs = classfier.predict_proba(fingerprints)
        inactivity = probs[:, 0]
        if any(inactivity < 0.5):
            score = 0
            nb_actives = 0
            for i in range(len(fingerprints)):
                if inactivity[i] < 0.5:
                    score += inactivity[i]
                    nb_actives += 1
            score /= nb_actives
        else:
            score = sum(inactivity) / len(inactivity)
        if len(fingerprints) > 1:
            similarities = []
            for i in range(len(fingerprints)):
                for j in range(i):
                    if i != j:
                        similarities.append(DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]))
            sim = sum(similarities) / len(similarities)
        else:
            sim = 0
        return score, fingerprints, probs, sim


    def filter_garbage(mols):
        filtered = []
        for mol in mols:
            nMacrocycles = 0
            ssr = Chem.GetSymmSSSR(mol)
            for i in range(len(ssr)):
                if len(list(ssr[i])) >= 8:
                    nMacrocycles += 1
            if nMacrocycles < 1:
                filtered.append(mol)
        return filtered


    def search_drd2(latent_point):
        latent_point = latent_point.flatten()
        mols = decode(latent_point, n=1)
        mols = filter_garbage(mols)
        score, fingerprints, probs, similarity = active_mol(mols, classfier=classfier)
        active = []
        avg_active_prob = 0
        for i, mol in enumerate(mols):
            try:
                if probs[i, 0] < 0.5:
                    active.append(Chem.MolToSmiles(mol))
                    avg_active_prob += probs[i, 1]
            except ValueError:
                pass
        active_tc = 0
        if len(active) > 0:
            avg_active_prob /= len(active)
            if len(active) > 1:
                fp = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=False) for smi in active]
                compared = 0
                for i in range(len(fp)):
                    for j in range(i):
                        if i != j:
                            active_tc += DataStructs.TanimotoSimilarity(fp[i], fp[j])
                            compared += 1
                active_tc /= compared
        if args.verbose:
            print("Got {} mols. Avg TC {}. {} active. Score {}. Avg aTC {}".format(len(mols), similarity, len(active),
                                                                                   1 - score, active_tc))
        plot_data(queue=queue, score=score, mols=active)
        log_mols(logqueue, mols=active, point=latent_point)
        return score


    bounds = [
        {'name': 'latent_space', 'type': 'continuous', 'domain': (-2, 2), 'dimensionality': encoded.size()[-1]}]
    myBopt = GPyOpt.methods.BayesianOptimization(f=search_drd2,  # function to optimize
                                                 domain=bounds,  # box-constrains of the problem
                                                 # initial_design_type='latin',
                                                 initial_design_type='random',
                                                 model_type='GP',
                                                 acquisition_type='EI',  # Selects the Expected improvement
                                                 exact_feval=True,
                                                 initial_design_numdata=nb_init,
                                                 evaluator_type='local_penalization',
                                                 normalize_Y=False,
                                                 optimize_restarts=50,
                                                 batch_size=1
                                                 )

    myBopt.run_optimization(max_iter=maxiter, verbosity=args.verbose)
    best_value = 1 - myBopt.fx_opt[0]
    best_point = myBopt.x_opt

    with open(pklfilename, 'wb') as fd:
        pickle.dump(myBopt, fd, protocol=pickle.HIGHEST_PROTOCOL)

    print(best_value)
    mols = decode(best_point, 500, 1)
    smis = [Chem.MolToSmiles(mol) for mol in mols]
    active = []
    print(smis)

    queue.put("DONE")
