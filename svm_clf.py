# coding=utf-8

import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score


def fingerprints_from_mols(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True) for mol in mols]
    return fps


def create_model(c=128, gamma=0.015625):
    clf = svm.SVC(probability=True, kernel='rbf', C=c, gamma=gamma)
    X, y, mols = get_data(purpose="train")
    print('Data preprocessed, fitting model...')
    clf.fit(X, y)
    with open(os.path.join(os.path.dirname(__file__), "clf_FCFP.pkl"), "wb") as f:
        pickle.dump(clf, f)
    train_probas = clf.predict_proba(X)
    train_roc_auc = roc_auc_score(y, train_probas[:, 1])
    print('Classifier created, training data accuracy: {:.3f}'.format(clf.score(X, y)))
    print('Classifier created, training data ROC AUC: {:.3f}'.format(train_roc_auc))


def validate_model():
    with open(os.path.join(os.path.dirname(__file__), "data/clf_FCFP.pkl"), "rb") as f:
        clf = pickle.load(f)
    print('Model restored...')
    X, y, mols = get_data(purpose="val")
    X_train, y_train, mols = get_data(purpose="train")
    print('Data preprocessed, predicting labels...')
    probas = clf.predict_proba(X)
    train_probas = clf.predict_proba(X_train)
    roc_auc = roc_auc_score(y, probas[:, 1])
    train_roc_auc = roc_auc_score(y_train, train_probas[:, 1])
    score = clf.score(X, y)
    training_score = clf.score(X_train, y_train)
    print('Training accuracy: {:.3f}'.format(training_score))
    print('Training ROC AUC: {:.3f}'.format(train_roc_auc))
    print('Validation accuracy: {:.3f}'.format(score))
    print('Validation ROC AUC: {:.3f}'.format(roc_auc))
    fpr, tpr, thresholds = roc_curve(y, probas[:, 1])

    plt.plot(fpr, tpr, alpha=0.7)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.title('ROC curve', y=1.05)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def validate_model_confusion_matrix():
    with open(os.path.join(os.path.dirname(__file__), "data/clf_FCFP.pkl"), "rb") as f:
        clf = pickle.load(f)
    print('Model restored...')
    X, y, mols = get_data(purpose="val")
    print('Data preprocessed, predicting labels...')
    preds = clf.predict(X)
    tp, tn, fp, fn = tp_tn_fp_fn(y, preds)
    print("Validation:\n")
    print("True positives: {}\n".format(tp))
    print("True negatives: {}\n".format(tn))
    print("False positives: {}\n".format(fp))
    print("False negatives: {}\n".format(fn))
    print("\n")


def tp_tn_fp_fn(y, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for true, pred in zip(y, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    return tp, tn, fp, fn


def get_data(purpose):
    if purpose == "train":
        data_path = os.path.join(os.path.dirname(__file__), "data/DRD2_train.smi_pkl")
    elif purpose == "val":
        data_path = os.path.join(os.path.dirname(__file__), "data/DRD2_validation.smi_pkl")
    elif purpose == "test":
        data_path = os.path.join(os.path.dirname(__file__), "data/DRD2_test.smi_pkl")
    else:
        print("Purpose must be either 'train', 'validation', or 'test'")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    X, y, mols = data
    return X, y, mols


def restore_model():
    with open(os.path.join(os.path.dirname(__file__), "data/clf_ECFP.pkl"), "rb") as f:
        clf = pickle.load(f)
    return clf


def test_parameters(kernel, C, gamma, save=True):
    print("test_parameters for Kernel: {}, C: {}, Gamma:{}".format(kernel, C, gamma))

    def create(kernel, C, gamma):
        print("Fitting model...")
        clf = svm.SVC(probability=True, kernel=kernel, C=C, gamma=gamma)
        X, y, mols = get_data(purpose="train")
        clf.fit(X, y)
        return clf

    def validate(clf, kernel, C, gamma):
        print("Validating model...")
        X, y, mols = get_data(purpose="test")
        X_train, y_train, mols = get_data(purpose="train")
        probas = clf.predict_proba(X)
        train_probas = clf.predict_proba(X_train)
        roc_auc = roc_auc_score(y, probas[:, 1])
        train_roc_auc = roc_auc_score(y_train, train_probas[:, 1])
        score = clf.score(X, y)
        training_score = clf.score(X_train, y_train)
        print("Kernel {} C {} gamma {}".format(kernel, C, gamma))
        print('Training accuracy: {:.3f}'.format(training_score))
        print('Training ROC AUC: {:.3f}'.format(train_roc_auc))
        print('Test accuracy: {:.3f}'.format(score))
        print('Test ROC AUC: {:.3f}'.format(roc_auc))
        return roc_auc

    clf = create(kernel, C, gamma)
    if save:
        with open(os.path.join(os.path.dirname(__file__), "data/clf_FCFP_c_{}_gamma_{}.pkl".format(C, gamma)),
                  "wb") as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    roc_auc = validate(clf, kernel, C, gamma)
    return roc_auc


def data_pkl_from_file(fname):
    import os
    if os.path.exists(fname + '_pkl'):
        print(fname + '_pkl already exists.')
        return
    pos_mols = []
    neg_mols = []
    with open(fname, "r") as f:
        next(f)
        for line in f:
            fields = line.split()
            mol = Chem.MolFromSmiles(fields[0])
            act = fields[1]
            try:
                if mol is not None:
                    if act == "A":
                        pos_mols.append(mol)
                    elif act == "N":
                        neg_mols.append(mol)
                    else:
                        print("No A or N value")
            except ValueError:
                if mol is not None:
                    neg_mols.append(mol)

    data = [[mol, 1] for mol in pos_mols] + [[mol, 0] for mol in neg_mols]
    print(("Number of datapoints: ", len(data)))
    print(("Number of pos: ", len(pos_mols)))
    print(("Number of neg: ", len(neg_mols)))
    print(("Fraction of pos: ", float(len(pos_mols)) / float(len(data))))

    random.shuffle(data)

    X, y = [mol[0] for mol in data], [mol[1] for mol in data]
    mols = X
    X = fingerprints_from_mols(X)
    with open(fname + '_pkl', "wb") as f:
        data = X, y, mols
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def test_param_wrapper(arglist):
    kernel, C, gamma = arglist
    return test_parameters(kernel, C, gamma, save=True)


def grid_search(pool):
    arglist = []
    for c in range(2, 10):
        for gamma in range(-9, 0):
            args = ('rbf', np.power(2.0, int(c)), np.power(2.0, int(gamma)))
            arglist.append(args)
    with pool:
        results = list(pool.map(test_param_wrapper, arglist))
    for idx, params in enumerate(arglist):
        kernel, c, gamma = params
        print("Kernel: {}, C: {}, Gamma:{}, ROCAUC:{}".format(kernel, c, gamma, results[idx]))


print("Hello World!")
if __name__ == "__main__":
    import schwimmbad
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Hello from rank {} / maxsize {}".format(rank, size))

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Schwimmbad example.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    #    data_pkl_from_file('data/DRD2_train.smi')
    #    data_pkl_from_file('data/DRD2_validation.smi')
    #    data_pkl_from_file('data/DRD2_test.smi')
    grid_search(pool)
#    create_model()
#    validate_model()
#    validate_model_confusion_matrix()
