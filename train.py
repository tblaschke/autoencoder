# coding=utf-8
import argparse
import gc
import os.path
import signal
import sys
import time
from threading import Thread

import torch
from torch.utils.data import DataLoader

from src.adversarial import AdversarialAE, AdversarialAEUni
from src.bombarelli import BombarelliAE
from src.datareader import SMILESReader, OnehotDecoder, DRD2Reader, MergedDataset
from src.noteacher import NoTeacherAE


def save(autoencoder, folder="./", filename=None, bestfilename=None):
    autoencoder.save(folder + autoencoder.default_file) if filename is None else autoencoder.save(filename)
    if autoencoder.is_best:
        bestfilename = folder + "best_" + autoencoder.default_file if bestfilename is None else bestfilename
        autoencoder.save(bestfilename)
    if (autoencoder.epoch + 1) % 10 == 0:
        name = folder + autoencoder.default_file if filename is None else filename
        name = name + "_epoch{}".format(autoencoder.epoch)
        autoencoder.save(name)


def load_autoencoder(args, explicit_file=None):
    if torch.cuda.is_available() and not args.nocuda:
        use_cuda = True
    else:
        use_cuda = False

    if args.adv:
        print("Use Adversarial AE")
        autoencoder = AdversarialAE(latentdim=args.dims, temperature=args.temperature)
    elif args.advuni:
        print("Use Adversarial AE Uniform")
        autoencoder = AdversarialAEUni(latentdim=args.dims, temperature=args.temperature)
    elif args.bombarelli:
        print("Use Bombarelli AE")
        autoencoder = BombarelliAE(latentdim=args.dims, temperature=args.temperature)
    elif args.noteacher:
        print("Use NoTeacher AE")
        autoencoder = NoTeacherAE(latentdim=args.dims, temperature=args.temperature)
    elif args.professor:
        print("Use Professor AE")
    else:
        raise NotImplementedError("Unknown autoencoder")

    if explicit_file is not None:
        print('Load ' + explicit_file)
        if not use_cuda:
            map_loc = lambda storage, loc: storage
        else:
            map_loc = None
        autoencoder.load(explicit_file, map_location=map_loc)
    else:
        if os.path.isfile(os.path.join(args.save_folder, autoencoder.default_file)):
            print('Continuing from previous checkpoint...')
            if not use_cuda:  # We have to map every saved gpu location to a cpu location in order to continue from a gpu trained model
                map_loc = lambda storage, loc: storage
            else:
                map_loc = None
            autoencoder.load(os.path.join(args.save_folder + autoencoder.default_file), map_location=map_loc)
        else:
            print("Can't find {}".format(os.path.join(args.save_folder, autoencoder.default_file)))
    if use_cuda:
        print("Activate cuda")
        autoencoder.cuda()
    else:
        autoencoder.cpu()

    if hasattr(autoencoder, "maxvaescale"):
        autoencoder.maxvaescale = args.maxvaescale

    return autoencoder


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bombarelli", action="store_true", default=False)
    parser.add_argument("--adv", action="store_true", default=False)
    parser.add_argument("--advuni", action="store_true", default=False)
    parser.add_argument("--noteacher", action="store_true", default=False)
    parser.add_argument("--batch_size", "-b", type=int, metavar='N', default=500)
    parser.add_argument("--epochs", "-e", type=int, default=300)
    parser.add_argument("--save_folder", "-s", type=str, default="./")
    parser.add_argument("--log_folder", "-l", type=str, default=None)
    parser.add_argument("--verbose", "-V", action="store_true", default=False)
    parser.add_argument("--nocuda", "-cpu", action="store_true", default=False)
    parser.add_argument("--nocudnn", action="store_true", default=False)
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--maxvaescale", "-v", type=float, default=0.1)
    parser.add_argument("--dims", "-d", type=int, default=56)
    parser.add_argument("--nolog", action='store_true', default=False)
    parser.add_argument("--nocelecoxib", action='store_true', default=False)
    parser.add_argument("--seed", type=int, metavar='N', default=None)
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print('Got unknown arguments: {}'.format(unknown))
    torch.set_num_threads(torch.get_num_threads())
    if torch.cuda.is_available() and not args.nocuda:
        use_cuda = True
        if args.nocudnn:
            print("Disable cuDNN")
            torch.backends.cudnn.enabled = False
    else:
        use_cuda = False

    return args, use_cuda


if __name__ == "__main__":
    args, use_cuda = getargs()
    dataset_path = os.path.join(os.path.dirname(__file__), "data/prior_trainingset_DRD2_actives_removed.smi")
    if args.nocelecoxib:
        dataset_path = os.path.join(os.path.dirname(__file__), "data/trainingset_DRD2_actives_removed_no_celecoxib.smi")
    labeledtraindataset_path = os.path.join(os.path.dirname(__file__), "data/DRD2_train.smi")
    labeledvaldataset_path = os.path.join(os.path.dirname(__file__), "data/DRD2_validation.smi")
    alphabet_path = os.path.join(os.path.dirname(__file__), "data/alphabet.json")

    train = SMILESReader(dataset_path, alphabet_path, subset=(0, 1200001))
    val = SMILESReader(dataset_path, alphabet_path, subset=(1200001, None))
    if args.advcats:
        label_train = DRD2Reader(labeledtraindataset_path, alphabet_path)
        label_val = DRD2Reader(labeledvaldataset_path, alphabet_path)
        train = MergedDataset(train, label_train)
        val = MergedDataset(val, label_val)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=use_cuda)
    autoencoder = load_autoencoder(args)


    def sigint_handler(signal, frame):
        try:
            autoencoder.save(args.save_folder + "interrupt." + autoencoder.default_file)
        except RuntimeError as e:
            if e.args[0].startswith('cuda runtime error (3) : initialization error'):
                # we are comming from a worker_loop just ignore
                sys.exit(1)
            else:
                print(e)
        else:
            print('Got Interrupt. Exit.')
            sys.exit(1)


    signal.signal(signal.SIGINT, handler=sigint_handler)
    signal.signal(signal.SIGTERM, handler=sigint_handler)

    onehot_decoder = OnehotDecoder(alphabet_path)

    if args.log_folder is None:
        # log in the savefolder
        autoencoder.setlogdir(args.save_folder + autoencoder.getlogdir())
    else:
        autoencoder.setlogdir(args.log_folder + autoencoder.getlogdir())

    for epoch in range(autoencoder.epoch, args.epochs):
        autoencoder.eval()
        x, y = train[0]
        print("Guess molecule %s" % onehot_decoder.decode(x))
        x = x[None, :, :]
        try:
            def sample():
                z = autoencoder.encode(x)
                sampled = autoencoder.decode(z, groundTruth=None, temperature=args.temperature)
                print("free  molecule %s" % onehot_decoder.decode(sampled.data))
                sampled = autoencoder.decode(z, groundTruth=x, temperature=args.temperature)
                print("force molecule %s" % onehot_decoder.decode(sampled.data))


            sample()
            gc.collect()


            def trainining():
                autoencoder.train_model(train_loader, print_every=50, log_scalar_every=50, log_grad_every=1000000,
                                        print_mem=args.verbose, nolog=args.nolog)


            trainining()
            gc.collect()
        except RuntimeError as e:
            if e.args[0].startswith('cuda runtime error (2) : out of memory'):
                print("Not enough GPU memory. Decrease batch_size or buy a new GPU. (Tesla P100 would be cool). Abort.")
                sys.exit(1)
            else:
                print(e)
                sys.exit(1)
        print("Start Validation")
        # since the validation has to unroll the rnn's it need more memory (with cudnn). to reach here training worked before,
        # so maybe we just have some other model running on th gpu as well,
        # just wait in total for 25min. If this doesn't help then save and exit
        max_retry = 25
        wait_time = 60
        while max_retry >= 0:
            try:
                def validate():
                    autoencoder.validate_model(val_loader, print_every=50, log_scalar_every=50, print_mem=args.verbose,
                                               temperature=args.temperature)


                validate()
                gc.collect()
                max_retry = 25
                break
            except RuntimeError as e:
                if e.args[0].startswith('cuda runtime error (2) : out of memory'):
                    print("Not enough memory. Wait for {}s. max_retry:{}".format(wait_time, max_retry))
                    max_retry -= 1
                    time.sleep(wait_time)
                else:
                    print(e)
                    autoencoder.save(args.save_folder + "unknown_error." + autoencoder.default_file)
                    sys.exit(1)

        if max_retry < 0:  # only happens if we can not run validation due to out of memory
            autoencoder.save(args.save_folder + "oom_error." + autoencoder.default_file)
            sys.exit(1)

        for scheduler in autoencoder.scheduler:
            scheduler.step(autoencoder.best_loss, epoch)

        # Put the saving in a extra thread such it's interrupt save
        if not autoencoder.nangrad or autoencoder.nanout:
            a = Thread(target=save, args=[autoencoder, args.save_folder])
            a.start()
            a.join()
