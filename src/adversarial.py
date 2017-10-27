# coding=utf-8
import datetime
from itertools import chain
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.autograd import Variable

from src.layers import ConvSELU, SELU
from .bombarelli import MolDecoder
from .logger import Logger, to_np
from .utils import get_memusage, ReduceLROnPlateau, register_nan_checks, acc, reset


class MolEncoder(nn.Module):
    def __init__(self, i=120, o=292, c=34):
        super(MolEncoder, self).__init__()

        self.conv1 = ConvSELU(i, 9, kernel_size=9)
        self.conv2 = ConvSELU(9, 9, kernel_size=9)
        self.conv3 = ConvSELU(9, 10, kernel_size=11)
        self.dense1 = nn.Sequential(nn.Linear(self._conv_output(i, c), 435),
                                    SELU())
        self.dense2 = nn.Sequential(nn.Linear(435, o),
                                    nn.Hardtanh(min_value=-2, max_value=2, inplace=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

    def _conv_output(self, i, c):
        tmp = Variable(torch.zeros((1, i, c)), requires_grad=False, volatile=True)
        out = self.conv1(tmp)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        size = out.size()[-1]
        return size


class Discriminator_gauss(nn.Module):
    def __init__(self, i=292):
        super(Discriminator_gauss, self).__init__()
        self.lin1 = nn.Linear(i, 1000)
        self.lin2 = nn.Sequential(nn.Linear(1000, 1000),
                                  SELU())
        self.lin3 = nn.Sequential(nn.Linear(1000, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class AdversarialAE(nn.Module):
    def __init__(self, maxlen=120, latentdim=56, alphabetlength=35, goindex=1, temperature=0.5,
                 distribution=torch.randn):
        super(AdversarialAE, self).__init__()
        self.encoder = MolEncoder(maxlen, latentdim, alphabetlength)
        self.decoder = MolDecoder(latentdim, maxlen, hidden=501, c=alphabetlength, goindex=goindex,
                                  temperature=temperature)
        self.discriminator = Discriminator_gauss(latentdim)
        self.distribution = distribution
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.00031, betas=(0.937, 0.999), eps=1e-8,
                                                  weight_decay=0.0)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.00031, betas=(0.937, 0.999), eps=1e-8,
                                                  weight_decay=0.0)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002,
                                                        betas=(0.937, 0.999), eps=1e-8, weight_decay=0.0)
        self.loss = None
        self.best_loss = 1E6
        self.epoch = -1
        self.is_best = False
        self.default_file = 'adv_checkpoint.pth.tar'
        self.uselogprop = False
        self.nangrad = False
        self.nanout = False
        self.temperature = temperature
        self.scheduler = [ReduceLROnPlateau(self.encoder_optimizer, mode='min', min_lr=1E-8),
                          ReduceLROnPlateau(self.decoder_optimizer, mode='min', min_lr=1E-8),
                          ReduceLROnPlateau(self.discriminator_optimizer, mode='min', min_lr=1E-9)]
        self.logdir = "/logs/" + self.__class__.__name__
        reset(self)
        register_nan_checks(self, grad=False, output=False)

    def load(self, filename=None, map_location=lambda storage, loc: storage):
        filename = self.default_file if filename is None else filename
        checkpoint = torch.load(filename, map_location=map_location)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.epoch = checkpoint['epoch']
        self.temperature = checkpoint['temperature']

    def save(self, filename=None):
        filename = self.default_file if filename is None else filename
        torch.save({
            'epoch': self.epoch,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'temperature': self.temperature,
        }, filename)

    def setuplogger(self, nolog=False):
        self.logger = Logger(log_dir=self.logdir, nolog=nolog)

    def setlogdir(self, log_dir):
        self.logdir = log_dir

    def getlogdir(self):
        return self.logdir

    def forward(self, x, onlySamples=False):
        dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        x_var = Variable(x.type(dtype))
        y_var = self.encoder(x_var)
        if self.training:
            z_var, samples = self.decoder(y_var, groundTruth=x_var)
        else:
            z_var, samples = self.decoder(y_var, groundTruth=None)
        return z_var, samples

    def train_model(self, train_loader, print_every=50, log_scalar_every=50, log_grad_every=1000, print_mem=False,
                    temperature=None, nolog=False):
        temperature = self.temperature if temperature is None else temperature
        self.train()
        self.epoch += 1
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        if hasattr(self, 'loss'):
            del self.loss  # delete previous loss e.g. Validation
        self.is_best = False
        self.setuplogger(nolog=nolog)
        print('Epoch {}:'.format(self.epoch))
        start = timer()
        for t, (x, y) in enumerate(train_loader):
            dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor

            #### Optimize reconstruction
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            x_var = Variable(x.type(dtype))
            y_var = self.encoder(x_var)
            z_var, samples = self.decoder(y_var, groundTruth=x_var, temperature=temperature)
            if self.nanout:
                break
            recon_loss = self.decoder.loss(x_var, z_var)
            recon_loss.backward()
            if self.nangrad:
                break

            if (t + 1) % log_grad_every == 0:
                step = (self.epoch * len(train_loader)) + t + 1
                for tag, value in chain(self.encoder.named_parameters(), self.decoder.named_parameters()):
                    if value.grad is not None:
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, to_np(value), step)
                        self.logger.histo_summary(tag + '/grad', to_np(value.grad), step)

            utils.clip_grad_norm(chain(self.encoder.parameters(), self.decoder.parameters()), 2.)  # clip the gradients
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            #### Optimize Discriminator
            bceloss = nn.BCELoss(size_average=True)
            fake_gauss = self.encoder(Variable(x.type(dtype)))
            D_fake_pred = self.discriminator(fake_gauss)
            D_fake_loss = bceloss(D_fake_pred, Variable(torch.zeros(*D_fake_pred.size())).type_as(
                fake_gauss))  # we should have assigned 0 to the fake gaussian
            D_fake_loss.backward()
            D_fake_acc = D_fake_pred < 0.5
            D_fake_acc = torch.mean(D_fake_acc.float())

            real_gauss = Variable(self.distribution(fake_gauss.size())).type_as(fake_gauss)
            D_real_pred = self.discriminator(real_gauss)
            D_real_loss = bceloss(D_real_pred, Variable(torch.ones(*D_real_pred.size())).type_as(
                fake_gauss))  # we should have assigned 1 for the real gaussian
            D_real_loss.backward()
            D_real_acc = D_real_pred > 0.5
            D_real_acc = torch.mean(D_real_acc.float())

            if self.nanout:
                break
            if self.nangrad:
                break

            if (t + 1) % log_grad_every == 0:
                step = (self.epoch * len(train_loader)) + t + 1
                for tag, value in chain(self.discriminator.named_parameters()):
                    if value.grad is not None:
                        tag = "discriminating." + tag
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, to_np(value), step)
                        self.logger.histo_summary(tag + '/grad', to_np(value.grad), step)

            utils.clip_grad_norm(chain(self.encoder.parameters(), self.discriminator.parameters()),
                                 2.)  # clip the gradients
            # we don't need a perfect classifier. But also learn at least a some kind of a gauss predictor
            # if D_fake_acc.data[0] < 0.9 or D_real_acc.data[0] < 0.2:
            self.discriminator_optimizer.step()
            self.encoder_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            ### Optimize Encoder to fool Discriminator
            fake_gauss = self.encoder(Variable(x.type(dtype)))
            D_fool_pred = self.discriminator(fake_gauss)
            D_fool_loss = bceloss(D_fool_pred, Variable(torch.ones(*D_fool_pred.size())).type_as(
                fake_gauss))  # we should have assigned 1 since we want to hide as real gaussian
            D_fool_loss.backward()
            D_fool_acc = D_fool_pred < 0.5
            D_fool_acc = torch.mean(D_fool_acc.float())

            if self.nanout:
                break
            if self.nangrad:
                break

            if (t + 1) % log_grad_every == 0:
                step = (self.epoch * len(train_loader)) + t + 1
                for tag, value in chain(self.encoder.named_parameters()):
                    if value.grad is not None:
                        tag = "fool." + tag
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, to_np(value), step)
                        self.logger.histo_summary(tag + '/grad', to_np(value.grad), step)

            utils.clip_grad_norm(chain(self.encoder.parameters(), self.discriminator.parameters()),
                                 2.)  # clip the gradients
            # don't fool a shitty discriminator. We are playing fair
            if D_real_acc.data[0] > 0.3 and D_fake_acc.data[0] > 0.3:
                self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            if (t + 1) % print_every == 0:
                end = timer()
                t_left = len(train_loader) - t + 1
                average_time = (end - start) / (t + 1)
                eta = str(datetime.timedelta(seconds=round(average_time * t_left)))
                ac = acc(z_var.data, x)
                if print_mem:
                    mem = get_memusage()
                    mem_str = ", used memory {} MB / avail memory {} MB".format(mem['memory.used'], mem['memory.total'])
                else:
                    mem_str = ""
                print('\rt = %d/%d, loss = %.4f, d_real_accuracy = %.4f, d_fake_accuracy = %.4f, acc = %.4f, eta = %s%s'
                      % (t + 1, len(train_loader), recon_loss.data[0], D_real_acc.data[0], D_fake_acc.data[0], ac, eta,
                         mem_str), end="")
            if (t + 1) % log_scalar_every == 0:
                ac = acc(z_var.data, x)
                step = (self.epoch * len(train_loader)) + t + 1
                # (1) Log the scalar values
                info = {
                    'loss': recon_loss.data[0],
                    'd_fake_loss': D_fake_loss.data[0],
                    'd_real_loss': D_real_loss.data[0],
                    'd_fool_loss': D_fool_loss.data[0],
                    'sum_loss': recon_loss.data[0] + D_fake_loss.data[0] + D_real_loss.data[0] + D_fool_loss.data[0],
                    'accuracy': ac,
                    'd_fake_accuracy': D_fake_acc.data[0],
                    'd_real_accuracy': D_real_acc.data[0],
                    'd_fool_accuracy': D_fool_acc.data[0],
                    'epoch': self.epoch
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

        print()

    def validate_model(self, val_loader, print_every=50, log_scalar_every=100, print_mem=False, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        self.eval()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        avg_val_loss = 0.
        start = timer()
        dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        for t, (x, y) in enumerate(val_loader):
            x_var = Variable(x.type(dtype), requires_grad=False)
            y_var = self.encoder(x_var)
            z_var, samples = self.decoder(y_var, temperature=temperature)
            recon_loss = self.decoder.loss(x=x_var, x_decoded=z_var)

            bceloss = nn.BCELoss(size_average=True)
            fake_gauss = self.encoder(x_var)
            D_fake_pred = self.discriminator(fake_gauss)
            D_fake_loss = bceloss(D_fake_pred, Variable(torch.zeros(*D_fake_pred.size())).type_as(
                fake_gauss))  # we should have assigned 0 to the fake gaussian
            D_fake_acc = D_fake_pred < 0.5
            D_fake_acc = torch.mean(D_fake_acc.float())

            real_gauss = Variable(self.distribution(fake_gauss.size())).type_as(fake_gauss)
            D_real_pred = self.discriminator(real_gauss)
            D_real_loss = bceloss(D_real_pred, Variable(torch.ones(*D_real_pred.size())).type_as(
                fake_gauss))  # we should have assigned 1 for the real gaussian
            D_real_acc = D_real_pred > 0.5
            D_real_acc = torch.mean(D_real_acc.float())

            fake_gauss = self.encoder(x_var)
            D_fool_pred = self.discriminator(fake_gauss)
            D_fool_loss = bceloss(D_fool_pred, Variable(torch.ones(*D_fool_pred.size())).type_as(
                fake_gauss))  # we should have assigned 1 since we want to hide as real gaussian
            D_fool_acc = D_fool_pred < 0.5
            D_fool_acc = torch.mean(D_fool_acc.float())

            avg_val_loss += recon_loss.data[0]
            if (t + 1) % print_every == 0:
                end = timer()
                t_left = len(val_loader) - t + 1
                average_time = (end - start) / (t + 1)
                eta = str(datetime.timedelta(seconds=round(average_time * t_left)))
                ac = acc(z_var.data, x)
                if print_mem:
                    mem = get_memusage()
                    mem_str = ", used memory {} MB / avail memory {} MB".format(mem['memory.used'], mem['memory.total'])
                else:
                    mem_str = ""
                print(
                    '\rt = %d/%d, val_loss = %.4f, val_d_real_accuracy = %.4f, val_d_fake_accuracy = %.4f, val_acc = %.4f, eta = %s%s'
                    % (t + 1, len(val_loader), recon_loss.data[0], D_real_acc.data[0], D_fake_acc.data[0], ac, eta,
                       mem_str), end="")
            if (t + 1) % print_every == 0:
                ac = acc(z_var.data, x)
                step = (self.epoch * len(val_loader)) + t + 1
                # (1) Log the scalar values
                info = {
                    'val_loss': recon_loss.data[0],
                    'val_d_fake_loss': D_fake_loss.data[0],
                    'val_d_real_loss': D_real_loss.data[0],
                    'val_d_fool_loss': D_fool_loss.data[0],
                    'val_sum_loss': recon_loss.data[0] + D_fake_loss.data[0] + D_real_loss.data[0] + D_fool_loss.data[
                        0],
                    'val_accuracy': ac,
                    'val_d_fake_accuracy': D_fake_acc.data[0],
                    'val_d_real_accuracy': D_real_acc.data[0],
                    'val_d_fool_accuracy': D_fool_acc.data[0],
                    'val_epoch': self.epoch
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)
        if t > 0:
            avg_val_loss /= t
        print('\naverage validation loss: %.4f' % avg_val_loss)
        if avg_val_loss < self.best_loss:
            self.is_best = True
            self.best_loss = avg_val_loss
        else:
            self.is_best = False
        return avg_val_loss

    def encode(self, onehot: torch.FloatTensor, volatile=True) -> torch.FloatTensor:
        self.eval()
        dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        if onehot.dim() == 2:
            onehot = onehot.unsqueeze_(0)
        x_var = Variable(onehot.type(dtype), requires_grad=False, volatile=volatile)
        y_var = self.encoder(x_var)
        return y_var.data

    def decode(self, latent: torch.FloatTensor, groundTruth: torch.FloatTensor = None, volatile=True,
               temperature=None) -> torch.FloatTensor:
        self.eval()
        dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        if isinstance(latent, np.ndarray):
            latent = dtype(latent)
        if latent.dim() == 1:
            latent = latent.unsqueeze_(0)
        y_var = Variable(latent.type(dtype), requires_grad=False, volatile=volatile)
        g_var = Variable(groundTruth.type(dtype), requires_grad=False,
                         volatile=volatile) if groundTruth is not None else None
        log_probs, samples = self.decoder(y_var, groundTruth=g_var, temperature=temperature)
        return samples


class AdversarialAEUni(AdversarialAE):
    def __init__(self, maxlen=120, latentdim=56, alphabetlength=35, goindex=1, temperature=0.5,
                 distribution=lambda *size: torch.Tensor(*size).uniform_(-2, 2)):
        super(AdversarialAEUni, self).__init__(maxlen=maxlen, latentdim=latentdim, alphabetlength=alphabetlength,
                                               goindex=goindex, temperature=temperature, distribution=distribution)
        self.default_file = 'adv_uni_checkpoint.pth.tar'
