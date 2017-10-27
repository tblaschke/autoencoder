# coding=utf-8
import datetime
from itertools import chain
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.autograd import Variable

from src.layers import Repeat, ConvSELU, VariationalDense
from .layers import SELU, teacherGRU
from .logger import Logger, to_np
from .utils import get_memusage, ReduceLROnPlateau, register_nan_checks, acc, reset, categorical_crossentropy


class MolEncoder(nn.Module):

    def __init__(self, i=120, o=292, c=35, **kwargs):
        super(MolEncoder, self).__init__()
        self.hyperparameter = {
            'input': i,
            'output': o,
            'chars': c,
            'conv1': {'ConvSELU': {'input': i,
                                   'output': 9,
                                   'kernel_size': 9}
                      },
            'conv2': {'ConvSELU': {'input': 9,
                                   'output': 9,
                                   'kernel_size': 9}
                      },
            'conv3': {'ConvSELU': {'input': 9,
                                   'output': 10,
                                   'kernel_size': 11}
                      },
            'dense1': {'output': 435},
            'varationaldense': {"VariationalDense": {'input': 435,
                                                     'output': o,
                                                     'scale': 1E-2}
                                }
        }
        if "param" in kwargs:
            self.hyperparameter.update(kwargs["param"]["MolEncoder"])
        param = self.hyperparameter
        self.conv1 = ConvSELU(param=param["conv1"])
        self.conv2 = ConvSELU(param=param["conv2"])
        self.conv3 = ConvSELU(param=param["conv3"])
        self.dense1 = nn.Sequential(
            nn.Linear(self._conv_output(param["input"], param["chars"]), param["dense1"]["output"]),
            SELU())
        self.variationaldense = VariationalDense(param=param["varationaldense"])

        self.hyperparameter["conv1"] = self.conv1.getHyperparameter()
        self.hyperparameter["conv2"] = self.conv2.getHyperparameter()
        self.hyperparameter["conv3"] = self.conv3.getHyperparameter()
        self.hyperparameter["varationaldense"] = self.variationaldense.getHyperparameter()

    def getHyperparameter(self):
        return {"MolEncoder": self.hyperparameter}

    def _conv_output(self, i, c):
        tmp = Variable(torch.zeros((1, i, c)), requires_grad=False, volatile=True)
        out = self.conv1(tmp)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        size = out.size()[-1]
        return size

    def forward(self, x, getMeanLogvar=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        out = self.dense1(out)
        out = self.variationaldense(out, getMeanLogvar)
        return out

    def loss(self, x):
        z_mean, z_log_var = self.variationaldense.mean, self.variationaldense.log_var

        prior_logsigma = 0.
        prior_mean = 0.
        regularizer_scale = 1.
        kl = (prior_logsigma - z_log_var +
              0.5 * (-1 + torch.exp(2 * z_log_var) + (z_mean - prior_mean) ** 2) /
              np.exp(2 * prior_logsigma))
        kl = torch.mean(kl) * regularizer_scale
        return kl


class MolDecoder(nn.Module):

    def __init__(self, i=292, o=120, hidden=501, c=35, goindex=1, temperature=0.5, useTeacher=True, **kwargs):
        super(MolDecoder, self).__init__()
        self.hyperparameter = {
            'input': i,
            'output': o,
            'dense1': {'output': i},
            'grus': {'output': hidden,
                     'layers': 3},
            'tgru': {'hidden': hidden,
                     'output': c},
            'temperature': temperature,
        }
        if "param" in kwargs:
            self.hyperparameter.update(kwargs["param"]["MolDecoder"])
        param = self.hyperparameter

        self.goindex = goindex
        self.temperature = param["temperature"]
        if useTeacher:
            self.gotoken = torch.FloatTensor(param["tgru"]['output']).zero_()
            self.gotoken[goindex] = 1
        else:
            self.gotoken = None

        self.latent_input = nn.Linear(param["input"], param["dense1"]["output"])
        self.repeat_vector = Repeat(param["output"], batch_first=True)
        self.gru = nn.GRU(param["dense1"]["output"], param["grus"]["output"],
                          num_layers=param["grus"]["layers"], batch_first=True)
        self.tgru = teacherGRU(input_size=param["grus"]["output"], hidden_size=param["tgru"]['hidden'],
                               output_size=param["tgru"]['output'], gotoken=self.gotoken, useTeacher=useTeacher)

    def forward(self, x, groundTruth=None, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        log_prob, preactivation, sampled_output, h = self.tgru(out, groundTruth=groundTruth, temperature=temperature)
        return log_prob, sampled_output

    def get_before_lastLayerOutput(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        return out

    def get_lastLayerOutput(self, out, groundTruth, temperature=None, getPreactivation=False):
        temperature = self.temperature if temperature is None else temperature
        log_prob, preactivation, sampled_output, h = self.tgru(out, groundTruth=groundTruth, temperature=temperature)
        if getPreactivation:
            return log_prob, sampled_output, preactivation
        else:
            return log_prob, sampled_output

    # Categorical crossentropy
    def loss(self, x, x_decoded: torch.FloatTensor, logProps=True, batch_average=True, timestep_average=True):
        if logProps:
            x_decoded = torch.exp(x_decoded)

        loss = categorical_crossentropy(x_decoded, x, batch_average, timestep_average)

        return loss


class BombarelliAE(nn.Module):

    def __init__(self, maxlen=120, latentdim=56, alphabetlength=35, goindex=1, temperature=0.5):
        super(BombarelliAE, self).__init__()
        self.encoder = MolEncoder(maxlen, latentdim, alphabetlength)
        self.decoder = MolDecoder(latentdim, maxlen, hidden=501, c=alphabetlength, goindex=goindex,
                                  temperature=temperature)
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.00031,
                                          betas=(0.937, 0.999), eps=1e-8, weight_decay=0.0)
        self.best_loss = 1E6
        self.temperature = temperature
        self.epoch = -1
        self.maxvaescale = 1.
        self.is_best = False
        self.default_file = 'bom_checkpoint.pth.tar'
        self.nangrad = False
        self.nanout = False
        self.logdir = "/logs/" + self.__class__.__name__
        self.scheduler = [ReduceLROnPlateau(self.optimizer, mode='min', min_lr=1E-8)]
        reset(self)
        register_nan_checks(self, grad=False, output=False)

    def set_noise(self, noise):
        self.encoder.variationaldense.set_scale(noise)

    def get_noise(self):
        return self.encoder.variationaldense.get_scale()

    def sigmoid_schedule(self, x, slope=0.7, start=10., scale=None):
        if scale is None:
            scale = self.maxvaescale
        return scale * float(1 / (1. + np.exp(slope * (start - float(x)))))

    def load(self, filename=None, map_location=lambda storage, loc: storage):
        filename = self.default_file if filename is None else filename
        checkpoint = torch.load(filename, map_location=map_location)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.epoch = checkpoint['epoch']
        self.temperature = checkpoint['temperature']

    def save(self, filename=None):
        filename = self.default_file if filename is None else filename
        torch.save({
            'epoch': self.epoch,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'temperature': self.temperature,
        }, filename)

    def setuplogger(self, nolog=False):
        self.logger = Logger(log_dir=self.logdir, nolog=nolog)

    def setlogdir(self, log_dir):
        self.logdir = log_dir

    def getlogdir(self):
        return self.logdir

    def forward(self, x, onlySamples=False, temperature=None):
        dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
        x_var = Variable(x.type(dtype))
        y_var = self.encoder(x_var)
        groundTruth = Variable(x.type(dtype), requires_grad=False) if self.training else None
        log_props, samples = self.decoder(y_var, groundTruth=groundTruth, temperature=temperature)
        return log_props, samples

    def train_model(self, train_loader, print_every=50, log_scalar_every=50, log_grad_every=1000, print_mem=False,
                    nolog=False):
        self.train()
        self.epoch += 1
        noise = self.sigmoid_schedule(self.epoch)
        self.set_noise(noise)
        self.optimizer.zero_grad()  # Reset the gradients
        self.is_best = False
        self.setuplogger(nolog=nolog)
        print('Epoch {} (VAE scale {}):'.format(self.epoch, noise))
        start = timer()
        for t, (x, y) in enumerate(train_loader):
            z_var = self(x)[0]
            if self.nanout:
                break
            dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
            x_var = Variable(x.type(dtype))
            encoder_loss = self.encoder.loss(x_var)
            if torch.sum(encoder_loss.data != encoder_loss.data) > 0:
                print("Nan encoder loss at {}".format(t))
            decoder_loss = self.decoder.loss(x_var, z_var, logProps=True)
            if torch.sum(decoder_loss.data != decoder_loss.data) > 0:
                print("Nan decoder loss at {}".format(t))
            loss = decoder_loss + encoder_loss
            loss.backward()  # This calculates the gradients

            if (t + 1) % print_every == 0:
                end = timer()
                t_left = len(train_loader) - t + 1
                average_time = (end - start) / (t + 1)
                eta = str(datetime.timedelta(seconds=round(average_time * t_left)))
                if print_mem:
                    mem = get_memusage()
                    mem_str = ", used memory {} MB / avail memory {} MB".format(mem['memory.used'], mem['memory.total'])
                else:
                    mem_str = ""
                ac = acc(z_var.data, x)
                print('\rt = %d/%d, loss = %.4f, acc = %.4f, encoder_loss = %.4f, decoder_loss = %.4f, eta = %s%s'
                      % (t + 1, len(train_loader), loss.data[0], ac, encoder_loss.data[0], decoder_loss.data[0], eta,
                         mem_str), end="")

            step = (self.epoch * len(train_loader)) + t + 1
            if (t + 1) % log_scalar_every == 0:
                ac = acc(z_var.data, x)
                # (1) Log the scalar values
                info = {
                    'loss': loss.data[0],
                    'encoder_loss': encoder_loss.data[0],
                    'decoder_loss': decoder_loss.data[0],
                    'accuracy': ac,
                    'epoch': self.epoch,
                    'vae_scale': self.encoder.variationaldense.get_scale()
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)
            if (t + 1) % log_grad_every == 0:
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in chain(self.encoder.named_parameters(), self.decoder.named_parameters()):
                    if value.grad is not None:  # We have to save only the leafs with grads
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, to_np(value), step)
                        self.logger.histo_summary(tag + '/grad', to_np(value.grad), step)

            if self.nangrad:
                break
            utils.clip_grad_norm(chain(self.encoder.parameters(), self.decoder.parameters()), 2.)  # clip the gradients
            self.optimizer.step()
            self.optimizer.zero_grad()  # This resets the gradients

        print()

    def validate_model(self, val_loader, print_every=50, log_scalar_every=50, print_mem=False, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        self.eval()
        self.optimizer.zero_grad()  # Reset the gradients
        avg_val_loss = 0.
        print('Validation Epoch {}:'.format(self.epoch))
        start = timer()
        for t, (x, y) in enumerate(val_loader):
            dtype = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor
            x_var = Variable(x.type(dtype), requires_grad=False, volatile=True)
            y_var = self.encoder(x_var)
            z_var, samples = self.decoder(y_var, temperature=temperature)
            encoder_loss = self.encoder.loss(x_var)
            decoder_loss = self.decoder.loss(x_var, z_var, logProps=True)
            loss = decoder_loss + encoder_loss
            avg_val_loss += loss.data[0]
            if (t + 1) % print_every == 0:
                end = timer()
                t_left = len(val_loader) - t + 1
                average_time = (end - start) / (t + 1)
                eta = str(datetime.timedelta(seconds=round(average_time * t_left)))
                if print_mem:
                    mem = get_memusage()
                    mem_str = ", used memory {} MB / avail memory {} MB".format(mem['memory.used'], mem['memory.total'])
                else:
                    mem_str = ""
                ac = acc(torch.exp(z_var.data), x)
                print(
                    '\rt = %d/%d, val_loss = %.4f, val_acc = %.4f, val_encoder_loss = %.4f, val_decoder_loss = %.4f, eta = %s%s'
                    % (
                    t + 1, len(val_loader), loss.data[0], ac, encoder_loss.data[0], decoder_loss.data[0], eta, mem_str),
                    end="")
            if (t + 1) % log_scalar_every == 0:
                ac = acc(z_var.data, x)
                step = (self.epoch * len(val_loader)) + t + 1
                # (1) Log the scalar values
                info = {
                    'val_loss': loss.data[0],
                    'val_encoder_loss': encoder_loss.data[0],
                    'val_decoder_loss': decoder_loss.data[0],
                    'val_accuracy': ac,
                    'val_epoch': self.epoch
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

        print()
        if t > 0:
            avg_val_loss /= t
        print('average validation loss: %.4f' % avg_val_loss)
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
