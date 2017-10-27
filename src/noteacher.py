# coding=utf-8
from itertools import chain

import torch

from .bombarelli import MolDecoder, BombarelliAE
from .utils import reset, ReduceLROnPlateau


class NoTeacherAE(BombarelliAE):
    def __init__(self, maxlen=120, latentdim=56, alphabetlength=35, goindex=1, temperature=0.5):
        super(NoTeacherAE, self).__init__(maxlen=maxlen, latentdim=latentdim, alphabetlength=alphabetlength,
                                          goindex=goindex, temperature=temperature)
        self.default_file = 'noteacher_checkpoint.pth.tar'
        self.decoder = MolDecoder(latentdim, maxlen, hidden=501, c=alphabetlength, temperature=temperature,
                                  useTeacher=False)
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.00031,
                                          betas=(0.937, 0.999), eps=1e-8, weight_decay=0.0)
        self.scheduler = [ReduceLROnPlateau(self.optimizer, mode='min', min_lr=1E-8)]
        reset(self)
