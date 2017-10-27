# coding=utf-8
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Repeat(nn.Module):

    def __init__(self, reps, batch_first=False):
        super(Repeat, self).__init__()
        self.reps = reps
        self.batch_first = batch_first

    def forward(self, x):
        dims = tuple(x.size())
        if self.batch_first:
            dims = (dims[0], 1) + dims[1:]
        else:
            dims = (1,) + dims[0:]
        x_repeated = x.view(*dims)
        repeats = [1 for i in dims]
        if self.batch_first:
            repeats[1] = self.reps
        else:
            repeats[0] = self.reps
        return x_repeated.repeat(*repeats)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha, inplace=True)


class ConvSELU(nn.Module):
    def __init__(self, i=0, o=0, kernel_size=3, padding=0, p=0., **kwargs):
        super(ConvSELU, self).__init__()
        self.hyperparameter = {
            'input': i,
            'output': o,
            'kernel_size': kernel_size,
            'padding': padding,
            'dropout': p, }
        if "param" in kwargs:
            self.hyperparameter.update(kwargs["param"]["ConvSELU"])
        param = self.hyperparameter
        self.conv = nn.Conv1d(param["input"], param["output"], kernel_size=param["kernel_size"],
                              padding=param["padding"])
        self.activation = selu
        self.dropout = None
        if param["dropout"] > 0.:
            raise NotImplementedError("No alpha dropout yet")
            self.dropout = nn.Dropout(param["dropout"])

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def getHyperparameter(self):
        return {"ConvSELU": self.hyperparameter}


class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()

    def forward(self, x):
        return selu(x)


class VariationalDense(nn.Module):

    def __init__(self, i=435, o=292, scale=1E-2, **kwargs):
        super(VariationalDense, self).__init__()
        self.hyperparameter = {
            'input': i,
            'output': o,
            'scale': scale,
        }
        if "param" in kwargs:
            self.hyperparameter.update(kwargs["param"]["VariationalDense"])
        param = self.hyperparameter
        self.scale = param["scale"]
        self.z_mean = nn.Sequential(nn.Linear(param["input"], param["output"]),
                                    nn.Hardtanh(min_value=-2, max_value=2, inplace=True))
        self.z_log_var = nn.Linear(param["input"], param["output"])
        self.mean = None
        self.log_var = None

    def getHyperparameter(self):
        return {"VariationalDense": self.hyperparameter}

    def forward(self, x, getMeanLogvar=False):
        self.mean = self.z_mean(x)
        self.log_var = self.z_log_var(x)
        if getMeanLogvar:
            return self.mean, self.log_var
        if self.training:
            eps = Variable(torch.FloatTensor(*self.log_var.size()).normal_(mean=0, std=1)).type_as(
                self.mean) * self.scale
            std = torch.exp(self.log_var / 2.)
            # we limit the output to -2 and 2. This makes samling and searching later way easier
            return nn.Hardtanh(min_value=-2, max_value=2, inplace=True)(eps.mul(std).add_(self.mean))
        else:
            return self.mean

    def set_scale(self, scale):
        self.scale = scale
        self.hyperparameter["scale"] = scale

    def get_scale(self):
        return self.scale


def sample_gumbel(input):
    noise = torch.rand(input.size())
    if input.is_cuda:
        noise = noise.cuda()
    eps = 1e-9
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)


def gumbel_softmax_sample(input_, hard=True, temperature=1., uselogprop=True):
    # Softmax has some undesired behaviour in pytorch 0.1.12:
    # if the input is 2D then it operates over dimension 1, if the input is 3D it operates on dimension 0
    # thus, to sample a 3D tensor (timestep, batch, chars) correctly, we have to reshape it to (timestep * batch, chars)
    # before we can continue
    size = tuple(input_.size())
    if input_.dim() >= 3:
        input_ = input_.view(-1, input_.size(-1))
    noise = sample_gumbel(input_)
    if uselogprop:
        x = (input_ + noise) / temperature
        x = F.log_softmax(x)
    else:
        x = (torch.log(input_) + noise) / temperature
        x = F.softmax(x)

    if hard == True:
        _, max_inx = torch.max(x, x.dim() - 1)
        if x.is_cuda:
            x_hard = torch.cuda.FloatTensor(x.size()).zero_().scatter_(x.dim() - 1, max_inx.data, 1.0)
        else:
            x_hard = torch.FloatTensor(x.size()).zero_().scatter_(x.dim() - 1, max_inx.data, 1.0)
        x2 = x.clone()
        tmp = Variable(x_hard - x2.data)
        tmp.detach_()

        x = tmp + x

    return x.view(size)


class hard_sigmoid(nn.Hardtanh):
    """Applies a linear approximated standard sigmoid"""

    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__(min_value=0, max_value=1, inplace=inplace)
        self.slope = 0.2
        self.shift = 0.5

    def forwad(self, x):
        x = x.mul(self.slope).add_(self.shift)
        return F.hardtanh(x, min_val=0, max_val=1, inplace=True)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + 'min_val=' + str(self.min_val) \
               + ', max_val=' + str(self.max_val) \
               + ', slope=' + str(self.slope) \
               + ', shift=' + str(self.shift) \
               + inplace_str + ')'


class advGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, activation=F.tanh, inner_activation=F.sigmoid):
        super(advGRUCell, self).__init__(input_size, hidden_size, bias=True)
        self.activation = activation
        self.inner_activation = inner_activation

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.autograd.Variable(input.data.new(
                input.size(0),
                self.hidden_size).zero_(), requires_grad=False)
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        resetgate = self.inner_activation(i_r + h_r)
        inputgate = self.inner_activation(i_z + h_z)
        preactivation = i_n + resetgate * h_n
        newgate = self.activation(preactivation)
        hy = newgate + inputgate * (hx - newgate)

        return hy, preactivation


class teacherGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=F.log_softmax,
                 gru_activation=F.tanh, gru_inner_activation=F.sigmoid, useTeacher=True,
                 gotoken=None):
        if useTeacher and gotoken is None:
            raise ValueError("Need to provide a gotoken when using teachers forcing")
        super(teacherGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.teacher = useTeacher
        self.gotoken = gotoken
        if useTeacher:
            self.cell = advGRUCell(input_size=input_size + output_size, hidden_size=hidden_size,
                                   activation=gru_activation, inner_activation=gru_inner_activation)
        else:
            self.cell = advGRUCell(input_size=input_size, hidden_size=hidden_size,
                                   activation=gru_activation, inner_activation=gru_inner_activation)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x, groundTruth=None, hx=None, max_length=None, temperature=0.5):
        # if self.teacher and self.training and groundTruth is None:
        #    raise NotImplementedError("No groundTruth in teachers trainingsmode")
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        sampled_output = []
        preactivation = []
        if hx is None:
            hx = Variable(x.data.new(
                batch_size,
                self.hidden_size).zero_(), requires_grad=False)
        for i in range(max_length):
            if self.teacher and i == 0:
                input_ = torch.cat(
                    [x[:, i, :], Variable(self.gotoken.repeat(batch_size, 1), requires_grad=False).type_as(x)], dim=-1)
                hx, pre = self.cell(input_, hx=hx)
            elif self.teacher and groundTruth is not None:
                input_ = torch.cat([x[:, i, :], groundTruth[:, i - 1, :]], dim=-1)
                hx, pre = self.cell(input_, hx=hx)
            elif self.teacher and groundTruth is None:
                input_ = torch.cat([x[:, i, :], sampled_output[-1]], dim=-1)
                hx, pre = self.cell(input_, hx=hx)
            elif not self.teacher:
                input_ = x[:, i, :]
                hx, pre = self.cell(input_, hx=hx)
            else:
                raise NotImplementedError("TeacherGRU. Unknown operation mode")
            output_ = self.activation(self.linear(hx))
            output.append(output_.view(batch_size, 1, self.output_size))
            preactivation.append(pre.view(batch_size, 1, self.hidden_size))
            sampled_output.append(gumbel_softmax_sample(output_, hard=True, temperature=temperature, uselogprop=True))

            ### Multinomial sampling if needed
            # indices = torch.multinomial(torch.exp(output_).data, 1)
            # one_hot = output_.data.new(output_.size(0),
            #                                self.hidden_size).zero_()
            # one_hot.scatter_(1, indices, 1)
            # one_hot = Variable(one_hot)
            # sampled_output.append(one_hot)

        output = torch.cat(output, 1)
        preactivation = torch.cat(preactivation, 1)
        sampled_output = torch.stack(sampled_output, 1)

        return output, preactivation, sampled_output, hx


class noteacherGRU(teacherGRU):
    def __init__(self, input_size, hidden_size, output_size, activation=F.log_softmax,
                 gru_activation=F.tanh, gru_inner_activation=F.sigmoid, useTeacher=False,
                 gotoken=None):
        if useTeacher:
            raise ValueError("noteacherGRU can't use teachersforcing.")
        super(noteacherGRU, self).__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                           activation=activation, gru_activation=gru_activation,
                                           gru_inner_activation=gru_inner_activation, useTeacher=False,
                                           gotoken=gotoken
                                           )
        self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, groundTruth=None, hx=None, max_length=None, temperature=0.5):
        if groundTruth is not None:
            ValueError("Can't use groundTruth")
        timesteps = x.size()[1]
        batch_size = x.size()[0]
        if hx is None:
            hx = Variable(x.data.new(
                1,
                batch_size,
                self.hidden_size).zero_(), requires_grad=False)
        out, hx = self.cell(x, hx=hx)
        if not out.is_contiguous():
            out.contiguous()
        preactivation = out.clone()
        out = out.view(-1, self.hidden_size)
        out = self.activation(self.linear(out))
        sampled_output = gumbel_softmax_sample(out, hard=True, temperature=temperature, uselogprop=True)
        output = out.view(batch_size, timesteps, self.output_size).contiguous()
        sampled_output = sampled_output.view(batch_size, timesteps, self.output_size).contiguous()

        return output, preactivation, sampled_output, hx
