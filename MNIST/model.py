from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from numbers import Number
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

Model = collections.namedtuple("MODEL",
                               "recon_Loss, img_cross_loss, txt_cross_loss, loss_C_E, sharedloss")

loss_MSE = nn.MSELoss()


def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    loss = y_onehot * log_input
    return -loss


def elbo_loss(recon_image, image, recon_text, text, mu, logvar, s_mu, s_logvar, e_mu, e_logvar,
              lambda_image=1.0, lambda_text=1.0, annealing_factor=1, distribution="bernoulli"):
    """Bimodal ELBO loss function.

    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_text: torch.Tensor
                       reconstructed text probabilities
    @param text: torch.Tensor
                 input text (one-hot)
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_text: float [default: 1.0]
                       weight for text BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """

    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        batch_size = image.size(0)
        if distribution == "bernoulli":
            image_bce = torch.sum(binary_cross_entropy_with_logits(
                recon_image.view(-1, 3 * 28 * 28),
                image.view(-1, 3 * 28 * 28)), dim=1)
        elif distribution == "gaussian":
            image_bce = F.mse_loss(recon_image.view(-1, 3 * 28 * 28) * 255,
                                   image.view(-1, 3 * 28 * 28) * 255, reduction="mean") / 255

    if recon_text is not None and text is not None:
        batch_size = text.size(0)
        if distribution == "bernoulli":
            text_bce = torch.sum(binary_cross_entropy_with_logits(
                recon_text.view(-1, 3 * 28 * 28),
                text.view(-1, 3 * 28 * 28)), dim=1)
        elif distribution == "gaussian":
            text_bce = F.mse_loss(recon_text.view(-1, 3 * 28 * 28) * 255,
                                  text.view(-1, 3 * 28 * 28) * 255, reduce="mean") / 255

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD_S = -0.5 * torch.sum(1 + s_logvar - s_mu.pow(2) - s_logvar.exp(), dim=1)
    KLD_E = -0.5 * torch.sum(1 + e_logvar - e_mu.pow(2) - e_logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + lambda_text * text_bce
                      + annealing_factor * (KLD + KLD_S + KLD_E))
    return ELBO


def log_density(mu, logval, sample):
    logsigma = 0.5 * logval
    c = Variable(torch.Tensor([np.log(2 * np.pi)])).type_as(mu.data)
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)


def logsumexp(value, dim=None, keepdim=True):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


criterion = nn.CrossEntropyLoss()


class MDR(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(MDR, self).__init__()
        self.imageA_encoderS = ImageEncoder(n_latents)
        self.imageA_decoder = ImageDecoder(n_latents)
        self.imageA_encoderE = ImageEncoder(n_latents)
        self.netIA_S2E = MLP(n_latents, n_latents)
        self.netIA_E2S = MLP(n_latents, n_latents)

        self.imageB_encoderS = ImageEncoder(n_latents)
        self.imageB_decoder = ImageDecoder(n_latents)
        self.imageB_encoderE = ImageEncoder(n_latents)
        self.netIB_S2E = MLP(n_latents, n_latents)
        self.netIB_E2S = MLP(n_latents, n_latents)

        self.netC_E = Discriminator(n_latents)

        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, image=None, text=None):
        img_mu_S, img_logvar_S, img_mu_E, img_logvar_E, txt_mu_S, txt_logvar_S, txt_mu_E, txt_logvar_E \
            = self.infer(image, text)
        img_mu, img_logvar = self.product([img_mu_S, img_logvar_S], [img_mu_E, img_logvar_E])
        txt_mu, txt_logvar = self.product([txt_mu_S, txt_logvar_S], [txt_mu_E, txt_logvar_E])
        # reparametrization trick to sample
        img_z_S = self.reparametrize(img_mu_S, img_logvar_S)
        img_z_E = self.reparametrize(img_mu_E, img_logvar_E)
        txt_z_S = self.reparametrize(txt_mu_S, txt_logvar_S)
        txt_z_E = self.reparametrize(txt_mu_E, txt_logvar_E)
        img_z = self.reparametrize(img_mu, img_logvar)
        txt_z = self.reparametrize(txt_mu, txt_logvar)
        # reconstruct inputs based on that gaussian
        img_recon = self.imageA_decoder(img_z)
        txt_recon = self.imageB_decoder(txt_z)

        product = (torch.sum((img_z_E - img_z_E.mean()).pow(2), 1) * torch.sum((img_z_S - img_z_S.mean()).pow(2),
                                                                               1))
        img_cross_loss = (
                torch.sum((img_z_S - img_z_S.mean()) * (img_z_E - img_z_E.mean()), 1) / product).abs().sum()
        product = (torch.sum((txt_z_E - txt_z_E.mean()).pow(2), 1) * torch.sum((txt_z_S - txt_z_S.mean()).pow(2),
                                                                               1))
        txt_cross_loss = (
                torch.sum((txt_z_S - txt_z_S.mean()) * (txt_z_E - txt_z_E.mean()), 1) / product).abs().sum()

        product = (torch.sum((txt_z_E - img_z_E.mean()).pow(2), 1) * torch.sum((txt_z_E - img_z_E.mean()).pow(2),
                                                                               1))
        # change
        loss_C_E = (torch.sum((txt_z_E - txt_z_E.mean()) * (img_z_E - img_z_E.mean()), 1) / product).abs().sum()
        sharedloss = shared_dv_loss(img_z_S, txt_z_S)
        image_loss = elbo_loss(img_recon, image, None, None, img_mu, img_logvar, img_mu_S, img_logvar_S, img_mu_E,
                               img_logvar_E)
        text_loss = elbo_loss(None, None, txt_recon, text, txt_mu, txt_logvar, txt_mu_S, txt_logvar_S, txt_mu_E,
                              txt_logvar_E)
        recon_Loss = image_loss + text_loss

        return Model(recon_Loss, img_cross_loss, txt_cross_loss, loss_C_E, sharedloss)

    def infer(self, image=None, text=None):
        batch_size = image.size(0) if image is not None else text.size(0)
        # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        img_mu_S = 0;
        img_logvar_S = 0;
        img_mu_E = 0;
        img_logvar_E = 0;
        txt_mu_S = 0;
        txt_logvar_S = 0;
        txt_mu_E = 0
        txt_logvar_E = 0
        if image is not None:
            img_mu_S, img_logvar_S = self.imageA_encoderS(image)
            img_mu_E, img_logvar_E = self.imageA_encoderE(image)
        if text is not None:
            txt_mu_S, txt_logvar_S = self.imageB_encoderS(text)
            txt_mu_E, txt_logvar_E = self.imageB_encoderE(text)

        return img_mu_S, img_logvar_S, img_mu_E, img_logvar_E, txt_mu_S, txt_logvar_S, txt_mu_E, txt_logvar_E

    def product(self, shared, exlusive):
        batch_size = shared[0].size(0)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        mu, logvar = prior_expert((1, batch_size, self.n_latents),
                                  use_cuda=use_cuda)

        mu = torch.cat((mu, shared[0].unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, shared[1].unsqueeze(0)), dim=0)

        mu = torch.cat((mu, exlusive[0].unsqueeze(0)), dim=0)
        logvar = torch.cat((logvar, exlusive[1].unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


def shared_dv_loss(img_z_S, txt_z_S):
    u = torch.mm(img_z_S, txt_z_S.t())
    mask = torch.eye(img_z_S.size()[0]).cuda()
    n_mask = 1 - mask.cuda()

    # Positive term is just the average of the diagonal.
    E_pos = (u * mask).sum() / mask.sum()

    # Negative term is the log sum exp of the off-diagonal terms. Mask out the positive.
    u -= 10 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos

    return loss


def shared_fenchel_dual_loss(img_z_S, txt_z_S, measure='DV'):
    u = torch.mm(img_z_S, txt_z_S.t())
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    mask = torch.eye(img_z_S.size()[0]).cuda()
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos = (E_pos * mask).sum() / mask.sum()
    n_mask = 1 - mask
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos
    return loss


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.n_latents = n_latents
        self.upsampler = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(10, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 10))

    def forward(self, z):
        z = self.net(z)
        return z  # NOTE: no softmax here. See train.py


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, y_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, y_dim),
            Swish(),
            nn.Linear(y_dim, y_dim),
            Swish(),
            nn.Linear(y_dim, 2)
        )

    def forward(self, y):
        return self.net(y).squeeze()


class DenseNet_Classifier(nn.Module):
    def __init__(self, dim=64, num_classes=10):
        super(DenseNet_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, z):
        out = self.classifier(z)
        return out


class MLP(nn.Module):
    def __init__(self, s_dim, t_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            Swish(),
            nn.Linear(t_dim, t_dim),
            Swish(),
            nn.Linear(t_dim, t_dim),
            Swish()
        )

    def forward(self, s):
        t = self.net(s)
        return t
