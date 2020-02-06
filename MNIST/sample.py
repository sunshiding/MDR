from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable
from data import create_dataset
import torch.nn.functional as F
from model import MDR
from torchvision.utils import save_image


def fetch_mnist_image(nsamples):
    test_loader = create_dataset(100, False)
    imagesA, imagesB = [], []
    for batch_idx, imageS in enumerate(test_loader):
        imagesA.append(imageS['A'])
        imagesB.append(imageS['B'])
    imagesA = torch.cat(imagesA).cpu().numpy()
    imagesB = torch.cat(imagesB).cpu().numpy()
    index = np.random.choice(np.arange(imagesA.shape[0]), size=nsamples)
    imageA = imagesA[index]
    imageB = imagesB[index]

    imageA = torch.from_numpy(imageA).float()
    imageB = torch.from_numpy(imageB).float()
    return Variable(imageA, volatile=True), Variable(imageB, volatile=True)


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MDR(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


def visualization(epoch, save_path='images'):
    n_samples = 64
    cuda = False
    model_path = './' + save_path + '/trained_models/checkpoint.pth.tar'
    model = load_checkpoint(model_path, use_cuda=cuda)
    model.eval()
    if cuda:
        model.cuda()

    imageA, imageB = fetch_mnist_image(n_samples)
    if cuda:
        imageA = imageA.cuda()
        imageB = imageB.cuda()

    img_muA_S, image_SA_logvar = model.imageA_encoderS(imageA)
    img_muA_E, image_EA_logvar = model.imageA_encoderE(imageA)
    imgA_mu, img_logvar = model.product([img_muA_S, image_SA_logvar], [img_muA_E, image_EA_logvar])

    std = image_SA_logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    img_zA_S = eps.mul(std).add_(img_muA_S)

    imgA_recon = F.sigmoid(model.imageA_decoder(imgA_mu))
    imgAs_recon = F.sigmoid(model.imageA_decoder(img_muA_S))
    imgAe_recon = F.sigmoid(model.imageA_decoder(img_muA_E))
    imgAs_reconB = F.sigmoid(model.imageB_decoder(img_muA_S))
    img_muB_S, image_SB_logvar = model.imageB_encoderS(imageB)
    img_muB_E, image_EB_logvar = model.imageB_encoderE(imageB)
    imgB_mu, img_logvar = model.product([img_muB_S, image_SB_logvar], [img_muB_E, image_EB_logvar])
    imgB_recon = F.sigmoid(model.imageB_decoder(imgB_mu))
    imgBs_recon = F.sigmoid(model.imageB_decoder(img_muB_S))
    imgBs_reconA = F.sigmoid(model.imageA_decoder(img_muB_S))
    imgBe_recon = F.sigmoid(model.imageB_decoder(img_muB_E))
    imgAB_mu, imgAB_logvar = model.product([img_muA_S, image_SA_logvar], [img_muB_E, image_EB_logvar])
    imgAB_recon = F.sigmoid(model.imageB_decoder(imgAB_mu))

    imgBA_mu, imgBA_logvar = model.product([img_muB_S, image_SB_logvar], [img_muA_E, image_EA_logvar])
    imgBA_recon = F.sigmoid(model.imageA_decoder(imgBA_mu))

    # save image samples to filesystem
    save_image(imgA_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgA_recon' + str(epoch) + '.png')
    save_image(imgAs_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgAs_recon' + str(epoch) + '.png')
    save_image(imgAe_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgAe_recon' + str(epoch) + '.png')
    save_image(imgBs_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgBs_recon' + str(epoch) + '.png')
    save_image(imgB_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgB_recon' + str(epoch) + '.png')
    save_image(imgBe_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgBe_recon' + str(epoch) + '.png')
    save_image(imgAB_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgAB_recon' + str(epoch) + '.png')
    save_image(imgBA_recon.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgBA_recon' + str(epoch) + '.png')
    save_image(imgAs_reconB.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgAs_reconB' + str(epoch) + '.png')
    save_image(imgBs_reconA.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imgBs_reconA' + str(epoch) + '.png')
    save_image(imageA.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imageA' + str(epoch) + '.png')
    save_image(imageB.view(n_samples, 3, 28, 28),
               './' + save_path + '/images/imageB' + str(epoch) + '.png')


if __name__ == "__main__":

    for epoch in range(20):
        visualization(epoch, 'paper_change')