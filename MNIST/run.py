from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os

import shutil
import torch
import torch.optim as optim
from torch.autograd import Variable
from model import MDR
from data import create_dataset
from sample import visualization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MDR(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=0, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: True]')
    parser.add_argument('--gamma', default=10, type=float, help='Cross inference')
    parser.add_argument('--alpha', default=10, type=float, help='Exclusive')
    parser.add_argument('--beta', default=10, type=float, help='Shared')

    args = parser.parse_args()
    args.save_path = 'paper_change'
    args.cuda = args.cuda and torch.cuda.is_available()
    args.dataroot = './datasets/MNIST'
    print(args)
    if not os.path.isdir('./' + args.save_path + '/trained_models'):
        os.makedirs('./' + args.save_path + '/trained_models')
    if not os.path.isdir('./' + args.save_path + '/images'):
        os.makedirs('./' + args.save_path + '/images')

    train_loader = create_dataset(args.batch_size)  # create a dataset given opt.dataset_mode and other options
    N_mini_batches = len(train_loader)  # get the number of images in the dataset.
    test_loader = create_dataset(args.batch_size, False)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % N_mini_batches)

    model = MDR(args.n_latents)
    if args.cuda:
        model.cuda()

    optimizerall = optim.Adam([{'params': model.imageA_encoderS.parameters()},
                               {'params': model.imageA_encoderE.parameters()},
                               {'params': model.imageB_encoderS.parameters()},
                               {'params': model.imageB_encoderE.parameters()},
                               {'params': model.imageA_decoder.parameters()},
                               {'params': model.imageB_decoder.parameters()}], lr=args.lr)

    def train(epoch):
        model.train()

        train_loss_meter = AverageMeter()
        train_recon_loss_meter = AverageMeter()
        train_cross_loss_meter = AverageMeter()
        train_shared_loss_meter = AverageMeter()
        for batch_idx, data in enumerate(train_loader):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx * args.batch_size + epoch * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            if args.cuda:
                image_A = data['A'].cuda()
                image_B = data['B'].cuda()

            image_A = Variable(image_A)
            image_B = Variable(image_B)
            batch_size = len(image_A)

            result = model(image_A, image_B)
            train_loss_all = result.recon_Loss + args.gamma * (result.img_cross_loss + result.txt_cross_loss) \
                             + args.alpha * result.loss_C_E + args.beta * result.sharedloss

            # refresh the optimizer
            optimizerall.zero_grad()
            train_loss_all.backward()
            optimizerall.step()

            train_loss_meter.update(train_loss_all.item(), batch_size)
            train_recon_loss_meter.update(result.recon_Loss.item(), batch_size)
            train_cross_loss_meter.update(args.gamma * (100 * result.img_cross_loss + result.txt_cross_loss).item(),
                                          batch_size)
            train_shared_loss_meter.update(args.beta * result.sharedloss.item(), batch_size)
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\trecon Loss: {:.6f}\tcross Loss: {:.6f}\tshared Loss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                        epoch, batch_idx * len(image_A), len(train_loader.dataset),
                               100. * batch_size * batch_idx / len(train_loader), train_loss_meter.avg,
                        train_recon_loss_meter.avg, train_cross_loss_meter.avg, train_shared_loss_meter.avg,
                        annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()
        train_recon_loss_meter = AverageMeter()
        train_cross_loss_meter = AverageMeter()
        train_shared_loss_meter = AverageMeter()
        for batch_idx, data in enumerate(test_loader):
            if args.cuda:
                image_A = data['A'].cuda()
                image_B = data['B'].cuda()
            with torch.no_grad():
                image_A = Variable(image_A)
                image_B = Variable(image_B)
            batch_size = len(image_A)

            result = model(image_A, image_B)
            test_loss_all = result.recon_Loss + args.gamma * (result.img_cross_loss + result.txt_cross_loss) \
                            + args.alpha * result.loss_C_E + args.beta * result.sharedloss

            test_loss_meter.update(test_loss_all.item(), batch_size)
            train_recon_loss_meter.update(result.recon_Loss.item(), batch_size)
            train_cross_loss_meter.update(args.gamma * (result.img_cross_loss + result.txt_cross_loss).item(),
                                          batch_size)
            train_shared_loss_meter.update(args.beta * result.sharedloss.item(), batch_size)
            if batch_idx % args.log_interval == 0:
                print(
                    'test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\trecon Loss: {:.6f}\tcross Loss: {:.6f}\tshared Loss: {:.6f}'.format(
                        epoch, batch_idx * len(image_A), len(train_loader.dataset),
                               100. * batch_size * batch_idx / len(train_loader), test_loss_meter.avg,
                        train_recon_loss_meter.avg, train_cross_loss_meter.avg, train_shared_loss_meter.avg))

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg


    def classification(epoch):

        model.train()
        classification_loss_meter = AverageMeter()
        optimizer = optim.Adam([{'params': model.classifier_S.parameters()},
                                {'params': model.classifier_E.parameters()},
                                {'params': model.classifier_C.parameters()}], lr=1e-3)
        for i in range(epoch):
            # NOTE: is_paired is 1 if the example is paired
            for batch_idx, (image_A, image_B) in enumerate(train_loader):

                if args.cuda:
                    image_A = image_A.cuda()
                    image_B = image_B.cuda()

                image_A = Variable(image_A)
                image_B = Variable(image_B)
                batch_size = len(image_A)

                # pass data through model
                result = model(image_A, image_B)

                classification_loss = result.s_criterion_loss + result.e_criterion_loss + result.c_criterion_loss

                # compute gradients and take step
                # refresh the optimizer
                optimizer.zero_grad()
                classification_loss.backward()
                optimizer.step()

                classification_loss_meter.update(classification_loss.item(), batch_size)
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(image_A), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), classification_loss_meter.avg))

            print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, classification_loss_meter.avg))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'n_latents': args.n_latents
            }, is_best, folder='./' + args.save_path + '/trained_models')


    best_loss = sys.maxsize
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test(epoch)
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents
        }, is_best, folder='./' + args.save_path + '/trained_models')
        model.eval()
        visualization(epoch, args.save_path)
