import os
import time
import random
import shutil
import argparse
import warnings
from datetime import datetime

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

import tblog
from utils import *
from model import Model
from dataset import get_loaders, dali_is_enabled
from cerranet import cerranet

if tblog.is_enabled:
    from tqdm import tqdm

model_names = sorted([name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                      and callable(models.__dict__[name])] + ['cerranet'])

accepted_metrics = ['loss', 'f1-score', 'accuracy', 'b-accuracy']

parser = argparse.ArgumentParser(description='PyTorch Cerrado/Dataset assessment')
parser.add_argument('--root', '-r', metavar='PATH', default='./data',
                    type=PathType(exists=True, type='dir'),
                    help='root of data directory')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--ts', '--training-split', default=0.8, type=float,
                    metavar='TS', help='fraction of data to reserve for training (default: 0.8)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--runs', default=1, type=int, metavar='N',
                    help='number of repetitions of the experiment (default: 5)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')                    
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--lr-steps', default=None, type=int, nargs="+",
                    metavar='N', help='epochs to decay learning rate (default=None)')
parser.add_argument('--lr-decay', default=1, type=float, metavar='LD',
                    help='lr decay factor (default: 1.0)')

parser.add_argument('--patience', default=10, type=int, metavar='N',
                    help='Number of epochs to wait if no improvement and then stop the training (default: none)')
parser.add_argument('--min-delta', default=0.0, type=float, metavar='MD',
                    help='A minimum increase in the score to qualify as an improvement (default: 0.0)')
parser.add_argument('--following-metric', dest='following_metric', default='f1-score', choices=accepted_metrics,
                    help=f"metric to be followed. ({' | '.join(accepted_metrics)}, default=f1-score)")

parser.add_argument('--print-freq', '-p', default=600, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training (default: none)')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU id to use (default: none)')
parser.add_argument('--save-every', dest='save_every', metavar='N',
                    help='Save checkpoints at every specified number of epochs (default: 25)',
                    type=int, default=25)
parser.add_argument('--exp-name', dest='exp_name', default='', type=str,
                    help='Suffix to be added at the end of the logdir')
parser.add_argument('--channels-last', type=bool, default=False)

num_classes = 5

def main():
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training.')

    if args.gpu is None:
        args.device = 'cpu'
        warnings.warn("You haven't chosen a specific GPU, so CPU will be used.")
    else:
        args.device = f'cuda:{args.gpu}'

    if args.following_metric == 'loss':
        args.compare_metrics = lambda before, after: before > after
    else:
        args.compare_metrics = lambda before, after: before < after

    test_averages = {x: np.zeros(args.runs) for x in accepted_metrics}
    for run in range(args.runs):

        train_loader, val_loader, test_loader = get_loaders(run, args)

        # Simply call main_worker function
        test_metrics = main_worker(run, train_loader, val_loader, test_loader, args)
        for key in test_metrics.keys(): test_averages[key][run] = test_metrics[key]

    print(' * Best Tesst average metrics -> ')
    print('\t' + ' '.join([f'{key}:{np.mean(test_averages[key]):.3f}/{np.std(test_averages[key]):.3f}' for key in test_averages.keys()]))
    

def main_worker(run, train_loader, val_loader, test_loader, args):

    run_prefix = args.arch.lower() + '_' + \
                 str(run + 1) + '_of_' + str(args.runs)

    run_exp = "{}_arch{}_runs{}_epochs{}_lr{}_batchsize{}_patience{}_pretrained{}_seed{}_trainingsplit{}_follow{}".format(
                                args.exp_name, 
                                args.arch,
                                args.runs,
                                args.epochs, 
                                args.lr, 
                                args.batch_size,
                                "True" if args.patience is not None else "False",
                                args.pretrained, 
                                args.seed,
                                args.ts,
                                args.following_metric)

    run_dir = os.path.join('runs', run_exp, run_prefix)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
        if tblog.is_enabled:
            tblog.logdir = run_dir

    print("Use {} for training".format(args.device))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if 'cerranet' in args.arch:
            backbone = cerranet(pretrained=True)
        elif 'inception' in args.arch:
            backbone = models.__dict__[args.arch](pretrained=True, aux_logits=False)
        else:
            backbone = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if 'cerranet' in args.arch:
            backbone = cerranet()
        elif 'inception' in args.arch:
            backbone = models.__dict__[args.arch](aux_logits=False)
        else:
            backbone = models.__dict__[args.arch]()

    model = Model(backbone, num_classes)
    print(model)

    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.to(args.device, memory_format=memory_format)
    else:
        model = model.to(args.device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if not args.lr_steps is None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            gamma=args.lr_decay,
                                                            milestones=args.lr_steps,
                                                            last_epoch=args.start_epoch - 1)

    # optionally resume from a checkpoint
    best_followed = 0.
    if args.resume:
        if args.evaluate:
            filename = os.path.join(run_dir, 'model_best.pth.tar')
        else:
            filename = os.path.join(run_dir, 'checkpoint.pth.tar')

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))

            if not args.gpu is None:
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location=args.device)

            rsetattr(args, 'start_epoch', checkpoint['epoch']) 
            best_followed = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.lr_steps is None:
                lr_scheduler.load_state_dict(checkpoint['scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    if args.evaluate:
        metrics = validate(test_loader, model, criterion, args)
        return metrics

    counter = 0
    writer = tblog.SummaryWriter(tblog.logdir) if tblog.is_enabled else None
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.lr_steps is None:
            lr_scheduler.step()

        # evaluate on validation set
        val_metrics = validate(val_loader, model, criterion, args)
        val_followed = val_metrics[args.following_metric]

        if tblog.is_enabled:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/test', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/test', val_metrics['accuracy'], epoch)
            writer.add_scalar('Balanced_Accuracy/train', train_metrics['b-accuracy'], epoch)
            writer.add_scalar('Balanced_Accuracy/test', val_metrics['b-accuracy'], epoch)
            writer.add_scalar('F1-score/train', train_metrics['f1-score'], epoch)
            writer.add_scalar('F1-score/test', val_metrics['f1-score'], epoch)

        # check early stopping condition
        if not args.patience is None:
            if not args.compare_metrics(best_followed + args.min_delta, val_followed):
                is_best = False
                counter += 1
                if counter >= args.patience:
                    #save last
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_followed,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : None if args.lr_steps is None else lr_scheduler.state_dict(),
                    }, is_best, run_dir)

                    #stop
                    print('Early Stopping: Stop training')
                    break

            else:
                best_followed = val_followed
                is_best = True
                counter = 0
        else:
            # remember best prec@1 and save checkpoint
            is_best = args.compare_metrics(best_followed, val_followed)
            best_followed = val_followed if is_best else best_followed

        if is_best or epoch % args.save_every == 0 or epoch == args.epochs-1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_followed,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : None if args.lr_steps is None else lr_scheduler.state_dict(),
            }, is_best, run_dir)

        if dali_is_enabled:
            train_loader.reset()
            val_loader.reset()


    print('test phase:')
    test_metrics = validate(test_loader, model, criterion, args, is_test=True)

    return test_metrics


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Prec@1', ':6.2f')
    top5 = AverageMeter('Prec@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    y_true, y_pred, = [], []
    for i, data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if dali_is_enabled:
            images = data[0]['images']
            target = data[0]["label"].squeeze(-1).long()
        else:
            if not args.gpu is None:
                images = data[0].to(args.gpu, non_blocking=True)
            else:
                images = data[0]

            if not args.gpu is None and torch.cuda.is_available():
                target = data[1].cuda(args.gpu, non_blocking=True)
            else:
                target = data[1]

        # compute output
        output = model(images)

        loss = criterion(output, target)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(train_loader) :
            print('[{0}]: '.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), end='')
            progress.display(i+1)

    f1 = f1_score(y_true, y_pred, average='weighted')*100.0
    bacc = balanced_accuracy_score(y_true, y_pred)*100.0
    return {'loss': losses.avg, 'accuracy': top1.avg, 'f1-score': f1, 'b-accuracy': bacc}

def validate(val_loader, model, criterion, args, is_test=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Prec@1', ':6.2f')
    top5 = AverageMeter('Prec@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        y_true, y_pred = [], []
        for i, data in enumerate(val_loader):

            if dali_is_enabled:
                images = data[0]['images']
                target = data[0]["label"].squeeze(-1).long()
            else:
                if not args.gpu is None:
                    images = data[0].to(args.gpu, non_blocking=True)
                else:
                    images = data[0]
                    
                if not args.gpu is None and torch.cuda.is_available():
                    target = data[1].cuda(args.gpu, non_blocking=True)
                else:
                    images = data[1]

            # compute output
            output = model(images)

            loss = criterion(output, target)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
                print('[{0}]: '.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), end='')
                progress.display(i+1)

        f1 = f1_score(y_true, y_pred, average='weighted')*100.0
        bacc = balanced_accuracy_score(y_true, y_pred)*100.0
        if is_test:
            print(' * TEST * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} F1-Score {f1:.3f} B.Acc. {bacc:.3f}'
                .format(top1=top1, 
                        top5=top5, 
                        f1=f1,
                        bacc=bacc))
        else:
            print(' * Validation * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} F1-Score {f1:.3f} B.Acc. {bacc:.3f}'
                .format(top1=top1, 
                        top5=top5, 
                        f1=f1,
                        bacc=bacc))

    return {'loss': losses.avg, 'accuracy': top1.avg, 'f1-score': f1, 'b-accuracy': bacc}


if __name__ == '__main__':
    main()
