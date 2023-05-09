'''
Created on Jan 8, 2019

@author: fwolf

'''
import argparse
import logging
import os
import copy

import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from seg_based.datasets.iam import IAMDataset
from seg_based.datasets.gw import GWDataset
from seg_based.datasets.rimes import RimesDataset
from seg_based.datasets.hwsynth import HWSynthDataset
from seg_based.datasets.botany import BOTDataset

from doc_analysis.evaluation.phocnet_evaluator import PHOCNet_Evaluator
from doc_analysis.transformations.homography_augmentation import HomographyAugmentation

from cnn.losses.cosine_loss import CosineLoss
from cnn.models.myphocnet import PHOCNet

from utils.save_load import my_torch_save


def learning_rate_step_parser(lrs_string):
    '''
    Parser learning rate string
    '''
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for
            elem in lrs_string.split(',')]


def train():
    '''
    Method for Training PHOCnets
    '''
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')
    # argument parsing
    parser = argparse.ArgumentParser()

    # -misc    
    parser.add_argument('--model_dir', '-mdir', action='store', type=str,
                        default='/vol/models/fwolf/phocnet3',
                        help='The path of the where to save trained models' + \
                             'Default: /vol/models/fwolf/phocnet3')
    parser.add_argument('--experiment_id', '-exp_id', action='store', type=int, default=3333,
                        help='The Experiment ID. Default: Based on Model directory')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=int, default=None,
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')

    # - dataset arguments
    parser.add_argument('--dataset', '-ds', choices=['gw', 'iam', 'hwsynth', 'rimes', 'bot'], default='gw',
                        help='The dataset to be trained on')
    parser.add_argument('--split', '-s', action='store', type=str, default='1',
                        choices=['1', '2', '3', '4', 'patrec', 'official'],
                        help='The split of the dataset. Default: 1')
    parser.add_argument('--augmentation', '-aug', choices=['none', 'balanced', 'unbalanced'], default='balanced',
                        help='Data augmentation type')
    parser.add_argument('--n_train_images', '-nti', action='store', type=int, default=500000,
                        help='The number of training images. Default: 500000')
    parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default=None,
                        help='Specifies the images to be resized to a fixed size when presented to the CNN.' + \
                             ' Argument must be two comma seperated numbers.')
    parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN.' + \
                             'Default: 26')

    # - embedding arguments
    parser.add_argument('--unigrams', '-u', action='store',
                        choices=['alnum36', 'all'],
                        default='alnum36',
                        help='Which unigrams to create the embedding.' + \
                             'Possible: alnum36, all. Default: alnum36')
    parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc. Default: phoc')
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')

    # - train arguments
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--loss_type', '-lt', choices=['BCE', 'cosine'], default='BCE',
                        help='The Type of loss function')
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser,
                        default='60000:1e-4,100000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')

    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=10,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')

    args = parser.parse_args()

    # sanity checks
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters (ID %i):', args.experiment_id)
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader
    logger.info('Loading dataset %s...', args.dataset)
    qry_set = None
    qry_loader = None

    # config dict for embeddings
    embedding_config = {'unigrams': args.unigrams,
                        'type': args.embedding_type,
                        'levels': args.phoc_unigram_levels}

    if args.dataset == 'gw':
        data_root_dir = '/vol/corpora/document-image-analysis/gw'
        train_set = GWDataset(gw_root_dir=data_root_dir,
                              split_idx=int(args.split),
                              embedding_config=embedding_config)

    if args.dataset == 'iam':
        data_root_dir = '/vol/corpora/document-image-analysis/iam-db'
        data_root_dir = '/data/eugen/datasets/iam-db'
        train_set = IAMDataset(iam_root_dir=data_root_dir,
                               embedding_config=embedding_config)

    if args.dataset == 'rimes':
        data_root_dir = '/vol/corpora/document-image-analysis/rimes/original/icdar2011/'
        train_set = RimesDataset(rimes_root_dir=data_root_dir)

    if args.dataset == 'hwsynth':
        data_root_dir = '/vol/corpora/document-image-analysis/hw-synth'
        train_set = HWSynthDataset(hw_root_dir=data_root_dir,
                                   split=args.split,
                                   embedding_config=embedding_config)

    if args.dataset == 'bot':
        data_root_dir = '/vol/corpora/document-image-analysis/competition_icfhr2016/'
        train_set = BOTDataset(bot_root_dir=data_root_dir,
                               embedding_config=embedding_config)
        qry_set = copy.copy(train_set)
        qry_set.mainLoader(partition='qbe', transforms=None)

    # if augmentation is none do not use any transforms
    if args.augmentation == 'none':
        transform = None
    else:
        transform = HomographyAugmentation()

    # prepare partitions
    train_set.mainLoader(partition='train', transforms=transform)
    test_set = copy.copy(train_set)
    test_set.mainLoader(partition='test', transforms=None)

    # prepare augmented dataloader
    if args.augmentation == 'none':
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=8)
    elif args.augmentation == 'balanced':
        random_sampler = WeightedRandomSampler(train_set.weights,
                                               args.n_train_images)
        train_loader = DataLoader(train_set,
                                  sampler=random_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=8)
    elif args.augmentation == 'unbalanced':
        train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=8)

    train_loader_iter = iter(train_loader)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8)

    if qry_set is not None:
        qry_loader = DataLoader(qry_set,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)

    # load CNN
    logger.info('Preparing PHOCNet...')

    phoc_size = train_set.embedding_size()

    cnn = PHOCNet(phoc_size,
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))

    cnn.init_weights()

    # select loss (BCE or cosine)
    loss_selection = args.loss_type
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=False)
    else:
        raise ValueError('not supported loss function')

    # select solver
    if args.solver_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(),
                                    args.learning_rate_step[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(),
                                     args.learning_rate_step[0][1],
                                     weight_decay=args.weight_decay)

    # move CNN to GPU
    if args.gpu_id is not None:
        cnn.cuda(args.gpu_id)

    # run training
    lr_cnt = 0
    max_iters = args.learning_rate_step[-1][0]

    optimizer.zero_grad()
    logger.info('Training:')
    for iter_idx in range(max_iters):
        if iter_idx % args.test_interval == 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            cnn.eval()
            evaluator = PHOCNet_Evaluator(cnn, test_loader, args.gpu_id, qry_loader)
            map_qbe, map_qbs, wer = evaluator.eval()
            logger.info('QbE mAP: %3.2f    QbS mAP: %3.2f    WER: %3.2f',
                        map_qbe * 100, map_qbs * 100, wer * 100)
            cnn.train()

        for _ in range(args.iter_size):
            try:
                word_img, embedding, _, _, _ = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                word_img, embedding, _, _, _ = train_loader_iter.next()
                logger.info('Resetting data loader')

            if args.gpu_id is not None:
                word_img = word_img.cuda(args.gpu_id)
                embedding = embedding.cuda(args.gpu_id)

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            output = cnn(word_img)

            loss_val = loss(output, embedding)
            loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print loss
        if (iter_idx + 1) % args.display == 0:
            logger.info('Iteration %*d: %f', len(str(max_iters)),
                        iter_idx + 1, loss_val.item())

        # change lr
        if ((iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx + 1) != max_iters):
            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate_step[lr_cnt][1]

    # save network
    file_name = ('PHOCNet_%s_%s.pt' % (args.dataset,
                                       str(args.experiment_id).zfill(4)))
    my_torch_save(cnn, os.path.join(args.model_dir, file_name))


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    train()
