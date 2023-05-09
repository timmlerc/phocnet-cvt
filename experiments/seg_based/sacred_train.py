'''
Created on Jan 8, 2019

@author: fwolf

'''
import logging
import os

from sacred import Experiment

import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


from seg_based.datasets.dataset_ingredient import dataset_ingredient, load_dataset
from doc_analysis.evaluation.phocnet_evaluator import PHOCNet_Evaluator

from cnn.losses.cosine_loss import CosineLoss
from cnn.models.myphocnet import PHOCNet

from utils.save_load import my_torch_save

ex = Experiment('phocnet', ingredients=[dataset_ingredient])

@ex.config
def phocnet_config():
    
    # experiment config
    exp = {
        'model_dir': '/vol/models/fwolf/phocnet3',
        'gpu_id': 0
        }
    
    # training
    training = {
        'solver_type': 'Adam',                    # Options: Adam, SGD
        'loss_type': 'BCE',                       # Options: BCE, cosine
        'learning_rate_steps': [( 80000,1e-4),
                                (100000,1e-5)],
        'momentum': 0.9,
        'momentum2': 0.999,
        'delta': 1e-8,
        'weight_decay': 0.00005,
    
        'display': 500,
        'test_interval': 2000,
        'iter_size': 10,
        'batch_size': 1
        }
    

@ex.automain
def train(dataset, exp, training, _run):
    '''
    Method for Training PHOCnets
    '''
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')

    
    # sanity checks
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        exp['gpu_id'] = None


    # prepare datset loader
    logger.info('Loading dataset %s...', dataset['name'])
    train_set, test_set, qry_set = load_dataset()

    
    # prepare augmented dataloader
    if dataset['augmentation'] == 'none':
        train_loader = DataLoader(train_set,
                                  batch_size=training['batch_size'],
                                  shuffle=True,
                                  num_workers=8)
    elif dataset['augmentation'] == 'balanced':
        random_sampler = WeightedRandomSampler(train_set.weights,
                                               dataset['n_train_images'])
        train_loader = DataLoader(train_set,
                                  sampler=random_sampler,
                                  batch_size=training['batch_size'],
                                  num_workers=8)
    elif dataset['augmentation'] == 'unbalanced':
        train_loader = DataLoader(train_set,
                                  shuffle=True,
                                  batch_size=training['batch_size'],
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
    else:
        qry_loader = None
        

    # load CNN
    logger.info('Preparing PHOCNet...')

    phoc_size = train_set.embedding_size()
    
    cnn = PHOCNet(phoc_size,
                  input_channels=1,
                  gpp_type='tpp',
                  pooling_levels=5)
    
    cnn.init_weights()


    # select loss (BCE or cosine)
    loss_selection = training['loss_type']
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')


    # select solver
    if training['solver_type'] == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(),
                                    training['learning_rate_steps'][0][1],
                                    momentum=training['momentum'],
                                    weight_decay=training['weight_decay'])

    if training['solver_type'] == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(),
                                     training['learning_rate_steps'][0][1],
                                     weight_decay=training['weight_decay'])

    # move CNN to GPU
    if exp['gpu_id'] is not None:
        cnn.cuda(exp['gpu_id'])


    # run training
    lr_cnt = 0
    max_iters = training['learning_rate_steps'][-1][0]


    optimizer.zero_grad()
    logger.info('Training:')
    for iter_idx in range(max_iters):
        if iter_idx % training['test_interval'] == 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            cnn.eval()
            evaluator = PHOCNet_Evaluator(cnn, test_loader, exp['gpu_id'], qry_loader)
            map_qbe, map_qbs, wer = evaluator.eval()
            logger.info('QbE mAP: %3.2f    QbS mAP: %3.2f    WER: %3.2f',
                        map_qbe*100, map_qbs*100, wer*100)
            _run.log_scalar('map_qbe', map_qbe*100, iter_idx+1)
            _run.log_scalar('map_qbs', map_qbs*100, iter_idx+1)
            _run.log_scalar('wer', wer*100, iter_idx+1)
            cnn.train()

        for _ in range(training['iter_size']):
            try:
                word_img, embedding, _, _, _ = train_loader_iter.next()
            except StopIteration:
                train_loader_iter = iter(train_loader)
                word_img, embedding, _, _, _ = train_loader_iter.next()
                logger.info('Resetting data loader')

            if exp['gpu_id'] is not None:
                word_img = word_img.cuda(exp['gpu_id'])
                embedding = embedding.cuda(exp['gpu_id'])

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            output = cnn(word_img)

            loss_val = loss(output, embedding)
            loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print loss
        if (iter_idx+1) % training['display'] == 0:
            logger.info('Iteration %*d: %f', len(str(max_iters)),
                        iter_idx+1, loss_val.item())
            _run.log_scalar('loss', loss_val.item(), iter_idx+1)
            _run.log_scalar('iteration', iter_idx+1, iter_idx+1)

        # change lr
        if ((iter_idx + 1) == training['learning_rate_steps'][lr_cnt][0] and
                (iter_idx+1) != max_iters):

            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = training['learning_rate_steps'][lr_cnt][1]

    # save network
    file_name = ('PHOCNet_%s_%s.pt' % (dataset['name'],
                                       str(_run._id).zfill(5)))
    my_torch_save(cnn, os.path.join(exp['model_dir'], file_name))

