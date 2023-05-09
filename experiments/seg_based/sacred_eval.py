'''
Created on Jun 3, 2019

@author: fwolf
'''

import logging
import copy

from sacred import Experiment

from torch.utils.data import DataLoader

from doc_analysis.evaluation.phocnet_evaluator import PHOCNet_Evaluator
from sacred_utils.mongo import load_phocnet_from_sacred_id
from sacred_utils.mongo import load_embedding_config_from_sacred_id

from seg_based.datasets.dataset_ingredient import dataset_ingredient, load_dataset

ex = Experiment('phocnet_evaluation', ingredients=[dataset_ingredient]) 

@ex.config
def phocnet_config():
    expid = 7
    model_dir = '/vol/models/fwolf/phocnet3/'
    
    test_set = 'train'
    test_split = '1'
        
    protocol = 'all'                                # Options: qbe, qbs, wr, all
    
    gpu_id = 1
    
    
    
@ex.automain
def evaluate(dataset, expid, model_dir, test_set, test_split, protocol, gpu_id):
    '''
    Evaluate PHOCNet
    
    Available Protocols: QbE, QbS, WR
    Available Datasets: George Washington, Botany, Konzilsprotokolle,
                        IAM, Rimes, HW Synth, Esposalles
    
    ''' 
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('PHOCNet-Evaluation::eval')
    
    cnn, train_set, dbsplit = load_phocnet_from_sacred_id(id=expid, model_dir=model_dir)
    embedding_config = load_embedding_config_from_sacred_id(id=expid)
    
    
    if test_set == 'train':
        test_set = train_set
        test_split = dbsplit

        
    # prepare datset loader
    logger.info('Loading dataset %s...', test_set)
    _, test_set, qry_set = load_dataset(name=test_set, split=test_split,
                                        embedding_unigrams=[embedding_config['unigrams']],
                                        embedding_type=[embedding_config['type']], 
                                        embedding_levels=embedding_config['levels'])
    
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
    

    
    
    evaluator = PHOCNet_Evaluator(cnn, test_loader, gpu_id, qry_loader)
    
    if protocol == 'qbe':
        map_qbe = evaluator.eval_qbe()
        logger.info('QbE mAP: %3.2f', map_qbe*100)
    if protocol == 'qbs':
        map_qbs = evaluator.eval_qbs()
        logger.info('QbS mAP: %3.2f', map_qbs*100)
    if protocol == 'wr':
        wer = evaluator.eval_wr()
        logger.info('Recognition WER: %3.2f', wer*100)
    if protocol == 'all':
        map_qbe, map_qbs, wer = evaluator.eval()
        logger.info('QbE mAP: %3.2f', map_qbe*100)
        logger.info('QbS mAP: %3.2f', map_qbs*100)
        logger.info('Recognition WER: %3.2f', wer*100)