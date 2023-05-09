'''
Created on Jun 3, 2019

@author: fwolf
'''

import logging
import argparse
import copy

from torch.utils.data import DataLoader

from doc_analysis.evaluation.phocnet_evaluator import PHOCNet_Evaluator
from cnn.models.myphocnet import PHOCNet
from utils.save_load import my_torch_load

from seg_based.datasets.iam import IAMDataset
from seg_based.datasets.gw import GWDataset
from seg_based.datasets.rimes import RimesDataset
from seg_based.datasets.hwsynth import HWSynthDataset
from seg_based.datasets.botany import BOTDataset

def load_phocnet(path, phoc_size=540):
    '''
    Loads standard PHOCNet Architecture
    
    @param path: path to the weight file (.pt)
    @param phoc_size: size of the embedding size (number of network outputs)

    '''
    cnn = PHOCNet(phoc_size,
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))

    cnn.init_weights()
        
    my_torch_load(cnn, path)
    return cnn

def load_dataset(dataset, split=None):
    '''
    Loads test dataset for evaluating word spotting and recogntion.
    If available a query loader is initialiazed.
    
    @param dataset: name of the test set. Options: gw, iam, rimes, bot, hwsynth
    @param split: if available defines split of the dataset 
     
    '''
    
    
    qry_loader = None

    if dataset == 'gw':
        data_root_dir = '/vol/corpora/document-image-analysis/gw'
        test_set = GWDataset(gw_root_dir=data_root_dir,
                              split_idx=int(split))
                            
    if dataset == 'iam':
        data_root_dir = '/vol/corpora/document-image-analysis/iam-db'
        test_set = IAMDataset(iam_root_dir=data_root_dir,
                              remove_punctuation=True)

    if dataset == 'rimes':
        data_root_dir = '/vol/corpora/document-image-analysis/rimes/original/icdar2011/'
        test_set = RimesDataset(rimes_root_dir=data_root_dir,
                                ignore_diacrits=True)

    if dataset == 'hwsynth':
        data_root_dir = '/vol/corpora/document-image-analysis/hw-synth'
        test_set = HWSynthDataset(hw_root_dir=data_root_dir,
                                   split=split)
    if dataset == 'bot':
        data_root_dir = '/vol/corpora/document-image-analysis/competition_icfhr2016/'
        test_set = BOTDataset(bot_root_dir=data_root_dir)

        qry_set = copy.copy(test_set)
        qry_set.mainLoader(partition='qbe', transforms=None)
        qry_loader = DataLoader(qry_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8)
        
    test_set.mainLoader(partition='test', transforms=None)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8)
    
    return test_loader, qry_loader
        
    

def evaluate():
    '''
    Evaluate standard PHOCNet
    
    Available Protocols: QbE, QbS, WR
    Available Datasets: George Washington, Botany, IAM, Rimes, HW Synth
    
    ''' 
    logger = logging.getLogger('PHOCNet-Evaluation::eval')
    # argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '-p', action='store', type=str,
                        default='/vol/models/fwolf/phocnet/PHOCNet_gw1_1001.pt',
                        help='The path of the model file')
    parser.add_argument('--dataset', '-ds', choices=['gw', 'iam', 'hwsynth', 'rimes', 'bot'], default='gw',
                        help='The dataset to be trained on')
    parser.add_argument('--split', '-s', action='store', type=str, default='1',
                        choices=['1','2','3','4','patrec', 'official'],
                        help='The split of the dataset. Default: 1')
    parser.add_argument('--protocol', '-prot', choices=['qbe', 'qbs', 'wr', 'all'], default='all',
                        help='The dataset to be trained on')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type = int, default='5',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
     
    args = parser.parse_args()
    
    
    test_loader, qry_loader = load_dataset(args.dataset, args.split)
    cnn = load_phocnet(args.path, test_loader.dataset.embedding_size())
    
    evaluator = PHOCNet_Evaluator(cnn, test_loader, args.gpu_id, qry_loader)
    
    if args.protocol == 'qbe':
        map_qbe = evaluator.eval_qbe()
        logger.info('QbE mAP: %3.2f', map_qbe*100)
    if args.protocol == 'qbs':
        map_qbs = evaluator.eval_qbs()
        logger.info('QbS mAP: %3.2f', map_qbs*100)
    if args.protocol == 'wr':
        wer = evaluator.eval_wr()
        logger.info('Recognition WER: %3.2f', wer*100)
    if args.protocol == 'all':
        map_qbe, map_qbs, wer = evaluator.eval()
        logger.info('QbE mAP: %3.2f', map_qbe*100)
        logger.info('QbS mAP: %3.2f', map_qbs*100)
        logger.info('Recognition WER: %3.2f', wer*100)
        
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    evaluate()