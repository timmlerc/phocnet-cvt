'''
Created on Jun 26, 2019

@author: fwolf
'''

import logging
import copy

from sacred import Ingredient

from seg_based.datasets.iam import IAMDataset
from seg_based.datasets.gw import GWDataset
from seg_based.datasets.rimes import RimesDataset
from seg_based.datasets.hwsynth import HWSynthDataset
from seg_based.datasets.botany import BOTDataset
#from seg_based.datasets.konzils import KONDataset
from seg_based.datasets.espo import EspoDataset

from doc_analysis.transformations.homography_augmentation import HomographyAugmentation


dataset_ingredient = Ingredient('dataset')

@dataset_ingredient.config
def cfg():
    name = 'gw'
    
     # embedding
    embedding_unigrams = 'all',                        # Options: all, alnum36
    embedding_type = 'phoc',
    embedding_levels = (2,3,4,5)
    
    augmentation = 'balanced' 
    n_train_images = 500000
    
    split = '1'
    

@dataset_ingredient.capture
def load_dataset(name, split, augmentation, 
                 embedding_unigrams, embedding_type, embedding_levels):

    logger = logging.getLogger('Dataset_Ingredient::Load')
    # prepare datset loader
    logger.info('Loading dataset %s...', name)
    qry_set = None
    
    embedding_config = {
        'unigrams': embedding_unigrams[0],
        'type': embedding_type[0], 
        'levels': embedding_levels
        }

    if name == 'gw':
        data_root_dir = '/vol/corpora/document-image-analysis/gw'
        train_set = GWDataset(gw_root_dir=data_root_dir,
                              split_idx=int(split),
                              embedding_config=embedding_config)  
        
    elif name == 'iam':
        data_root_dir = '/vol/corpora/document-image-analysis/iam-db'
        train_set = IAMDataset(iam_root_dir=data_root_dir,
                               embedding_config=embedding_config)

    elif name == 'rimes':
        data_root_dir = '/vol/corpora/document-image-analysis/rimes/original/icdar2011/'
        train_set = RimesDataset(rimes_root_dir=data_root_dir,
                                 embedding_config=embedding_config)

    elif name == 'hwsynth':
        data_root_dir = '/vol/corpora/document-image-analysis/hw-synth'
        train_set = HWSynthDataset(hw_root_dir=data_root_dir,
                                   split=split,
                                   embedding_config=embedding_config)

    elif name == 'bot':
        data_root_dir = '/vol/corpora/document-image-analysis/competition_icfhr2016/'
        train_set = BOTDataset(bot_root_dir=data_root_dir,
                               split=int(split),
                               embedding_config=embedding_config)
        qry_set = copy.copy(train_set)
        qry_set.mainLoader(partition='qbe', transforms=None)
        
#    elif name == 'kon':
#        data_root_dir = '/vol/corpora/document-image-analysis/competition_icfhr2016/'
#        train_set = KONDataset(kon_root_dir=data_root_dir,
#                               embedding_config=embedding_config)
#        qry_set = copy.copy(train_set)
#        qry_set.mainLoader(partition='qbe', transforms=None)
        
    elif name == 'espo':
        data_root_dir = '/vol/corpora/document-image-analysis/esposalles'
        train_set = EspoDataset(espo_root_dir=data_root_dir,
                                embedding_config=embedding_config) 
        
    else:
        raise ValueError('Unknown Dataset!')
    
    
    # if augmentation is none do not use any transforms
    if augmentation == 'none':
        transform = None
    else:
        transform = HomographyAugmentation()

        
    # prepare partitions 
    train_set.mainLoader(partition='train', transforms=transform)
    test_set = copy.copy(train_set)
    test_set.mainLoader(partition='test', transforms=None)
    
    return train_set, test_set, qry_set

