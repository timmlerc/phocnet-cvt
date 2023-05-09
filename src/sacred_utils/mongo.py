'''
Created on Jun 7, 2019

@author: fwolf
'''
import os

from cnn.models.myphocnet import PHOCNet
from utils.save_load import my_torch_load
from pymongo import MongoClient

def load_phocnet_from_sacred_id(id, db_host='tykwer', port='27017', db='phocnet', 
                                model_dir='/vol/models/fwolf/phocnet3'): 

    mongo_client = MongoClient('mongodb://%s:%s/' % (db_host, port))
    
    db = mongo_client[db]
    
    info = db.runs.find_one({'_id': id})['info']
    config = db.runs.find_one({'_id': id})['config']
    
    
    cnn = PHOCNet(info['embedding_size'],
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))
    
    train_set = config['dataset_name']
    file = 'PHOCNet_%s_%s.pt' % (train_set, str(id).zfill(5)) 
    my_torch_load(cnn, os.path.join(model_dir, file))
    
    return cnn, train_set, config['split']

def load_embedding_config_from_sacred_id(id, db_host='tykwer', port='27017', db='phocnet'):
    mongo_client = MongoClient('mongodb://%s:%s/' % (db_host, port))
    
    db = mongo_client[db]
    
    config = db.runs.find_one({'_id': id})['config']
   
    embedding_config = {'unigrams': config['embedding_unigrams'][0],
                        'type': config['embedding_type'][0], 
                        'levels': config['embedding_levels']}
    
    return embedding_config