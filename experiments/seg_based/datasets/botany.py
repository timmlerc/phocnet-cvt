'''
Created on Sep 3, 2017

@author: ssudholt
'''
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

from utils.list_io import LineListIO
from seg_based.datasets.dataset_loader import DatasetLoader

from doc_analysis.string_embeddings.phoc import build_phoc_descriptor
from doc_analysis.string_embeddings.phoc import get_unigrams_from_strings
from doc_analysis.transformations.image_size import check_size
from doc_analysis.transformations.homography_augmentation import HomographyAugmentation



class BOTDataset(Dataset):
    '''
    PyTorch dataset class for the segmentation-based Botany dataset
    '''

    def __init__(self, 
                 bot_root_dir,
                 split=1,
                 embedding_config = {'unigrams': 'alnum36', 'type': 'phoc', 'levels': (1, 2, 4, 8)},
                 fixed_image_size=None,
                 min_image_width_height=30):
        '''
        Constructor

        @param bot_root_dir: full path to the Botany root dir
        @param embedding_config: configuration of embedding (only phoc available)
                                 -unigrams: alnum36 or all
                                 -type: phoc
                                 -levels: embedding levels 
        @param fixed_image_size: resize images to a fixed size
        @param min_image_width_height: the minimum height or width a word image
                                       has to have
        '''
        
        # sanity checks
        if embedding_config['unigrams'] not in ['alnum36', 'all']:
            raise ValueError('Unknown unigram definition')
        if embedding_config['type'] not in ['phoc']:
            raise ValueError('embedding must be phoc')
        
        # class members
        self.bot_root_dir = bot_root_dir
        self.embedding_config = embedding_config
        self.fixed_image_size = fixed_image_size
        self.min_image_width_height = min_image_width_height


        # load the dataset
        if split == 1:
            train_period = 'I'
        elif split == 2:
            train_period = 'II'
        elif split == 3:
            train_period = 'III'

        self.train_list, self.test_list, self.qbe_list = DatasetLoader.load_icfhr2016_competition(dataset_name='botany',
                                                                                                  train_set='Train_%s' % train_period,
                                                                                                  test_annotation='SegBased',
                                                                                                  path=bot_root_dir)
        

        

        # extract unigrams from train split
        if embedding_config['unigrams'] == 'alnum36':
            self.unigrams = [chr(i) for i in np.hstack([np.arange(ord('a'), ord('z') + 1),
                                                        np.arange(ord('0'), ord('9') + 1)])]
        elif embedding_config['unigrams'] == 'all':
            self.unigrams = get_unigrams_from_strings([word.get_transcription() 
                                                       for word in self.train_list + self.test_list]) 
        else:
            raise ValueError('Unknown unigram type')
        
        if embedding_config['type'] == 'phoc':
            self.train_embeddings = build_phoc_descriptor(words = [word.get_transcription() for word in self.train_list], 
                                                          phoc_unigrams=self.unigrams, 
                                                          unigram_levels=embedding_config['levels']) 
            self.test_embeddings = build_phoc_descriptor(words = [word.get_transcription() for word in self.test_list], 
                                                          phoc_unigrams=self.unigrams, 
                                                          unigram_levels=embedding_config['levels']) 
        else:
            raise ValueError('Unknown embedding type')
        
        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([word.get_transcription() for word in self.train_list + self.test_list])
        
        
    
    def mainLoader(self, partition=None, transforms=HomographyAugmentation()):
        '''
        Initializes Dataloader for desired partition
        '''
        self.transforms = transforms
        
        if partition not in [None, 'train', 'test', 'qbe']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                self.word_list = self.train_list
                self.word_string_embeddings = self.train_embeddings 
            elif partition == 'test':
                self.word_list = self.test_list
                self.word_string_embeddings = self.test_embeddings
            elif partition == 'qbe':
                self.word_list = self.qbe_list
                # !WARNING! Word string embeddings for Query Set are zero vectors!!!
                self.word_string_embeddings = np.zeros((len(self.qbe_list), self.embedding_size()), dtype=np.float32)
        else:
            # use the entire dataset
            self.word_list = self.train_list + self.word_list
            self.word_string_embeddings = self.train_embeddings_embeddings + self.test_embeddings
            
            
        # weights for sampling
        if partition == 'train':
            train_strings = [word.get_transcription() for word in self.train_list]
            unique_word_strings, counts = np.unique(train_strings,
                                                    return_counts=True)
            ref_count_strings = {uword : count for uword, count
                                 in zip(unique_word_strings, counts)}
            weights = [1.0/ref_count_strings[word] for word in train_strings]
            self.weights = np.array(weights)/sum(weights)


        # create queries
        if partition == 'test':
            test_strings = [word.get_transcription() for word in self.test_list]
            unique_word_strings, counts = np.unique(test_strings,
                                                    return_counts=True)
            qry_word_strings = unique_word_strings[np.where(counts > 1)[0]]

            query_list = np.zeros(len(self.test_list), np.int8)
            qry_ids = [i for i in range(len(self.test_list))
                       if test_strings[i] in qry_word_strings]
            query_list[qry_ids] = 1

            self.query_list = query_list
        elif partition == 'qbe':
            self.query_list = np.ones_like(self.word_list)
        else:
            self.query_list = np.zeros_like(self.word_list)
            
    def lexicon(self):
        '''
        returns the closed lexicon (train + test)
        '''
        word_strings = np.hstack(([word.get_transcription() for word in self.train_list],
                                  [word.get_transcription() for word in self.qbe_list]))
        unique_word_strings = np.unique(word_strings)
        
        unique_word_embeddings = build_phoc_descriptor(words=unique_word_strings,
                                                       phoc_unigrams=self.unigrams,
                                                       unigram_levels=self.embedding_config['levels']) 
        
        
        # make sure all strings result in a valid phoc encoding (no zero strings)
        non_zero_embeddings = np.where((np.sum(unique_word_embeddings, axis=1) > 0))
    
        unique_word_strings = unique_word_strings[non_zero_embeddings]
        unique_word_embeddings = unique_word_embeddings[non_zero_embeddings]
        
          
        class_ids = self.label_encoder.transform(unique_word_strings)
        
        return unique_word_strings, unique_word_embeddings, class_ids
            
    def get_qry_strings(self):
        '''
        returns query_strings, embeddings, ids for qbs word spotting
        '''
        
        # load predefined queries
        qry_string_path = self.bot_root_dir + 'Botany_Test_QryStrings.lst'
        qry_string_lines = LineListIO.read_list(qry_string_path)
        qry_strings = [q.lower() for q in qry_string_lines]
        
        qry_string_ids = self.label_encoder.transform(qry_strings) 
        
        qry_string_embeddings = build_phoc_descriptor(words=qry_strings,
                                                      phoc_unigrams=self.unigrams,
                                                      unigram_levels=self.embedding_config['levels'])  
        
        return qry_strings, qry_string_embeddings, qry_string_ids
    
    def embedding_size(self):
        '''
        returns length of the embedding
        '''
        return len(self.train_embeddings[0])

    def unigrams(self):
        return self.unigrams

    def __len__(self):
        '''
        returns length of the dataset. Partition is required to be initialized
        '''
        return len(self.word_list)


    def __getitem__(self, index): 
        word_img = self.word_list[index].get_word_image()
        word_img = 1 - word_img.astype(np.float32) / 255.0

        transcription = self.word_list[index].get_transcription()
        class_id = self.label_encoder.transform([transcription])[0]
        is_query = self.query_list[index]
        
        # augmentation
        if self.transforms is not None:
            word_img = self.transforms(word_img)

        # check image size
        word_img = check_size(word_img, self.min_image_width_height, self.fixed_image_size)

        # prepare data for torch
        word_img = word_img.reshape((1,) + word_img.shape)
        word_img = torch.from_numpy(word_img.astype(np.float32))
        embedding = self.word_string_embeddings[index]
        embedding = torch.from_numpy(embedding.astype(np.float32))

        return word_img, embedding, class_id, is_query, transcription
    
    
    