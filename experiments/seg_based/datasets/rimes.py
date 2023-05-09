'''
Created on May 28, 2019

@author: fwolf
'''
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

from seg_based.datasets.dataset_loader import DatasetLoader

from doc_analysis.string_embeddings.phoc import build_phoc_descriptor
from doc_analysis.string_embeddings.phoc import get_unigrams_from_strings
from doc_analysis.transformations.image_size import check_size
from doc_analysis.transformations.homography_augmentation import HomographyAugmentation


class RimesDataset(Dataset):
    '''
    PyTorch dataset class for the segmentation-based Rimes dataset
    '''

    def __init__(self,
                 rimes_root_dir,
                 embedding_config = {'unigrams': 'all', 'type': 'phoc', 'levels': (1, 2, 4, 8)},
                 ignore_diacrits = False,
                 fixed_image_size=None,
                 min_image_width_height=30):
        '''
        Constructor

        @param rimes_root_dir: full path to the Rimes root dir
        @param embedding_config: configuration of embedding (only phoc available)
                                 -unigrams: alnum36 or all
                                 -type: phoc
                                 -levels: embedding levels 
        @param ignore_diacrits: diacrits are mapped to ascii chars
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
        self.embedding_config = embedding_config
        self.fixed_image_size = fixed_image_size
        self.min_image_width_height = min_image_width_height

        # load the dataset
        self.train_list, self.test_list = DatasetLoader.load_rimes(path = rimes_root_dir,
                                                                   ignore_diacrits=ignore_diacrits)

        
        # compute string embeddings
        
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
        
        if partition not in [None, 'train', 'test']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                self.word_list = self.train_list
                self.word_string_embeddings = self.train_embeddings 
            else:
                self.word_list = self.test_list
                self.word_string_embeddings = self.test_embeddings
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
        else:
            self.query_list = np.zeros(len(self.word_list), np.int8)


    def lexicon(self):
        '''
        returns the closed lexicon (train + test)
        '''
        
        unique_word_strings = np.unique(self.label_encoder.classes_)
        
        unique_word_embeddings = build_phoc_descriptor(words=unique_word_strings,
                                                       phoc_unigrams=self.unigrams,
                                                       unigram_levels=self.embedding_config['levels']) 
        
        class_ids = self.label_encoder.transform(unique_word_strings)
     
        return unique_word_strings, unique_word_embeddings, class_ids
    
    def get_qry_strings(self):
        '''
        returns query_strings, embeddings, ids for qbs word spotting
        '''
        qry_strings = np.unique([word.get_transcription() for word in self.test_list])
        
        qry_string_embeddings = build_phoc_descriptor(words=qry_strings,
                                                      phoc_unigrams=self.unigrams,
                                                      unigram_levels=self.embedding_config['levels'])  
        
        # make sure all strings result in a valid phoc encoding (no zero strings)
        non_zero_embeddings = np.where((np.sum(qry_string_embeddings, axis=1) > 0)) 
        qry_strings = qry_strings[non_zero_embeddings]
        qry_string_embeddings = qry_string_embeddings[non_zero_embeddings]
         
        
        qry_string_ids = self.label_encoder.transform(qry_strings) 
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