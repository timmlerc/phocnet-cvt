# pylint: disable=too-many-locals, missing-docstring
'''
Created on Jan 7, 2016

@author: ssudholt
'''
import os
import logging
import string
import unicodedata
from xml.etree import ElementTree
import random
import sys
from PIL import Image

import cv2
import numpy as np
import scipy.io as sio
import tqdm

from utils.list_io import LineListIO
from utils.gt_reader import GroundTruthReader

from utils.parse_groundtruth import parse_readstyle_gt
from utils.union_find import UnionFind
from utils.word_list import WordList
from utils.word_container import DocImageWordContainer, SimpleWordContainer

class DatasetLoader(object):  # pylint: disable=abstract-class-not-used
    ''' class for loading various DIA datasets '''

    def __init__(self):
        pass

    @staticmethod
    def load_gw(path='/vol/corpora/document-image-analysis/gw', container_class=DocImageWordContainer):
        ''' Loads the George Washington dataset '''
        logger = logging.getLogger('DatasetLoader::GW')
        logger.info('Loading GW from %s...', path)
        # acquire all file paths
        doc_image_files = [file_name for file_name in os.listdir(os.path.join(path, 'pages'))
                           if file_name.endswith('.png')]
        ground_truth_files = [file_name for file_name in os.listdir(os.path.join(path, 'ground_truth'))
                              if file_name.endswith('.gtp')]
        if not (len(doc_image_files) == len(ground_truth_files)):
            raise ValueError('The number of document image files and ground truth files does not match')
        doc_image_files = sorted(doc_image_files)

        # load the words
        word_list = []
        gtr = GroundTruthReader(base_path='', gt_dir=os.path.join(path, 'ground_truth'),
                                gt_file_suffix='.gtp', ignore_missing_file=False)
        for doc_image_file in tqdm.tqdm(doc_image_files):
            doc_id = doc_image_file.split('/')[-1].split('.')[0]
            image_path = os.path.join(path, 'pages', doc_image_file)

            # load the ground truth
            ground_truths = gtr.read_ground_truth('/' + doc_id)
            # extract the segmented words
            for gt_idx, ground_truth in enumerate(ground_truths):
                bounding_box = {'upperLeft': np.array(ground_truth[1][0]),
                               'widthHeight': np.array(ground_truth[1][1]) - np.array(ground_truth[1][0])}
                word = container_class(transcription=ground_truth[0], bounding_box=bounding_box, image_path=image_path,
                                       page=doc_id, id_on_page=gt_idx)
                word_list.append(word)
        return word_list

    @staticmethod
    def load_rendered_gw_online(path='/vol/corpora/document-image-analysis/gw-online', stroke_width=10,
                                scale_factor=1.0, container_class=DocImageWordContainer):
        logger = logging.getLogger('DatasetLoader::GW-Online')
        logger.info('Loading GW-Online...')
        word_list = WordList()
        for img_file in tqdm.tqdm(sorted(os.listdir(os.path.join(path, 'renderedImages_sw%d_scale1.00' % stroke_width)))):
            page, _, annotation = img_file.split('.')[0].split('_')
            img_path = os.path.join(path, 'renderedImages_sw%d_scale%2.2f' % (stroke_width, scale_factor), img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            word_list.append(container_class(transcription=annotation,
                                             page=page,
                                             bounding_box={'upperLeft': np.zeros(2, dtype=np.int32),
                                                           'widthHeight': np.array((img.shape[1], img.shape[0]),
                                                                                   dtype=np.int32)},
                                             id_on_page=-1,
                                             image_path=img_path))
        return word_list

    @staticmethod
    def load_bentham_icdar2015(path='/vol/corpora/document-image-analysis/competition_icdar2015',
                               track='Track_II', container_class=SimpleWordContainer):
        logger = logging.getLogger('DatasetLoader::BenthamICDAR2015')
        if track == 'Track_I':
            logger.info('Loading Bentham (ICDAR 2015 Comp.) - Track I, Assignment A (unsupervised, seg.-based)')
            logger.warning('---\n\n!WARNING! Do not use this annotation if you are planning to use word string embeddings such as the PHOC\n\n')
            logger.warning('---')
            # load the query and test list
            query_list = WordList()
            for filename in os.listdir(os.path.join(path, 'Test_QueryByExample', 'gray')):
                query_list.append(container_class(transcription=filename,
                                                  bounding_box=dict(upperLeft=np.zeros(2, dtype=np.int32),
                                                                    widthHeight=-1 * np.ones(2, dtype=np.int32)),
                                                  image_path=os.path.join(os.path.join(path,
                                                                                       'Test_QueryByExample',
                                                                                       'gray',
                                                                                       filename))))
            test_list = WordList()
            for filename in os.listdir(os.path.join(path, 'Test_SegmentedWord_Images', 'gray')):
                test_list.append(container_class(transcription=filename,
                                                 bounding_box=dict(upperLeft=np.zeros(2, dtype=np.int32),
                                                                   widthHeight=-1 * np.ones(2, dtype=np.int32)),
                                                 image_path=os.path.join(os.path.join(path,
                                                                                      'Test_SegmentedWord_Images',
                                                                                      'gray',
                                                                                      filename))))
            # use UnionFind to find the word classes
            gt_filepath = os.path.join(path, 'Validation_GT_TrackI_AssignmentA.txt')
            union_find = UnionFind()
            with open(gt_filepath, 'r') as gt_file:
                for line in gt_file:
                    pair = line.split(' ')
                    union_find.union(pair[0], pair[1])
            # assign normalized class names
            for word in query_list:
                word.set_transcription(union_find[word.get_transcription()])
            for word in test_list:
                word.set_transcription(union_find[word.get_transcription()])
            return query_list, test_list
        elif track == 'Track_II':
            logger.info('Loading Bentham (ICDAR 2015 Comp.) - Track II (supervised)')
            raise NotImplementedError('Loading Bentham (ICDAR 2015 Comp.) Track II has not been implemented yet')
        else:
            raise ValueError('Unknown Track for ICDAR 2015 Competition')

    @staticmethod
    def load_esposalles(path='/vol/corpora/document-image-analysis/esposalles',
                        gt_lower_case=False, container_class=DocImageWordContainer):
        logger = logging.getLogger('DatasetLoader::Esposalles')
        logger.info('Loading Esposalles from %s...', path)
        # read the ground truth and partition file
        partition_list = LineListIO.read_list(os.path.join(path, 'words_officialPartitionWords.txt'))

        # acquire all file paths
        doc_image_files = [file_name for file_name in os.listdir(os.path.join(path, 'pages'))
                           if file_name.endswith('.png')]
        ground_truth_files = [file_name for file_name in os.listdir(os.path.join(path, 'patrec_ground_truth'))
                              if file_name.endswith('.gtp')]
        if not (len(doc_image_files) == len(ground_truth_files)):
            raise ValueError('The number of document image files and ground truth files does not match')
        doc_image_files = sorted(doc_image_files)

        # load the words
        word_list = []
        gtr = GroundTruthReader(base_path='', gt_dir=os.path.join(path, 'patrec_ground_truth'),
                                gt_file_suffix='.gtp', ignore_missing_file=False, gtp_encoding='latin1')
        for doc_image_file in tqdm.tqdm(doc_image_files):
            doc_id = doc_image_file.split('/')[-1].split('.')[0]
            image_path = os.path.join(path, 'pages', doc_image_file)

            # load the ground truth
            ground_truths = gtr.read_ground_truth('/' + doc_id)
            # extract the segmented words
            for gt_idx, ground_truth in enumerate(ground_truths):
                transcription = ground_truth[0]
                if gt_lower_case:
                    transcription = transcription.lower()
                bounding_box = {'upperLeft': np.array(ground_truth[1][0]),
                               'widthHeight': np.array(ground_truth[1][1]) - np.array(ground_truth[1][0])}
                word = container_class(transcription=transcription, bounding_box=bounding_box, image_path=image_path,
                                       page=doc_id, id_on_page=gt_idx)
                word_list.append(word)
        if len(word_list) != len(partition_list):
            raise ValueError('The loaded words and partition list do not match in size')
        train_list = WordList([word for word, partition_line in zip(word_list, partition_list)
                               if partition_line.split(' ')[1] == 'train'])
        val_list = WordList([word for word, partition_line in zip(word_list, partition_list)
                             if partition_line.split(' ')[1] == 'val'])
        test_list = WordList([word for word, partition_line in zip(word_list, partition_list)
                              if partition_line.split(' ')[1] == 'test'])
        return train_list, val_list, test_list

    @staticmethod
    def load_ifnenit(path, train_sets, test_sets, transcription_file=None):
        logger = logging.getLogger('DatasetLoader::IFNENIT')
        logger.info('Loading IFNENIT from %s...', path)
        if transcription_file is None:
            raise NotImplementedError('The auto loading of city zip codes is not implemented yet')
        transcription_lines = LineListIO.read_list(transcription_file)
        train_list = []
        test_list = []

        for transcrition_line in tqdm.tqdm(transcription_lines):
            file_name = transcrition_line.split('\t')[0][3:12] + 'pgm'
            transcription = transcrition_line.split('\t')[1].split(' ')
            image_path = os.path.join(path, 'set_%s/%s' % (file_name[0], file_name.split('_')[0]), file_name)
            bounding_box = {'upperLeft': np.zeros(2, dtype=np.int32),
                            'widthHeight': np.ones(2, dtype=np.int32) * -1}
            word = SimpleWordContainer(transcription=transcription, bounding_box=bounding_box,
                                       image_path=image_path)

            if file_name[0] in train_sets:
                train_list.append(word)
            elif file_name[0] in test_sets:
                test_list.append(word)
            else:
                raise ValueError('The given image file %s can not be put in either train list or test list' % (file_name))
        return train_list, test_list

    @staticmethod
    def load_iam(path='/vol/corpora/document-image-analysis/iam-db/', container_class=SimpleWordContainer):
        logger = logging.getLogger('DatasetLoader::IAM')
        logger.info('Loading IAM from %s...', path)

        # load everything
        partition_file = os.path.join(path, 'IAM_words_indexes_sets.mat')
        query_file = os.path.join(path, 'queries.gtp')
        gt_list = LineListIO.read_list(query_file)
        word_list = []
        for line in tqdm.tqdm(gt_list):
            img_id, ul_x, ul_y, lr_x, lr_y, transcription, _ = line.split(' ')
            ul_x = int(ul_x)
            ul_y = int(ul_y)
            lr_x = int(lr_x)
            lr_y = int(lr_y)
            transcription = transcription.lower()
            word = container_class(transcription=transcription,
                                   bounding_box={'upperLeft': np.array((ul_x, ul_y), dtype=np.int32),
                                                 'widthHeight': np.array((lr_x - ul_x, lr_y - ul_y), dtype=np.int32)},
                                   image_path=os.path.join(path, 'images', img_id.split('-')[0], img_id + '.png'))
            word_list.append(word)
        partition_dict = sio.loadmat(file_name=partition_file, squeeze_me=True)
        # join validation set and train set
        partition_dict['idxTrain'] = np.logical_or(partition_dict['idxTrain'],
                                                   partition_dict['idxValidation'])
        # select the partitions
        word_list = np.array(word_list)
        train_list = WordList([word for word in word_list[partition_dict['idxTrain'].astype(np.bool8)]])
        test_list = WordList([word for word in word_list[partition_dict['idxTest'].astype(np.bool8)]])
        return train_list, test_list

    @staticmethod
    def load_iamon(path='/vol/corpora/document-image-analysis/iam-db/', stroke_width=20, scale_factor=0.3):
        img_dir = 'renderedImages_sw%d_scale%2.2f' % (stroke_width, scale_factor)
        img_dir_path = os.path.join(path, img_dir)
        if not os.path.exists(img_dir_path):
            raise ValueError('The rendered IAM-OnDB does not exist for stroke_width %d and scale_factor %f' % (stroke_width, scale_factor))
        else:
            train_list = DatasetLoader.load_from_READ_xml(xml_filename='iamon_f_train.xml',
                                                          img_dir=img_dir_path,
                                                          dataset_name='iamon',
                                                          path=path)
            test_list = DatasetLoader.load_from_READ_xml(xml_filename='iamon_f_test.xml',
                                                         img_dir=img_dir_path,
                                                         dataset_name='iamon',
                                                         path=path)
            return train_list, test_list

    @staticmethod
    def load_icfhr2016_competition(dataset_name,
                                   train_set,
                                   test_annotation='SegBased',
                                   path='/vol/corpora/document-image-analysis/competition_icfhr2016/'):
        '''
        load one of the datasets from the ICFHR 2016 competition
        '''
        if dataset_name not in ['botany', 'konzilsprotokolle']:
            raise ValueError('I do not know the dataset %s' % dataset_name)
        else:
            dataset_name = dataset_name.capitalize()

        if test_annotation not in ['SegBased', 'SegFree']:
            raise ValueError('test_annotation must be one of SegBased or SegFree, found %s', test_annotation)
        # find the XML file and the image dir to be used in training
        logger = logging.getLogger('DatasetLoader::%s' % dataset_name)
        logger.info('Loading %s dataset...', dataset_name)

        # train
        train_list = DatasetLoader.load_from_READ_xml(xml_filename='_'.join((dataset_name, train_set, 'WL_CASE_INSENSITIVE.xml')),
                                                                img_dir='_'.join((dataset_name, train_set, 'PageImages')),
                                                                dataset_name=dataset_name,
                                                                path=path)

        # test
        if test_annotation == 'SegFree':
            # we must load the QbS file here as only this file has unique word image locations
            test_list = DatasetLoader.load_from_READ_xml(xml_filename='_'.join((dataset_name, 'Test_GT_SegFree_QbS.xml')),
                                                                   img_dir='_'.join((dataset_name, 'Test', 'PageImages')),
                                                                   dataset_name=dataset_name,
                                                                   path=path)
        else:
            logger.warning('!WARNING! Using filename for query and test transcriptions')
            test_list = DatasetLoader.load_from_READ_xml(xml_filename='_'.join((dataset_name, 'Test', 'WordImages', 'WL_DUMMY.xml')),
                                                                   img_dir='_'.join((dataset_name, 'Test', 'WordImages')),
                                                                   dataset_name=dataset_name,
                                                                   path=path)
        # queries
        qry_list = DatasetLoader.load_from_READ_xml(xml_filename='_'.join((dataset_name, 'Test', 'QryImages', 'WL_DUMMY.xml')),
                                                              img_dir='_'.join((dataset_name, 'Test', 'QryImages')),
                                                              dataset_name=dataset_name,
                                                              path=path)
        # parse GT...
        uf_qbe = parse_readstyle_gt(os.path.join(path, '_'.join((dataset_name, 'Test', 'GT', 'SegBased', 'QbE.xml'))))
        uf_qbs = parse_readstyle_gt(os.path.join(path, '_'.join((dataset_name, 'Test', 'GT', 'SegBased', 'QbS.xml'))))
        uf_obj = UnionFind()
        # insert all unioned elements
        for elem in list(uf_qbe.parents.items()) + list(uf_qbs.parents.items()):
            uf_obj.union(*elem)  # pylint: disable=star-args
        root_label_dict = {value: key.lower() for key, value in uf_obj.parents.items()
                           if not key.endswith('.jpg')}
        class_label_dict = {name: root_label_dict[uf_obj[name]] for name in uf_obj.parents
                            if name.endswith('.jpg')}

        # ... and set the correct labels
        if test_annotation == 'SegBased':
            for word in test_list:
                if word.get_transcription() in class_label_dict:
                    word.set_transcription(class_label_dict[word.get_transcription()])
        for word in qry_list:
            word.set_transcription(class_label_dict[word.get_transcription()])
        # return
        return train_list, test_list, qry_list

    @staticmethod
    def load_icdar2017_competition(train_set='normalized-double-height',
                                   val_set='normalized-double-height',
                                   test_set='validation',
                                   query_set='val-all',
                                   query_mode='qbs',
                                   random_query_count=100,
                                   random_seed=None,
                                   path='/vol/corpora/document-image-analysis/competition_icdar2017',
                                   remove_punctuation=False,
                                   train_as_val=False):
        ''' Loads dataset(s) of icdar2017_competition.

        Params:
            train_set: Either 'double-height', 'double-height-removed-punctuation', 
                        'double-height-alpha-numeric'.
            val_set:   Either 'double-height', 'double-height-removed-punctuation', 
                        'double-height-alpha-numeric'.
            query_set: 'val-all', 'val-random'.
            
            
            TODO some more documentation.
            TODO Shrink a bit after competition.
        '''
        logger = logging.getLogger('DatasetLoader::load_icdar2017_competition') 
        train_image_dir = 'train'
        val_image_dir = 'val'
        
        test_page_list = []
        
        # Load words for training and validation.
        train_list = DatasetLoader.load_from_READ_xml(xml_filename='train-{}.xml'.format(train_set),
                                                       img_dir='_'.join(train_image_dir),
                                                       dataset_name='icdar2017comp-{}_train'.format(train_set),
                                                       path=path)
        
        val_list = DatasetLoader.load_from_READ_xml(xml_filename='val-{}.xml'.format(val_set),
                                                           img_dir=val_image_dir,
                                                           dataset_name='icdar2017comp-{}_val'.format(val_set),
                                                           path=path)
        
        
        # Remove punctuation from transcription and then remove empty word 
        # form lists.
        if remove_punctuation:
            train_list_temp = []
            for word in train_list:
                transcription = word.transcription.strip(string.punctuation) 
                if transcription != u' ' and len(transcription) > 0:
                    word.transcription = transcription
                    train_list_temp.append(word)
            train_list = train_list_temp
            
            val_list_temp = []
            for word in val_list:
                transcription = word.transcription.strip(string.punctuation) 
                if transcription != u' ' and len(transcription) > 0:
                    word.transcription = transcription
                    val_list_temp.append(word)
            val_list = val_list_temp
            
        if train_as_val:
            val_list = [word for word in train_list]
        
        # Assume the qbe scenario first.
        if query_set == 'submission':
            queries_dir = '/vol/corpora/document-image-analysis/'\
                          'competition_icdar2017/Eval_Query_Images'
                          
            query_ids = [img_name.split('.')[0] 
                         for img_name in os.listdir(queries_dir)
                         if img_name.endswith('.jpg')]
            query_ids = sorted(query_ids)
            
            qry_list = []
            for ind, query_id in enumerate(query_ids):
                image_path = os.path.join(queries_dir, '{}.jpg'.format(query_id))
                image_descr = Image.open(image_path)
                width = image_descr.size[0]
                height = image_descr.size[1]
                bounding_box=dict(upperLeft=np.array([0, 0]),
                                  widthHeight=np.array([width, height]))
                
                qry_list.append(DocImageWordContainer(transcription=query_id,
                                                page=query_id,
                                                bounding_box=bounding_box,
                                                id_on_page=query_id,
                                                image_path=image_path))
            
            
        
        elif 'val' in query_set:
            qry_list = [word for word in val_list]
            qry_list = sorted(qry_list, key=lambda word: word.transcription)
            
            # Sample a 'pseudo' random subset of queries, control the randomness
            # with seed provided.
            if 'random' in query_set:
                random_sampler = random.Random()
                
                if random_seed is None:
                    random_seed = random.randint(0, sys.maxsize)
                random_sampler.seed(random_seed)
               
                logger.info('Loading {}/{} "random" Queries for seed: {}'
                            .format(random_query_count, len(qry_list),
                                    random_seed))
    
                #sampler = random.Random(random_seed)
                qry_list = list(qry_list) 
                random_query_count = int(random_query_count)
                qry_list = random_sampler.sample(qry_list, random_query_count)
                qry_list = sorted(qry_list, key=lambda word: word.transcription)

        
        # If query mode is qbs take the transcriptions only.
        if query_mode == 'qbs':
            if query_set == 'submission' and query_mode == 'qbs':
                logger.info('Query mode is qbs, selecting submission queries.')
                
                qry_list = LineListIO.read_list('/vol/corpora/document-image-'\
                                                'analysis/competition_icdar'\
                                                '2017/Eval_Query_Keywords.txt', 
                                                encoding='utf-8')
                qry_list = zip(qry_list, qry_list)
            
            elif query_set == 'val-example':
                logger.info('Query mode is qbs, selecting validation example '\
                            'queries.')
                
                # Example qbs queries from:
                # https://scriptnet.iit.demokritos.gr/media/databases/8b99192f55434cc7b8c94eee5292f993/examQueries.txt
                qry_list = [u'1821', u'AARGAU', u'AUCH', u'AUSSERSIHL', 
                            u'BEDINGUNG', u'BEYLAGEN', u'COMMISSION', u'DAHER',
                            u'DEZBR', u'DIESE', u'DONNERSTAGS', u'ENTSCHIEDEN', 
                            u'ERFREULICHEN', u'ERZIEHUNGSRATH', u'GEGENSTAND', 
                            u'GEGENWARTIGER', u'GRUNINGEN', u'HABEN', u'HIERVON', 
                            u'HINSICHTLICH', u'HOHEN', u'HUJUS', u'KANTONS', 
                            u'KANTONSSPITAL', u'KEINE', u'KLEINEN', u'KUSNACHT', 
                            u'KUSSNACHT', u'LANDWIRTHSCHAFT', u'LESEN', u'MESSUNG',
                            u'MOGLICH', u'NACHDEM', u'NOTIZ', u'PETITION', 
                            u'SAMSTAG', u'SICH', u'STAATSBEITRAG', u'STAATSKANZLEI', 
                            u'STRASSENZUG', u'THEILS', u'THEODOR', u'UBERWIESEN', 
                            u'VERDANKT', u'WALD', u'WIEDER', u'WOLLEN', u'WOVON', 
                            u'ZEIGT', u'ZUSCHRIFT']
                # Special ids for example queries.
                ids = ['e_{}'.format(ind) for ind in range(len(qry_list))]
                qry_list = zip(ids, qry_list)
            else:
                logger.info('Query mode is qbs, collecting unique '\
                            'transcriptions from query list set.')
                # Find unique queries.
                qry_transcriptions = [word.transcription for word in qry_list 
                                      if len(word.transcription) > 0]
                 
                qry_transcriptions = set(qry_transcriptions)
                logger.info('{} Unique transcriptions found.'
                                 .format(len(qry_transcriptions)))
    
                qry_list = zip(range(len(qry_transcriptions)),
                               sorted(list(qry_transcriptions))) 
        
        query_list = qry_list
        
        if test_set == 'validation':
            test_list = val_list
            # Sorted unique list of all test pages.
            test_page_list = sorted(list(set([word.page 
                                              for word in test_list])))
        elif test_set == 'submission':
            # We currently have no annotation for submissions test pages. 
            test_list = []
            
            test_pages_dir = '/vol/corpora/document-image-analysis/'\
                             'competition_icdar2017/EvaluationPublic'
                          
            page_ids = [img_name.split('.')[0] 
                        for img_name in os.listdir(test_pages_dir)
                        if img_name.endswith('.jpg')]
            page_ids = sorted(page_ids)
            
            test_page_list = page_ids
        else:
            raise NotImplementedError('Test set {} unknown.'.format(test_set))

        return train_list, val_list, test_list, query_list, test_page_list
        
    @staticmethod
    def load_hw_synth_10k(split, path='/vol/big/ssudholt/hw-synth'):
        '''
        Load the HW-Synth 10K dataset
        @param split: (str)
            Which split to load for the dataset. Possible: patrec, official
            If none, the dataset is returned as a whole.
        @param path: (str)
            The path to the base HW-Synth dirrectory
        '''
        logger = logging.getLogger('DatasetLoader::HW-SYNTH')
        logger.info('Loading HW-Synth 10K...')
        gt_file = np.load(os.path.join(path, 'groundtruth', 'IIIT-HWS-10K.npy'))

        word_list = np.array([SimpleWordContainer(transcription=transcription.decode('ascii'),
                                                  bounding_box={'upperLeft': np.zeros(2, dtype=np.int32),
                                                                'widthHeight': np.ones(2, dtype=np.int32) * -1},
                                                  image_path=os.path.join(path, 'Images_90K_Normalized', img_path.decode('ascii')))
                              for img_path, transcription in gt_file])
        if split == 'official':
            logger.info('Using official IIIT split for train/val')
            train_inds = np.load(os.path.join(path, 'groundtruth', 'IIIT-HWS-10K-train-indices.npy'))
            val_inds = np.load(os.path.join(path, 'groundtruth', 'IIIT-HWS-10K-val-indices.npy'))
            train_list = word_list[train_inds]
            train_list = WordList(list(train_list))
            val_list = word_list[val_inds]
            val_list = WordList(list(val_list))
            return train_list, val_list
        elif split == 'official2':
            logger.info('Using official IIIT split for train. Reduced Validation')
            train_inds = np.load(os.path.join(path, 'groundtruth', 'IIIT-HWS-10K-train-indices.npy'))
            val_inds = np.load(os.path.join(path, 'groundtruth', 'IIIT-HWS-10K-val-indices.npy'))
            val_inds = val_inds[:10000]
            train_list = word_list[train_inds]
            train_list = WordList(list(train_list))
            val_list = word_list[val_inds]
            val_list = WordList(list(val_list))
            return train_list, val_list
        elif split == 'patrec':
            # the patrec spit is simply for word spotting testing purposes
            logger.info('Using PatRec split for train/val')
            train_inds = np.where((np.arange(100000) % 30) != 0)[0]
            train_inds = np.hstack((train_inds, np.arange(100000, 1000000)))
            val_inds = np.where((np.arange(100000) % 30) == 0)[0]
            train_list = word_list[train_inds]
            train_list = WordList(list(train_list))
            val_list = word_list[val_inds]
            val_list = WordList(list(val_list))
            return train_list, val_list
        elif split == None:
            logger.info('Using no split, return dataset as a whole')
            return WordList(list(word_list))
        else:
            raise ValueError('I don\'t know the split type \'%s\'', split)
        
        
    @staticmethod
    def load_rimes(path='/vol/corpora/document-image-analysis/rimes/original/icdar2011/',
                   ignore_diacrits=True):
        ''' Loads Rimes dataset '''
        logger = logging.getLogger('DatasetLoader::Rimes')
        logger.info('Loading Rimes from %s...', path)
        
        # load trainset
        annotation_lines = LineListIO.read_list(os.path.join(path,
                                                             'word-level/ground_truth_train_icdar2011.txt'),
                                                             encoding='utf-8-sig')
            
        train_gt = [line.split(' ') for line in annotation_lines[1:]]
        
        train_gt = [(elem[0], elem[1].lower()) for elem in train_gt]
        
        if ignore_diacrits:
            train_gt = [(elem[0], unicodedata.normalize('NFKD', elem[1]).encode('ascii', 'ignore').decode('utf-8')) 
                        for elem in train_gt]
                
        train_list = [SimpleWordContainer(transcription=transcription,
                                          bounding_box={'upperLeft': np.zeros(2, dtype=np.int32),
                                                        'widthHeight': np.ones(2, dtype=np.int32) * -1},
                                          image_path=os.path.join(path,
                                                                  'word-level/trainingsnippets_icdar/training_WR',
                                                                  file_name))
                                          for file_name, transcription in train_gt]
        
        # load testset
        annotation_lines = LineListIO.read_list(os.path.join(path,
                                                             'word-level/ground_truth_test_icdar2011.txt'),
                                                             encoding='utf-8-sig')
            
        test_gt = [line.split(' ') for line in annotation_lines]
        
        test_gt = [(elem[0], elem[1].lower()) for elem in test_gt]
        
        if ignore_diacrits:
            test_gt = [(elem[0], unicodedata.normalize('NFKD', elem[1]).encode('ascii', 'ignore').decode('utf-8')) 
                        for elem in test_gt]
                
        test_list = [SimpleWordContainer(transcription=transcription,
                                         bounding_box={'upperLeft': np.zeros(2, dtype=np.int32),
                                                       'widthHeight': np.ones(2, dtype=np.int32) * -1},
                                         image_path=os.path.join(path,
                                                                 'word-level/data_test',
                                                                 file_name))
                                        for file_name, transcription in test_gt]
        
        return train_list, test_list

    @staticmethod
    def load_from_READ_xml(xml_filename, img_dir, dataset_name, path,  # pylint: disable=invalid-name
                           container_class=DocImageWordContainer,
                           decapitalize_annotation=True):
        logger = logging.getLogger('DatasetLoader::READ-XML')
        logger.debug('Using XML-File at %s and image directory %s...', xml_filename, img_dir)
        tree = ElementTree.parse(os.path.join(path, xml_filename))
        root = tree.getroot()
        # check if we have the correct XML
        if root.attrib['dataset'] != dataset_name:
            raise ValueError('The supplied training XML %s is not for the %s dataset' % (xml_filename, dataset_name))

        # iterate through all word bounding boxes and put them in a word list
        word_list = []
        for word_idx, word_elem in enumerate(root.findall('spot')):
            if decapitalize_annotation:
                transcription = word_elem.attrib['word'].lower()
            else:
                transcription = word_elem.attrib['word']
            word_list.append(container_class(transcription=transcription,
                                             page=word_elem.attrib['image'].split('.')[0],
                                             bounding_box=dict(upperLeft=np.array([int(word_elem.attrib['x']),
                                                                                   int(word_elem.attrib['y'])]),
                                                               widthHeight=np.array([int(word_elem.attrib['w']),
                                                                                     int(word_elem.attrib['h'])])),
                                             id_on_page=word_idx,
                                             image_path=os.path.join(path, img_dir, word_elem.attrib['image'])))
        return WordList(word_list)
