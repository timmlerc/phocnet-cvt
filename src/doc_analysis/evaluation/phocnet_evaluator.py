'''
Created on Jan 9, 2019

@author: fwolf
'''

import logging
import numpy as np
import tqdm

import torch.autograd

from doc_analysis.evaluation.retrieval import map_from_query_test_feature_matrices
from doc_analysis.evaluation.recognition import er_from_query_lexicon_feature_matrices


class PHOCNet_Evaluator(object):
    '''
    Class for evaluating PHOCNets
    '''

    def __init__(self, cnn, test_loader, gpu_id, qry_loader=None):
        '''
        Constructor
        
        @param cnn: trained PHOCNet
        @param test_loader: torch dataloader providing the testset
        @param gpu_id: id of gpu to be used for evalution
        @param qry_loader: designated dataloader for qbe queries 
        '''
        self.logger = logging.getLogger('PHOCNet-Evaluation::eval')

        self.dataset_loader = test_loader
        self.qry_loader = qry_loader
        self.cnn = cnn
        self.gpu_id = gpu_id

        # move CNN to GPU
        if self.gpu_id is not None:
            cnn.cuda(self.gpu_id)

    def _compute_net_outputs(self, data_loader):
        '''
        Computes the CNN ouputs for the elements of the provided dataloader
        
        @param data_loader: torch dataloader providing the testset
        '''

        # initialize Data structures
        class_ids = np.zeros(len(data_loader), dtype=np.int32)
        embedding_size = data_loader.dataset.embedding_size()
        embeddings = np.zeros((len(data_loader), embedding_size),
                              dtype=np.float32)
        outputs = np.zeros((len(data_loader), embedding_size), dtype=np.float32)
        qry_ids = []

        # compute net outputs for test set
        for sample_idx, (word_img, embedding, class_id, is_query, _) in enumerate(tqdm.tqdm(data_loader)):
            if self.gpu_id is not None:
                # in one gpu!!
                word_img = word_img.cuda(self.gpu_id)
                embedding = embedding.cuda(self.gpu_id)

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)

            output = torch.sigmoid(self.cnn(word_img))

            outputs[sample_idx] = output.data.cpu().numpy().flatten()
            embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
            class_ids[sample_idx] = class_id.numpy()
            if is_query[0] == 1:
                qry_ids.append(sample_idx)

        qry_outputs = outputs[qry_ids][:]
        qry_class_ids = class_ids[qry_ids]

        return outputs, class_ids, qry_outputs, qry_class_ids

    def eval_qbe(self):
        '''
        Run evaluation for query-by-example word spotting
        '''
        self.logger.info('---Running QbE Evaluation---')
        self.cnn.eval()

        self.logger.info('Computing net output:')
        outputs, class_ids, qry_outputs, qry_class_ids = self._compute_net_outputs(self.dataset_loader)

        # compute net outputs for qry images (if not part of test set)
        if self.qry_loader is not None:
            _, _, qry_outputs, qry_class_ids = self._compute_net_outputs(self.qry_loader)

        # run word spotting
        self.logger.info('Computing mAP...')
        mAP, _ = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                      test_features=outputs,
                                                      query_labels=qry_class_ids,
                                                      test_labels=class_ids,
                                                      metric='cosine',
                                                      drop_first=True)

        return mAP

    def eval_qbs(self):
        '''
        Run evaluation for query-by-string word spotting
        '''
        self.logger.info('---Running QbS Evaluation---')
        self.cnn.eval()

        self.logger.info('Computing net output:')
        outputs, class_ids, _, _ = self._compute_net_outputs(self.dataset_loader)

        _, qry_phocs, qry_class_ids = self.dataset_loader.dataset.get_qry_strings()

        self.logger.info('Calculating mAP...')
        mAP, _ = map_from_query_test_feature_matrices(query_features=qry_phocs,
                                                      test_features=outputs,
                                                      query_labels=qry_class_ids,
                                                      test_labels=class_ids,
                                                      metric='cosine',
                                                      drop_first=False)

        return mAP

    def eval_wr(self):
        '''
        Run evaluation for word recognition
        '''
        self.logger.info('---Running Word Recognition Evaluation---')
        self.cnn.eval()

        _, lexicon_embeddings, lexicon_ids = self.dataset_loader.dataset.lexicon()

        self.logger.info('Computing net output:')
        outputs, class_ids, _, _ = self._compute_net_outputs(self.dataset_loader)

        wer, _ = er_from_query_lexicon_feature_matrices(query_features=outputs,
                                                        lexicon_features=lexicon_embeddings,
                                                        qry_labels=class_ids,
                                                        lexicon_labels=lexicon_ids,
                                                        metric='cosine')

        return wer

    def eval(self):
        '''
        Run evaluation for QbE, QbS and Word Recognition
        '''
        self.logger.info('---Running PHOCNet Evaluation---')
        self.cnn.eval()

        self.logger.info('Computing net output:')
        outputs, class_ids, qry_outputs, qry_class_ids = self._compute_net_outputs(self.dataset_loader)

        # QbE
        self.logger.info('Running QbE evaluation...')

        # compute net outputs for qry images (if not part of test set)
        if self.qry_loader is not None:
            _, _, qry_outputs, qry_class_ids = self._compute_net_outputs(self.qry_loader)

        mAP_qbe, _ = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                          test_features=outputs,
                                                          query_labels=qry_class_ids,
                                                          test_labels=class_ids,
                                                          metric='cosine',
                                                          drop_first=True)

        # QbS
        self.logger.info('Running QbS evaluation...')

        # get embedded query strings
        _, qry_phocs, qry_class_ids = self.dataset_loader.dataset.get_qry_strings()

        mAP_qbs, _ = map_from_query_test_feature_matrices(query_features=qry_phocs,
                                                          test_features=outputs,
                                                          query_labels=qry_class_ids,
                                                          test_labels=class_ids,
                                                          metric='cosine',
                                                          drop_first=False)

        # Word Recognition
        self.logger.info('Running Word Recognition evaluation...')

        # get lexicon embeddings
        _, lexicon_embeddings, lexicon_ids = self.dataset_loader.dataset.lexicon()

        wer, _ = er_from_query_lexicon_feature_matrices(query_features=outputs,
                                                        lexicon_features=lexicon_embeddings,
                                                        qry_labels=class_ids,
                                                        lexicon_labels=lexicon_ids,
                                                        metric='cosine')

        return mAP_qbe, mAP_qbs, wer
