'''
Created on Mar 19, 2015

@author: ssudholt
'''
import logging

class WordList(object):
    '''
    this class holds a list of segmented words
    standard operations can be run on this data structure
    '''
    def __init__(self, precomputed_list=None):
        self.logger = logging.getLogger('WordList')
        if precomputed_list is not None:
            self.__internal_list = precomputed_list
        else:
            self.__internal_list = []

    def __getitem__(self, item):
        return self.__internal_list[item]

    def __iter__(self):
        return iter(self.__internal_list)

    def __len__(self):
        return len(self.__internal_list)

    def __add__(self, x):
        return WordList(self.__internal_list + x.get_internal_list())

    def get_internal_list(self):  # pylint: disable=missing-docstring
        return self.__internal_list

    def itervalues(self):  # pylint: disable=missing-docstring
        return iter(self.__internal_list)

    def append(self, item):  # pylint: disable=missing-docstring
        self.__internal_list.append(item)

    def clear(self):
        self.__internal_list[:] = []