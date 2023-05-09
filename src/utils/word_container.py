#pylint: disable=missing-docstring
#pylint: disable=invalid-name
#pylint: disable=too-many-public-methods
#pylint: disable=too-many-arguments
'''
Created on Aug 29, 2014

@author: ssudholt
'''
import cv2
import numpy as np


class SimpleWordContainer(object):

    def __init__(self, transcription, bounding_box, image_path):
        '''
        Constructor
        @param transcription: str
            the transcription for this word object
        @param bounding_box: dict
            the bounding box of the word in image coordinates
            the dict needs to contain two keys: 'upperLeft' and 'widthHeight' both keys need a
            2-element ndarray (vector) containing the x,y and width/height values respectively
        @param image_path: str
            the absolute path to the word image
        '''
        self.__transcription = transcription
        self.__bounding_box = bounding_box
        self.__image_path = image_path

    def __getitem__(self, key):
        # function to make a word container compatible to Leonards implementation
        if key == 1:
            return self.get_transcription()
        elif key == 2:
            ul = self.get_bounding_box()['upperLeft']
            lr = self.get_bounding_box()['upperLeft'] + self.get_bounding_box()['widthHeight']
            return (tuple(ul), tuple(lr))
        else:
            raise ValueError('The key %s is not supported' % str(key))

    def get_transcription(self):
        return self.__transcription

    def get_bounding_box(self):
        return self.__bounding_box

    def get_image_path(self):
        return self.__image_path

    def set_transcription(self, value):
        self.__transcription = value

    def set_bounding_box(self, value):
        self.__bounding_box = value

    def set_image_path(self, value):
        self.__image_path = value

    def del_transcription(self):
        del self.__transcription

    def del_bounding_box(self):
        del self.__bounding_box

    def del_image_path(self):
        del self.__image_path

    def get_word_image(self, gray_scale=True, extend_bb=0):
        '''
        get the word image for the current word

        :param gray_scale: whether to extract the word image in gray scale (single channel)
        :param extend_bb: try to extend the bounding box by this amount of pixels
                          does not extend the BB if resulting image would be bigger than
                          page
        '''
        col_type = None
        if gray_scale:
            col_type = cv2.IMREAD_GRAYSCALE
        else:
            col_type = cv2.IMREAD_COLOR

        # extract the original bounding box coordinates
        ul = self.bounding_box['upperLeft']
        lr = ul + self.bounding_box['widthHeight']
        img = cv2.imread(self.image_path, col_type)
        if not np.all(self.bounding_box['widthHeight'] == -1):
            # try to extend the bounding box
            ul = np.maximum([0, 0], ul - extend_bb)
            lr = np.minimum(img.shape[::-1], lr + extend_bb)
            img = img[ul[1]:lr[1], ul[0]:lr[0]]
        return img

    transcription = property(get_transcription, set_transcription, del_transcription, "transcription's docstring")
    bounding_box = property(get_bounding_box, set_bounding_box, del_bounding_box, "bounding_box's docstring")
    image_path = property(get_image_path, set_image_path, del_image_path, "image_path's docstring")


class DocImageWordContainer(SimpleWordContainer):
    def __init__(self, transcription, page, bounding_box,
                 id_on_page, image_path):
        super(DocImageWordContainer, self).__init__(transcription, bounding_box, image_path)
        self.__page = page
        self.__id_on_page = id_on_page

    def get_page(self):
        return self.__page


    def get_id_on_page(self):
        return self.__id_on_page


    def set_page(self, value):
        self.__page = value


    def set_id_on_page(self, value):
        self.__id_on_page = value


    def del_page(self):
        del self.__page


    def del_id_on_page(self):
        del self.__id_on_page

    page = property(get_page, set_page, del_page, "page's docstring")
    id_on_page = property(get_id_on_page, set_id_on_page, del_id_on_page, "id_on_page's docstring")
