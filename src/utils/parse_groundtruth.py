'''
Created on Aug 17, 2016

@author: ssudholt
'''
from xml.etree import ElementTree
from utils.union_find import UnionFind

def parse_readstyle_gt(xml_path):
    uf = UnionFind()
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    for word_elem in root.findall('spot'):
        uf.union(word_elem.attrib['image'], word_elem.attrib['word'])
    return uf
