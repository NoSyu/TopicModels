__author__ = 'NoSyu'

"""
    Functions for handling documents
"""
import re
import random


class Document:
    """

    """
    def __init__(self, wordids, wordcts):
        self.wordids = wordids
        self.wordcts = wordcts
        self.N = len(wordids)


def Read_BOW_docs_for_Gibbs(file_path):
    """
    Read BOW file to run topic models with Gibbs sampling
    :param file_path: The path of BOW file
    :return: documents list
    """
    splitexp = re.compile(r'[ :]')
    docs = []

    with open(file_path, 'r') as bow_file:
        for each_line in bow_file:
            splitline = splitexp.split(each_line)
            cur_doc = []
            #doc_id = splitline[0]
            wordids = [int(x) for x in splitline[2::2]]
            wordcounts = [int(x) for x in splitline[3::2]]

            for wordid, wordct in zip(wordids, wordcounts):
                for each_time in xrange(wordct):
                    cur_doc.append(wordid)

            docs.append(cur_doc)

    return docs

def Read_BOW_docs_for_VI(file_path):
    """
    Read BOW file to run topic models with Variational Inference
    :param file_path: The path of BOW file
    :return: documents list
    """
    splitexp = re.compile(r'[ :]')
    docs = []

    with open(file_path, 'r') as bow_file:
        for each_line in bow_file:
            splitline = splitexp.split(each_line)

            wordids = [int(x) for x in splitline[2::2]]
            wordcounts = [int(x) for x in splitline[3::2]]

            cur_doc = Document(wordids, wordcounts)

            docs.append(cur_doc)

    return docs


def Read_Voca_File(file_path):
    """
    Read vocabulary file
    :param file_path: The path of vocabulary file
    :return: vocabulary list
    """
    vocas = []

    with open(file_path, 'r') as voca_file:
        for each_line in voca_file:
            vocas.append(each_line.strip())

    return vocas

