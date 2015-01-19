__author__ = 'NoSyu'

"""
    Functions for handling documents
"""
import re


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
            doc_id = splitline[0]
            wordids = [int(x) for x in splitline[2::2]]
            wordcounts = [int(x) for x in splitline[3::2]]

            for wordid, wordct in zip(wordids, wordcounts):
                for each_time in wordct:
                    cur_doc.append(wordid)

            docs.append(cur_doc)

    return docs