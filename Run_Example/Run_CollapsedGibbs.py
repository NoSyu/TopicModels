__author__ = 'NoSyu'

from Models.LDA import CollapsedGibbs
import Utils.Documents


if __name__ == '__main__':
    """
    Run LDA by CollapsedGibbs sampling to ap news
    """
    document_file_path = '../ap_news/ap.dat'
    voca_file_path = '../ap_news/vocab.txt'

    docs = Utils.Documents.Read_BOW_docs_for_Gibbs(document_file_path)
    vocas = Utils.Documents.Read_Voca_File(voca_file_path)

    LDAModel = CollapsedGibbs(30, docs, vocas)
    LDAModel.run(max_iter=2000, do_print_log=True)
    LDAModel.ExportResultCSV('ap_news_30')
