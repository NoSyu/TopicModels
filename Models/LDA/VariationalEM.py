import time
import numpy
import scipy.special


class VariationalEM:
    """
    Implements Variational EM for LDA
    """
    def __init__(self, numtopic, docs, words, alpha=0.1, eta=0.01, small_num=1e-100):
        """
        Initialize parameters and variables
        :param numtopic: Number of topics
        :param docs: documents formed BOW
        :param words: words list
        :param alpha: alpha value
        :param eta: eta value
        :param small_num: small number for avoid divide by zero
        :return: void
        """
        self._K = numtopic
        self._words = words
        self._docs = docs

        self._W = len(self._words)
        self._D = len(self._docs)
        self._alpha = alpha
        self._eta = eta
        self._small_num = small_num

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = numpy.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = self._dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)

        self._gamma = numpy.random.gamma(100., 1./100., (self._D, self._K))

    def _dirichlet_expectation(self, alpha):
        """
        Compute E[log(theta)] given alpha
        :param alpha: alpha value
        :return: E[log(theta)]
        """
        if 1 == len(alpha.shape):
            return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha))
        return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha, 1))[:, numpy.newaxis]

    def _e_step(self, e_max_iter, meanchangethresh):
        """
        Run E step in variational EM
        :param e_max_iter: maximum iterations for e step
        :param meanchangethresh: minimum value for checking convergence
        :return: sufficient statistics for updating lambda
        """
        Elogtheta = self._dirichlet_expectation(self._gamma)
        expElogtheta = numpy.exp(Elogtheta)
        sstats = numpy.zeros(self._lambda.shape)

        for doc_idx, doc in enumerate(self._docs):
            ids = doc.wordids
            cts = doc.wordcts
            gammad = self._gamma[doc_idx, :]
            expElogthetad = expElogtheta[doc_idx, :]
            expElogbetad = self._expElogbeta[:, ids]

            phinorm = numpy.dot(expElogthetad, expElogbetad) + self._small_num

            for _ in xrange(e_max_iter):
                lastgamma = gammad
                gammad = self._alpha + expElogthetad * numpy.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = self._dirichlet_expectation(gammad)
                expElogthetad = numpy.exp(Elogthetad)
                phinorm = numpy.dot(expElogthetad, expElogbetad) + self._small_num

                meanchange = numpy.mean(abs(gammad - lastgamma))
                if meanchange < meanchangethresh:
                    break
            self._gamma[doc_idx, :] = gammad
            sstats[:, ids] += numpy.outer(expElogthetad.T, cts/phinorm)

        sstats *= self._expElogbeta

        return sstats

    def run(self, max_iter=10, e_max_iter=100, echangethresh=1e-6, emchangethresh=1e-6, do_print_log=False):
        """
        Run variational EM algorithm
        :param max_iter: maximum iterations for EM
        :param e_max_iter: maximum iterations for e step
        :param echangethresh: minimum value for checking convergence in e step
        :param emchangethresh: minimum value for checking convergence for EM
        :param do_print_log: Do we print the result on each iteration?
        :return: void
        """
        num_words_docs = 0
        last_estimated_perp = 0
        for doc in self._docs:
            num_words_docs += sum(doc.wordcts)
        if do_print_log:
            prev = time.time()

        for idx in xrange(max_iter):
            sstats = self._e_step(e_max_iter, echangethresh)

            self._lambda = self._eta + sstats
            self._Elogbeta = self._dirichlet_expectation(self._lambda)
            self._expElogbeta = numpy.exp(self._Elogbeta)

            bound = self._approx_bound() / num_words_docs
            estimated_perp = numpy.exp(-bound)
            meanchange = numpy.mean(abs(estimated_perp - last_estimated_perp))
            last_estimated_perp = estimated_perp

            if do_print_log:
                print '{}\t{}\theld-out perplexity estimate:\t{}'.format(idx, time.time() - prev, estimated_perp)
                prev = time.time()

            if meanchange < emchangethresh:
                break

    def _approx_bound(self):
        """
        Approximate the bound for perplexity
        :return: bound
        """
        score = 0
        Elogtheta = self._dirichlet_expectation(self._gamma)
        expElogtheta = numpy.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for doc_idx, doc in enumerate(self._docs):
            ids = doc.wordids
            cts = doc.wordcts
            phinorm = numpy.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[doc_idx, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
            score += numpy.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += numpy.sum((self._alpha - self._gamma) * Elogtheta)
        score += numpy.sum(scipy.special.gammaln(self._gamma) - scipy.special.gammaln(self._alpha))
        score += sum(scipy.special.gammaln(self._alpha * self._K) - scipy.special.gammaln(numpy.sum(self._gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += numpy.sum((self._eta - self._lambda) * self._Elogbeta)
        score += numpy.sum(scipy.special.gammaln(self._lambda) - scipy.special.gammaln(self._eta))
        score += numpy.sum(scipy.special.gammaln(self._eta * self._W) - scipy.special.gammaln(numpy.sum(self._lambda, 1)))

        return score

    def ExportResultCSV(self, output_file_name, rank_idx=100):
        """
        Export Algorithm Result to File
        :param output_file_name: output file name
        :param rank_idx: how many topics are printed?
        :return: void
        """
        # Raw data
        numpy.savetxt("{}_gamma.csv".format(output_file_name), self._gamma, delimiter=",")
        numpy.savetxt("{}_lambda.csv".format(output_file_name), self._lambda, delimiter=",")

        # Ranked data
        with open("{}_topics_Ranked.csv".format(output_file_name), "w") as ranked_topic_word_file:
            norm_lambda = self._lambda / self._lambda.sum(axis=1)[:, numpy.newaxis]
            row_idx = -1

            for each_row in norm_lambda:
                row_idx += 1
                enum_each_row = enumerate(each_row)
                ranked_one = sorted(enum_each_row, key=lambda x: x[1], reverse=True)
                temp_str = 'topic {}\t'.format(row_idx)
                for i in xrange(rank_idx):
                    temp_str += '%s %.6f\t' % (self._words[ranked_one[i][0]], ranked_one[i][1])
                print >>ranked_topic_word_file, temp_str

