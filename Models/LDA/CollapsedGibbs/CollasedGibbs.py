import time
import numpy
from scipy.special import gammaln, psi

__author__ = 'NoSyu'


class CollasedGibbs:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003

    Latent Dirichlet allocation with collapsed Gibbs sampling

    Original code is coming from
    https://github.com/arongdari/python-topic-model
    """
    def __init__(self, numtopic, docs, words, alpha=0.1, beta=0.01):
        """

        :param numtopic:
        :param docs:
        :param words:
        :param alpha:
        :param beta:
        :return:
        """
        self.docs = docs
        self.words = words
        self.K = numtopic
        self.D = len(docs)
        self.W = len(words)

        # Hyper-parameters
        self.alpha = alpha
        self.beta = beta

        self.WK = numpy.zeros([self.W, self.K]) + self.beta
        self.sumK = numpy.zeros([self.K]) + self.beta * self.W
        self.doc_topic_sum = numpy.zeros([self.D, self.K]) + self.alpha

        # Random initialization of topics
        self.doc_topics = list()

        for di in xrange(self.D):
            doc = self.docs[di]
            topics = numpy.random.randint(self.K, size=len(doc))
            self.doc_topics.append(topics)

            for wi in xrange(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                self.WK[word, topic] += 1
                self.sumK[topic] += 1
                self.doc_topic_sum[di, topic] += 1

    def gibbs_sampling(self, max_iter):
        """
        Argument:
        max_iter:
        """
        prev = time.clock()

        for iteration in xrange(max_iter):

            print iteration, time.clock() - prev, self.loglikelihood()
            prev = time.clock()

            for di in xrange(self.D):
                doc = self.docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.doc_topics[di][wi]

                    self.WK[word, old_topic] -= 1
                    self.sumK[old_topic] -= 1
                    self.doc_topic_sum[di, old_topic] -= 1

                    #update
                    prob = ((self.WK[word, :]) / (self.sumK[:]))\
                           * (self.doc_topic_sum[di, :])

                    #new_topic = sampling_from_dist(prob)
                    prob_sum = prob.sum()
                    prob = prob / prob_sum
                    new_topic = (numpy.random.multinomial(1, prob))[0]

                    self.doc_topics[di][wi] = new_topic
                    self.WK[word,new_topic] += 1
                    self.sumK[new_topic] += 1
                    self.doc_topic_sum[di, new_topic] += 1

    def loglikelihood(self):
        """
        Compute log likelihood function
        :return: log likelihood function
        """
        return self._topic_loglikelihood() + self._document_loglikelihood()

    def _topic_loglikelihood(self):
        """
        Compute log likelihood by topics
        :return: log likelihood by topics
        """
        ll = self.K * gammaln(self.beta * self.W)
        ll -= self.K * self.W * gammaln(self.beta)

        for ki in xrange(self.K):
            ll += gammaln(self.WK[:, ki]).sum() - gammaln(self.WK[:, ki].sum())

        return ll

    def _document_loglikelihood(self):
        """
        Compute log likelihood by documents
        :return: log likelihood by documents
        """
        ll = self.D * gammaln(self.alpha * self.K)
        ll -= self.D * self.K * gammaln(self.alpha)

        for di in xrange(self.D):
            ll += gammaln(self.doc_topic_sum[di, :]).sum() - gammaln(self.doc_topic_sum[di, :].sum())

        return ll

    def _optimize(self):
        """
        Optimize hyperparameters
        :return: void
        """
        self._alphaoptimize()
        self._betaoptimize()

    def _alphaoptimize(self, conv_threshold=0.001):
        """
        Optimize alpha vector
        :return: void
        """
        is_converge = False
        old_ll = self._topic_loglikelihood()

        while not is_converge:
            alpha_sum = self.alpha.sum()
            alpha_temp = numpy.zeros([self.K])

            for topic_idx in xrange(self.K):
                numerator = psi(self.doc_topic_sum[:, topic_idx] + self.alpha[topic_idx]).sum() - (self.D * psi(self.alpha[topic_idx]))
                denominator = psi(map(self.docs, lambda x: len(x)) + alpha_sum).sum() - (self.D * psi(alpha_sum))

                alpha_temp[topic_idx] = self.alpha[topic_idx] * (numerator / denominator)

                if alpha_temp[topic_idx] <= 0:
                    return

            self.alpha = alpha_temp

            new_ll = self._topic_loglikelihood()

            if abs(new_ll - old_ll) < conv_threshold:
                is_converge = True
            else:
                old_ll = new_ll

    def _betaoptimize(self, conv_threshold=0.001):
        """
        Optimize beta value
        :return: void
        """
        is_converge = False
        old_ll = self._document_loglikelihood()

        while not is_converge:
            beta_sum = self.beta * self.W
            #beta_temp = numpy.zeros([self.K])
            numerator = 0
            denominator = 0

            for topic_idx in xrange(self.K):
                numerator += psi(self.WK[:, topic_idx] + self.beta).sum()
                denominator += psi(self.sumK[topic_idx] + beta_sum)

            numerator -= self.W * self.K * psi(self.beta)
            denominator -= self.K * psi(beta_sum)
            denominator *= self.W

            beta_temp = self.beta * (numerator / denominator)

            if beta_temp <= 0:
                return

            self.beta = beta_temp

            new_ll = self._document_loglikelihood()

            if abs(new_ll - old_ll) < conv_threshold:
                is_converge = True
            else:
                old_ll = new_ll

    def ExportResultCSV(self, output_file_name, rank_idx=100):
        """
        Export Algorithm Result to File
        :return: void
        """
        # Raw data
        numpy.savetxt("WK_%s.csv" % output_file_name, self.WK, delimiter=",")
        numpy.savetxt("doc_topic_sum_%s.csv" % output_file_name, self.doc_topic_sum, delimiter=",")

        # Ranked data
        with open("KW_Ranked_%s.csv" % output_file_name, "w") as ranked_topic_word_file:
            for topic_idx in xrange(self.K):
                temp_pair = zip(self.WK[:, topic_idx], xrange(self.W))
                temp_sorted_pair = sorted(temp_pair, key=lambda x: x[0], reverse=True)
                temp_str = 'topic %d,' % topic_idx
                for idx in xrange(rank_idx):
                    temp_str += '%s %.6f,' % (self.words[temp_sorted_pair[idx][1]], temp_sorted_pair[idx][0])
                print >>ranked_topic_word_file, temp_str


if __name__ == '__main__':
    #test
    docs = [[0,1,2,3,3,4,3,4,5], [2,3,3,5,6,7,8,3,8,9,5]]

    model = CollasedGibbs(2, 10)
    model.random_init(docs)
    model.gibbs_sampling(100,docs)
