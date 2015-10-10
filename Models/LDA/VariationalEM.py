import sys
import re
import time
import string
import numpy
import scipy.special


class VariationalEM:
    """
    Implements Variational EM for LDA
    """
    def __init__(self, numtopic, docs, words, alpha=0.1, eta=0.01, small_num=1e-100):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
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

    def _dirichlet_expectation(self, alpha):
        """
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
        """
        if 1 == len(alpha.shape):
            return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha))
        return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha, 1))[:, numpy.newaxis]

    def _e_step(self, e_max_iter, meanchangethresh):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = numpy.random.gamma(100., 1./100., (self._D, self._K))
        Elogtheta = self._dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)
        sstats = numpy.zeros(self._lambda.shape)

        # Now, for each document d update that document's gamma and phi
        # for d in xrange(0, batchD):
        for doc_idx, doc in enumerate(self._docs):
            # These are mostly just shorthand (but might help cache locality)
            ids = doc.wordids
            cts = doc.wordcts
            gammad = gamma[doc_idx, :]
            # Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[doc_idx, :]
            expElogbetad = self._expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = numpy.dot(expElogthetad, expElogbetad) + self._small_num

            # Iterate between gamma and phi until convergence
            for _ in xrange(e_max_iter):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * numpy.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = self._dirichlet_expectation(gammad)
                expElogthetad = numpy.exp(Elogthetad)
                phinorm = numpy.dot(expElogthetad, expElogbetad) + self._small_num
                # If gamma hasn't changed much, we're done.
                meanchange = numpy.mean(abs(gammad - lastgamma))
                if meanchange < meanchangethresh:
                    break
            gamma[doc_idx, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += numpy.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats *= self._expElogbeta

        return gamma, sstats

    def run(self, max_iter=10, e_max_iter=100, meanchangethresh=0.001, do_print_log=False):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        if do_print_log:
            prev = time.time()
            num_words_docs = 0
            for doc in self._docs:
                num_words_docs += sum(doc.wordcts)

        for idx in xrange(max_iter):
            gamma, sstats = self._e_step(e_max_iter, meanchangethresh)

            # Update lambda based on documents
            self._lambda = self._eta + sstats
            self._Elogbeta = self._dirichlet_expectation(self._lambda)
            self._expElogbeta = numpy.exp(self._Elogbeta)

            if do_print_log:
                bound = self._approx_bound(gamma)
                perwordbound = bound / num_words_docs
                print '{}\t{}\theld-out perplexity estimate:\t{}'.format(idx, time.time() - prev, numpy.exp(-perwordbound))
                prev = time.time()

    def _approx_bound(self, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        score = 0
        Elogtheta = self._dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for doc_idx, doc in enumerate(self._docs):
            # These are mostly just shorthand (but might help cache locality)
            ids = doc.wordids
            cts = doc.wordcts
            gammad = gamma[doc_idx, :]
            phinorm = numpy.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[doc_idx, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
            score += numpy.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += numpy.sum((self._alpha - gamma) * Elogtheta)
        score += numpy.sum(scipy.special.gammaln(gamma) - scipy.special.gammaln(self._alpha))
        score += sum(scipy.special.gammaln(self._alpha * self._K) - scipy.special.gammaln(numpy.sum(gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += numpy.sum((self._eta - self._lambda) * self._Elogbeta)
        score += numpy.sum(scipy.special.gammaln(self._lambda) - scipy.special.gammaln(self._eta))
        score += numpy.sum(scipy.special.gammaln(self._eta * self._W) - scipy.special.gammaln(numpy.sum(self._lambda, 1)))

        return score

