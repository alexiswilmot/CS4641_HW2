import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = False  # Set False if the covariance matrix is a diagonal matrix


class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        # prob = exp(logit) / sum(exp(logit i, d))
        maxxy = np.max(logit, axis = 1, keepdims=True)
        betterLogit = logit - maxxy
        numerator = np.exp(betterLogit)
        denom = np.sum(np.exp(betterLogit),axis=1,  keepdims=True)
        return np.divide(numerator, denom)
        #raise NotImplementedError

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        maxxy = np.max(logit, axis=1, keepdims=True)

        logitExp = np.exp(logit - maxxy)
        summed = np.sum(logitExp, axis=1, keepdims=True)
        # take log of sum
        logged = np.log(summed) + maxxy
        return logged
        # raise NotImplementedError

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        # denom = np.sqrt(2 * np.pi * sigma_i)
        # expPart = -.5 * (np.divide((points - mu_i), sigma_i)) ** 2
        # return np.divide(np.exp(expPart), denom)
        # 1 feature
        # denom = np.sqrt(2 * np.pi * (sigma_i ** 2))
        # expPart = np.exp(-(1/2) * np.multiply(np.divide(1, sigma_i**2), (points - mu_i)))
        # return np.multiply(denom, expPart)
        if points.shape[1] == 1:
            sigmaDiag = np.diagonal(sigma_i)

            # firstPart = np.sqrt(2 * np.pi * sigmaDiag)
            # subtract = points - mu_i
            # expPart = -.5 * np.square(subtract) / sigmaDiag
            # #print(firstPart * np.exp(expPart))
            # return 1/firstPart *  np.exp(expPart)


            denom = np.power( 2 * np.pi, -(points.shape[0] / 2))
            firstSig = np.power(np.linalg.det(sigmaDiag), -1/2)
            firstPart = denom * firstSig
            # now the exp part
            subtract = np.subtract(points, mu_i)
            # transposed
            subtractT = subtract.T
            # inverse of sigma
            sigmaInv = np.linalg.inv(sigmaDiag)
            # exponent part
            expPart = -.5 * np.dot(np.dot(subtract, sigmaInv, subtractT))
            # multiply them together
            return np.multiply(firstPart, np.exp(expPart))
        # multiple features
        else:

            # denom = np.power(2 * np.pi, (-self.D / 2))
            # firstSig = np.power(np.linalg.det(sigma_i), -1/2)
            # firstPart = np.multiply(denom, firstSig)
            # subtract = points - mu_i
            # sigmaInv = np.linalg.inv(sigma_i)
            # expPart = -.5 * np.einsum('ij, ij->i', np.matmul(subtract, sigmaInv), subtract)
            # #print(firstPart * np.exp(expPart))
            # return firstPart * np.exp(expPart)


            dim = points.shape[1]
            #sigma_i =  np.diagonal(sigma_i)
            # same equation as previously
            denom = np.power(2 * np.pi, (-dim/2))
            firstSig = np.power(np.linalg.det(sigma_i), -1/2)
            firstPart = denom * firstSig
            subtract = np.subtract(points, mu_i)
            sigmaInv = np.linalg.inv(sigma_i)
            expPart = -.5 * np.einsum('ij, ij->i', np.matmul(subtract, sigmaInv), subtract)
            return firstPart * np.exp(expPart)

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        raise NotImplementedError

    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        # set the prior probability pi the same for each class
        return np.full(self.K, 1/self.K)
        #raise NotImplementedError

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        # randomly select K numbers of observations as the initial 
        # mean vectors. Use int(np.random.uniform()) to get row index number of datapoints randomly
        mu = []
        for e in range(self.K):
            randoInd = int(np.random.uniform(self.N))
            mu.append(self.points[randoInd])
        return np.array(mu)
        #raise NotImplementedError

    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        # initialize covariance matrix with np.eye for each k
        #return np.array([np.eye(self.points.shape[1]) for _ in range(self.K)])
        i = np.eye(self.D)
        sigma = np.repeat(i[np.newaxis, :, :],self.K, axis=0)
        return sigma
        #raise NotImplementedError

    def _init_components(self, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5)  # Do Not Remove Seed
        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()
        return pi, mu, sigma
        #raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        # if full_matrix is True:
        # ...

        # === undergraduate implementation
        # if full_matrix is False:
        # ...
        N = self.N
        K = self.K
        #print(N)
        #print(K)
        ll = np.zeros((N, K))
        #whereZero = np.isclose(pi, 0)
        #pi[whereZero] = 1e-32
        pi = pi + LOG_CONST
        piLog = np.log(pi)
        for k in range(K):
            piK = piLog[k]
            pdfLog = np.log(self.normalPDF(self.points, mu[k], sigma[k]) + LOG_CONST)
            #pdfLog = np.log(pdf)
            ll[:, k] = piK + pdfLog
        return ll


        #raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        # if full_matrix is True:
        # ...

        # === undergraduate implementation
        # if full_matrix is False:
        #use ll joint and softmax() to get responsibilities
        FULL_MATRIX = False
        ll = self._ll_joint(pi, mu, sigma, full_matrix, **kwargs)
        return self.softmax(ll)

        # ...
        #raise NotImplementedError

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        r = np.sum(gamma, axis=0)
        # new pi
        piNew = np.divide(r, self.N)
        #muNew = np.multiply(np.power(r[:, np.newaxis], -1), np.matmul(r, self.points))
        muNew = np.divide(np.dot(self.points.T, gamma), r).T
        sigmaNew = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            xMinusMu = self.points - muNew[k]
            weightedMinus = gamma[:, k, np.newaxis] * xMinusMu
            sigmaNew[k] = np.divide(np.dot(weightedMinus.T, xMinusMu), r[k])
        # only get diagonal,create identity matrix
        i = np.eye(self.D)[np.newaxis, :, :]
        kddI = np.repeat(i, self.K, axis=0)
        sigmaNew = np.multiply(kddI, sigmaNew)
        #print("pi: ", piNew, "\n mu ", muNew,"\n sigma: ", sigmaNew)
        return piNew, muNew, sigmaNew
        #raise NotImplementedError

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the parameters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
        return gamma, (pi, mu, sigma)
