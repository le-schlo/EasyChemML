import numpy as np

class RegressionMetrics:

    def __init__(self, true_e_t, pred_e_t):
        self.true_value = np.array(true_e_t)
        self.pred_value = np.array(pred_e_t)

    def MSE(self) -> float:
        """
        AbsoluteError for triplet energy: True - Predicted
        """

        return np.mean(self.true_value - self.pred_value)

    def MAE(self) -> float:
        return np.mean(np.abs(self.true_value - self.pred_value))

    def RMSE(self) -> float:
        return np.sqrt(np.mean((self.true_value - self.pred_value) ** 2))

    def R2(self) -> float:
        r2 = (np.corrcoef(self.true_value, self.pred_value)[0][1]) ** 2
        return r2
class DensityMetrics:
    def __init__(self, true_Dens, pred_Dens):

        self.tD = true_Dens
        self.pD = pred_Dens

        if len(self.tD) != len(self.pD):
            self.equal_length = False
            while len(self.tD) > len(self.pD):
                self.pD = np.append(self.pD, 1)
            while len(self.tD) < len(self.pD):
                self.pD = self.pD[:-1]

            if len(self.tD) == len(self.pD):
                self.equal_length = True
        else:
            self.equal_length = True

        self.true_Dens = self.tD
        self.pred_Dens = self.pD


        self.topk = None

    def CosineSimilarity(self) -> float:

        """
        Got metric from:
        https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        """
        if self.equal_length == False:
            return 0

        return np.dot(self.true_Dens, self.pred_Dens) / (np.linalg.norm(self.true_Dens) * np.linalg.norm(self.pred_Dens))

    def InverseEuclideanDistance(self) -> float:

        """
        Got metric from:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean
        """
        if self.equal_length == False:
            return 0

        return 1/(distance.euclidean(self.true_Dens, self.pred_Dens)+0.1)

    def InverseCanberraDistance(self, squared=True) -> float:

        """
        Got metric from:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html#scipy.spatial.distance.canberra

        Square the denominater for a harsher metric!
        """
        if self.equal_length == False:
            return 0
        if squared==True:
            return 1/(distance.canberra(self.true_Dens, self.pred_Dens)+0.1)**2

        return 1/(distance.canberra(self.true_Dens, self.pred_Dens)+0.1)

    def RootMeanSquaredError(self) -> float:
        """
        Return Mean squared error per Molecule.
        """
        if self.equal_length == False:
            return 5

        return np.sqrt(np.mean((self.true_Dens - self.pred_Dens) ** 2))
        # error = 0
        # for pos, spin in enumerate(a):
        #     if spin != b[pos]:
        #         error += (spin-b[pos])**2
        #
        # return error/len(a)

    def PearsonR2_np(self):
        np.seterr(all='raise')
        if self.equal_length == False:
            return 0

                #Check if array contains all the same value --> if so no correlation can be computed
        if np.all(self.pred_Dens == self.pred_Dens[0]):
            return 0
        if np.all(self.true_Dens == self.true_Dens[0]):
            return 0

        r2 = (np.corrcoef(self.true_Dens, self.pred_Dens)[0][1]) ** 2
        return r2


    def HighestDens(self, a: np.ndarray, num_highest: int) -> np.ndarray:

        """
        Problem: Bei gleichen Werten wird der letztere genommen
        """

        if len(self.true_Dens) < num_highest:
            num_highest=len(self.true_Dens)
        if len(self.pred_Dens) < num_highest:
            num_highest = len(self.pred_Dens)

        try:
            idx_dens = np.array([np.argsort(a)[-(i + 1)] for i in range(num_highest)])
        except:

            print('Too Short!')
            print(a)
            return 0

        return idx_dens

    def CompareHighestDens(self, num_highest=5) -> float:
        self.topk = num_highest
        true_idx = self.HighestDens(self.true_Dens, num_highest=num_highest)
        pred_idx = self.HighestDens(self.pred_Dens, num_highest=num_highest)

        if type(true_idx) == int or type(pred_idx) == int:
            return len(self.pred_Dens)

        error = 0
        for i, pos in enumerate(true_idx):
            if pos != pred_idx[i]:
                error +=1
                if pos in pred_idx:
                    error -= float(1/num_highest)

        return error/len(self.pred_Dens)


    def RankDensities(self, num_highest=10) -> float:
        #true_idx = self.HighestDens(self.true_Dens, num_highest=num_highest)
        #pred_idx = self.HighestDens(self.pred_Dens, num_highest=num_highest)
        if len(self.true_Dens) < num_highest:
            num_highest=len(self.true_Dens)
        if len(self.pred_Dens) < num_highest:
            num_highest = len(self.pred_Dens)

        ha = self.HighestDens(self.true_Dens, num_highest=num_highest)
        hb = self.HighestDens(self.pred_Dens, num_highest=num_highest)

        take_into_account = int(num_highest/2)

        error = 0

        for i, posi in enumerate(ha[:take_into_account]):
            if posi in hb:
                error += np.abs(list(hb).index(posi) - i)
            else:
                error += num_highest

        for i, posi in enumerate(hb[:take_into_account]):
            if posi in ha:
                error += np.abs(list(ha).index(posi) - i)
            else:
                error += num_highest

        return error/(2*take_into_account)
