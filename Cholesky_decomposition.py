import numpy as np


A = np.array([10, 9, 7, 9, 12, 6, 7, 6, 7]).reshape(3, 3)

L = np.zeros((len(A), len(A)))
res = np.linalg.cholesky(A)


class Cholesky_Decomposition:
    def __init__(self, matrice):
        self.A = matrice

    def decomposition(self):
        """
        Allow to find the Triangular Lower matrix used in the decompostion of a matrix:
        A = LL^(-1)
        :return: L
        """
        if np.sum(np.linalg.eigvals(self.A) < 0) > 0:
            raise ValueError(
                "Matrice has to be semi definite. Here is the eigenvalues found: "
                + str(np.linalg.eigvals(self.A))
            )

        L = np.zeros((len(self.A), len(self.A)))

        for j in range(len(self.A)):

            L[j, j] = np.sqrt(
                self.A[j, j] - np.sum([L[j, k] ** 2 for k in range(0, j)])
            )

            for i in range(j + 1, len(self.A)):

                L[i, j] = (
                    self.A[i, j] - np.sum([L[i, j - 1] * L[j, k] for k in range(0, j)])
                ) / L[j, j]

        return L


met = Cholesky_Decomposition(A)

ch = met.decomposition()


