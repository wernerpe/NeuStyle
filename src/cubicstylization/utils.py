import numpy as np
import scipy
from scipy.sparse import csr_matrix

# https://github.com/HTDerekLiu/CubicStylization_MATLAB/blob/master for inspiration ~ using the same function structure to minimize chance of bugs


class RotData:
    def __init__(self):
        # ADMM rotation fitting parameters
        self.lambda_ = 4e-1  # cubeness
        self.rho = 1e-4
        self.ABSTOL = 1e-5
        self.RELTOL = 1e-3
        self.mu = 5
        self.tau = 2
        self.maxIter_ADMM = 100


rotdata = RotData()


def vertexFaceAdjacency(F):
    """
    VERTEXFACEADJACENCYLIST constructs a list indicates the
    indices of adjacent faces

    adjF = vertexFaceAdjacencyList(F)

    Input:
      F   |F| x 3   list of face indices
    Output:
      adjF |V| list where adjF[ii] outputs the adjacent face
      indices of a vertex
    """
    i = np.arange(0, F.shape[0])
    i = np.kron(i, np.array([1, 1, 1]))
    j = F.ravel()
    VT = csr_matrix(([1] * len(i), (i, j)))

    indices = np.arange(0, F.shape[0])
    adjF = [None] * VT.shape[1]
    for ii in range(VT.shape[1]):
        adjF[ii] = indices[VT[:, ii].toarray().ravel().astype(bool)].tolist()
    return adjF


def ArapConstrainedSolve(L, L_red, Rall, P, ConstraintLocation, FixedIndex):
    # RT_stack = np.concatenate(tuple([Rall[:,:,idx].T for idx in range(Rall.shape[2])]))
    Pprime = P.copy()
    Pprime[FixedIndex, :] = ConstraintLocation
    Pcons = 0 * P.copy()
    Pcons[FixedIndex, :] = ConstraintLocation
    # LP05 = 0.5*L@P
    rhs = np.zeros(P.shape)
    for i in range(len(P)):
        nzentries = L[i, :].tolil().rows[0]
        nzentries.remove(i)
        for j in nzentries:
            rhs[i, :] -= (
                0.5
                * L[i, j]
                * (
                    (P[i, :].reshape(1, 3) - P[j, :].reshape(1, 3))
                    @ (Rall[:, :, i].T + Rall[:, :, j].T)
                ).squeeze()
            )

    eqcons = L @ Pcons
    rhs = rhs - eqcons
    rows = [i for i in range(len(P))]
    for v in FixedIndex:
        rows.remove(v)
    Pprime_desc = scipy.sparse.linalg.spsolve(L_red, rhs[rows, :])
    # assert np.max(np.abs(ppr_col[FixedIndex]))<1e-4
    Pprime[rows, :] = Pprime_desc
    return Pprime


def shrinkage(x, k):
    """
    SHRINKAGE is the standard shrinkage operator

    Reference:
    Tibshirani, "Regression shrinkage and selection via the lasso", 1996
    S_k(a) =
    \bagin{cases}
        a-k , when  a  > 0
        0,    when |a| < k
        a+k,  when  a  < 0
    \end{cases}
    """
    z = np.maximum(0, x - k) - np.maximum(0, -x - k)
    return z


def fit_rotation(S):
    U, _, Vt = np.linalg.svd(S, full_matrices=True)
    U = U[:, :3]
    Vt = Vt[:3, :]
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        U[:, 2] = -U[:, 2]
        R[:] = np.dot(Vt.T, U.T)

    assert np.linalg.det(R) > 0
    return R


def fitRotationL1(U, rotData: RotData):
    """
    FITROTATIONL1 solves the following problem for each vertex i:
    Ri <- argmin Wi/2*||Ri*dVi - dUi||^2_F + lambda*VAi*|| Ri*ni||_1
    where R is a rotation matrix

    Reference:
    Liu & Jacobson, "Cubic Stylization", 2019 (Section 3.1)
    """

    nV = U.shape[0]
    RAll = np.zeros((3, 3, nV))  # all rotation
    objVal = 0

    # initialization (warm start for consecutive iterations)
    if not hasattr(rotData, "zAll"):
        rotData.zAll = np.zeros((3, nV))
        rotData.uAll = np.zeros((3, nV))
        rotData.rhoAll = rotData.rho * np.ones(nV)

        adjFList = vertexFaceAdjacency(rotData.F)
        rotData.hEList = [None] * nV  # half edge list
        rotData.WList = [None] * nV  # weight matrix W
        rotData.dVList = [None] * nV  # dV spokes and rims
        for ii in range(nV):
            adjF = adjFList[ii]
            first_cols = np.concatenate(
                (rotData.F[adjF, 0], rotData.F[adjF, 1], rotData.F[adjF, 2])
            )
            second_cols = np.concatenate(
                (rotData.F[adjF, 1], rotData.F[adjF, 2], rotData.F[adjF, 0])
            )
            hE = np.column_stack((first_cols, second_cols))
            np.ravel_multi_index((hE[:, 0], hE[:, 1]), rotData.L.shape)

            rotData.hEList[ii] = hE
            rotData.WList[ii] = np.diag(
                np.array(rotData.L[hE[:, 0], hE[:, 1]]).squeeze()
            )
            rotData.dVList[ii] = (rotData.V[hE[:, 1], :] - rotData.V[hE[:, 0], :]).T

    # start rotation fitting with ADMM
    for ii in range(nV):
        if ii % 500 == 0:
            print("rotation fitting: ", ii, "/", nV)
        # warm start parameters
        z = rotData.zAll[:, ii].reshape(-1, 1)
        u = rotData.uAll[:, ii].reshape(-1, 1)
        n = rotData.N[ii, :].reshape(-1, 1)
        rho = rotData.rhoAll[ii]

        # get geometry params
        hE = rotData.hEList[ii]
        W = rotData.WList[ii]
        dV = rotData.dVList[ii]
        dU = (U[hE[:, 1], :] - U[hE[:, 0], :]).T
        Spre = dV @ W @ dU.T

        # ADMM
        for k in range(rotData.maxIter_ADMM):
            # R step
            S = Spre + (rho * np.outer(n, (z - u)))
            R = fit_rotation(S)

            # z step
            zOld = z
            z = shrinkage(R @ n + u, rotData.lambda_ * rotData.VA[ii] / rho)

            # u step
            u = u + R @ n - z

            # compute residual, objective function
            r_norm = np.linalg.norm(z - R @ n)  # primal
            s_norm = np.linalg.norm(-rho * (z - zOld))  # dual

            # rho step
            if r_norm > rotData.mu * s_norm:
                rho = rotData.tau * rho
                u = u / rotData.tau
            elif s_norm > rotData.mu * r_norm:
                rho = rho / rotData.tau
                u = u * rotData.tau

            # check stopping criteria
            numEle = len(z)
            eps_pri = np.sqrt(numEle * 2) * rotData.ABSTOL + rotData.RELTOL * max(
                np.linalg.norm(R @ n), np.linalg.norm(z)
            )
            eps_dual = np.sqrt(
                numEle
            ) * rotData.ABSTOL + rotData.RELTOL * np.linalg.norm(rho * u)
            if r_norm < eps_pri and s_norm < eps_dual:
                # save parameters for future warm start
                rotData.zAll[:, ii] = z.squeeze()
                rotData.uAll[:, ii] = u.squeeze()
                rotData.rhoAll[ii] = rho
                # wierd bug i cannot find
                Rperm = -R
                Rperm[:, -1] *= -1
                RAll[:, :, ii] = R  # Rperm

                # save ADMM info
                objVal = (
                    objVal
                    - 0.5 * np.trace((R @ dV - dU) @ W @ (R @ dV - dU).T)
                    + rotData.lambda_ * rotData.VA[ii] * np.linalg.norm(R @ n, ord=1)
                )
                break

    return RAll, objVal, rotData
