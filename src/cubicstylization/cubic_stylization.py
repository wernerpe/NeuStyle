import igl
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from .utils import ArapConstrainedSolve, RotData, fitRotationL1


def CubiclyStylize(
    V,
    F,
    V_pinned_index_list,
    V_pinned_locations,
    max_alternations=20,
    lambda_=0.2,
    ADMM_iters=100,
    plotting_handle=None,
):
    """
    This function applies cubic stylization to a triangle mesh as described in
    https://www.dgp.toronto.edu/projects/cubic-stylization/cubicStyle_high.pdf

    input:
    V                    |V|x3 vertex list
    F                    |F|x3 face list containing vertex indeces
    V_pinned_index_list  List with k vertex indeces pinned to a preset location
    V_pinned_locations   kx3 array containing preset locations of pinned vertices
    max_alternations     max number of stylization iterations (local-global steps)
    lambda_              relative weighting of cubeness cost
    ADMM_iters           max number of ADMM iterations to fit the rotation matrices
    plotting_handle      function handle of form f(V,F, V_pinned_locations)
                         that is called during the iterations to display intermediate
                         results.
    output:
    U                    |V|x3 Deformed vertices satisfying pinned constraints
    Rall                 3x3x|V| Per vertex rotation matrices of last step
    """
    # check that the number of delivered constraints and locations matches
    if len(V_pinned_index_list) < 1:
        raise ValueError("Please pin at least one vertex")
    assert len(V_pinned_index_list) == V_pinned_locations.shape[0]
    assert ADMM_iters > 10

    # load solver params
    rotdata = RotData()
    rotdata.lambda_ = lambda_
    rotdata.maxIter_ADMM = ADMM_iters

    rotdata.F = F.copy()
    rotdata.L = csr_matrix(0.5 * (igl.cotmatrix(V, F)))
    rotdata.V = V.copy()
    rotdata.N = igl.per_vertex_normals(V, F)
    rotdata.VA = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()

    vpin = V_pinned_index_list

    tol = 1e-3

    objHis = []
    UHis = np.zeros((len(V), 3, max_alternations))

    # reduce laplacian
    cols = [i for i in range(len(V))]
    for v in vpin:
        cols.remove(v)
    L_red = scipy.sparse.lil_matrix(rotdata.L[:, cols])
    L_red = scipy.sparse.lil_matrix(L_red[cols, :])
    U = V.copy()
    U[vpin, :] = V_pinned_locations

    # plot initial mesh
    if plotting_handle is not None:
        plotting_handle(U, F, V_pinned_locations)

    for it in range(max_alternations):
        # fit rotations using admm
        RAll, val, rotdata = fitRotationL1(U, rotdata)
        objHis.append(val)
        UHis[:, :, it] = U

        # global step
        UPre = U
        U = ArapConstrainedSolve(rotdata.L, L_red, RAll, V, V_pinned_locations, vpin)
        # stopping criteria
        dU = np.sqrt(np.sum((U - UPre) ** 2, axis=1))
        dUV = np.sqrt(np.sum((U - V) ** 2, axis=1))
        if np.max(dUV) == 0:
            print("converged")
            break
        reldV = np.max(dU) / np.max(dUV)
        print("iter: %d, objective: %d, reldV: %d" % (it, val, reldV))
        if plotting_handle is not None:
            plotting_handle(U, F, V_pinned_locations)
        if reldV < tol:
            break

    return U, RAll
