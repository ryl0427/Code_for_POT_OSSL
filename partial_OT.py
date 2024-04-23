import numpy as np
import torch
# solve partial optimal transport problem
def entropic_partial(a, b, M, reg, numItermax=1000,
                                 stopThr=1e-3, verbose=False, log=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape
    dy = np.ones(dim_b, dtype=np.float64)

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    err, cpt = 1, 0
    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)

    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K = K * q1
        dia = (a / np.sum(K, axis=1))
        for i in range(len(dia)):
            K[i] = K[i]*dia[i]
        q1 = q1 * Kprev / K
        
        K1prev = K
        K = K * q2
        dia = np.minimum(b / np.sum(K, axis=0), dy)
        for i in range(len(dia)):
            K[:,i] = K[:,i]*dia[i]
        q2 = q2 * K1prev / K
        
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    log_e['partial_w_dist'] = np.sum(M * K)
    if log:
        return K, log_e
    else:
        return K

# calculate transport mass
def check_result(opt_result):
    s = np.ones(len(opt_result))
    opt = np.dot(s, opt_result)
    return opt