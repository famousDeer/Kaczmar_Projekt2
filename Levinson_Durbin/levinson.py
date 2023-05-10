import numpy as np

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

def rlevinson(a, efinal):
    """computes the autocorrelation coefficients, R based
    on the prediction polynomial A and the final prediction error Efinal,
    using the stepdown algorithm.

    Works for real or complex data

    :param a:
    :param efinal:

    :return:
        * R, the autocorrelation
        * U  prediction coefficient
        * kr reflection coefficients
        * e errors

    A should be a minimum phase polynomial and A(1) is assumed to be unity.

    :returns: (P+1) by (P+1) upper triangular matrix, U,
        that holds the i'th order prediction polynomials
        Ai, i=1:P, where P is the order of the input
        polynomial, A.



             [ 1  a1(1)*  a2(2)* ..... aP(P)  * ]
             [ 0  1       a2(1)* ..... aP(P-1)* ]
       U  =  [ .................................]
             [ 0  0       0      ..... 1        ]

    from which the i'th order prediction polynomial can be extracted
    using Ai=U(i+1:-1:1,i+1)'. The first row of U contains the
    conjugates of the reflection coefficients, and the K's may be
    extracted using, K=conj(U(1,2:end)).

    .. todo:: remove the conjugate when data is real data, clean up the code
       test and doc.

    """
    a = np.array(a)
    realdata = np.isrealobj(a)


    assert a[0] == 1, 'First coefficient of the prediction polynomial must be unity'

    p = len(a)

    if p < 2:
        raise ValueError('Polynomial should have at least two coefficients')

    if realdata == True:
        U = np.zeros((p, p)) # This matrix will have the prediction
                                # polynomials of orders 1:p
    else:
        U = np.zeros((p, p), dtype=complex)
    U[:, p-1] = np.conj(a[-1::-1]) # Prediction coefficients of order p

    p = p -1
    e = np.zeros(p)

    # First we find the prediction coefficients of smaller orders and form the
    # Matrix U

    # Initialize the step down

    e[-1] = efinal # Prediction error of order p

    # Step down
    for k in range(p-1, 0, -1):
        [a, e[k-1]] = levdown(a, e[k])
        U[:, k] = np.concatenate((np.conj(a[-1::-1].transpose()) ,
                                      [0]*(p-k) ))




    e0 = e[0]/(1.-abs(a[1]**2)) #% Because a[1]=1 (true polynomial)
    U[0,0] = 1                #% Prediction coefficient of zeroth order
    kr = np.conj(U[0,1:])     #% The reflection coefficients
    kr = kr.transpose()                 #% To make it into a column vector

    #   % Once we have the matrix U and the prediction error at various orders, we can
    #  % use this information to find the autocorrelation coefficients.

    R = np.zeros(1, dtype=complex)
    #% Initialize recursion
    k = 1
    R0 = e0 # To take care of the zero indexing problem
    R[0] = -np.conj(U[0,1])*R0   # R[1]=-a1[1]*R[0]

    # Actual recursion
    for k in range(1,p):
        r = -sum(np.conj(U[k-1::-1,k])*R[-1::-1]) - kr[k]*e[k-1]
        R = np.insert(R, len(R), r)

    # Include R(0) and make it a column vector. Note the dot transpose

    #R = [R0 R].';
    R = np.insert(R, 0, e0)
    return R, U, kr, e



def levdown(anxt, enxt=None):
    """One step backward Levinson recursion

    :param anxt:
    :param enxt:
    :return:
        * acur the P'th order prediction polynomial based on the P+1'th order prediction polynomial, anxt.
        * ecur the the P'th order prediction error  based on the P+1'th order prediction error, enxt.

    ..  * knxt the P+1'th order reflection coefficient.

    """
    #% Some preliminaries first
    #if nargout>=2 & nargin<2
    #    raise ValueError('Insufficient number of input arguments');
    if anxt[0] != 1:
        raise ValueError('At least one of the reflection coefficients is equal to one.')
    anxt = anxt[1:] #  Drop the leading 1, it is not needed
                    #  in the step down

    # Extract the k+1'th reflection coefficient
    knxt = anxt[-1]
    if knxt == 1.0:
        raise ValueError('At least one of the reflection coefficients is equal to one.')

    # A Matrix formulation from Stoica is used to avoid looping
    acur = (anxt[0:-1]-knxt*np.conj(anxt[-2::-1]))/(1.-abs(knxt)**2)
    ecur = None
    if enxt is not None:
        ecur = enxt/(1.-np.dot(knxt.conj().transpose(),knxt))

    acur = np.insert(acur, 0, 1)

    return acur, ecur


def levup(acur, knxt, ecur=None):
    """LEVUP  One step forward Levinson recursion

    :param acur:
    :param knxt:
    :return:
        * anxt the P+1'th order prediction polynomial based on the P'th order prediction polynomial, acur, and the
          P+1'th order reflection coefficient, Knxt.
        * enxt the P+1'th order prediction  prediction error, based on the P'th order prediction error, ecur.


    :References:  P. Stoica R. Moses, Introduction to Spectral Analysis  Prentice Hall, N.J., 1997, Chapter 3.
    """
    if acur[0] != 1:
        raise ValueError('At least one of the reflection coefficients is equal to one.')
    acur = acur[1:] #  Drop the leading 1, it is not needed

    # Matrix formulation from Stoica is used to avoid looping
    anxt = np.concatenate((acur, [0])) + knxt * np.concatenate((np.conj(acur[-1::-1]), [1]))

    enxt = None
    if ecur is not None:
        # matlab version enxt = (1-knxt'.*knxt)*ecur
        enxt = (1. - np.dot(np.conj(knxt), knxt)) * ecur

    anxt = np.insert(anxt, 0, 1)

    return anxt, enxt