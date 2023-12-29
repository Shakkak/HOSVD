import numpy as np

def Unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """
    Unfold the given tensor along a specified mode.

    Parameters:
    - tensor (numpy.ndarray): The input tensor to be unfolded.
    - mode (int): The mode along which the tensor should be unfolded.
        - 1: Unfold along the first mode (columns).
        - 2: Unfold along the second mode (rows).
        - 3: Unfold along the third mode (depth).

    Input Shape (tensor): (z, x, y)

    Output Shape:
    - Mode 1: (x, z*y)
    - Mode 2: (y, x*z)
    - Mode 3: (z, x*y)
    
    Returns:
    - numpy.ndarray: The unfolded tensor.

    Raises:
    - ValueError: If an invalid mode is provided.

    Example:
    >>> tensor = np.array([[[  1,   2,   3],
        [  4,   5,   6],
        [  7,   8,   9],
        [ 10,  11,  12]],

       [[ 11,  22,  33],
        [ 44,  55,  66],
        [ 77,  88,  99],
        [1010, 1111, 1212]]])
    >>> Unfold(tensor, mode=1)
    array([[   1,   11,    2,   22,    3,   33],
       [   4,   44,    5,   55,    6,   66],
       [   7,   77,    8,   88,    9,   99],
       [  10, 1010,   11, 1111,   12, 1212]])

    >>> Unfold(tensor, mode=2)
    array([[   1,    4,    7,   10,   11,   44,   77, 1010],
       [   2,    5,    8,   11,   22,   55,   88, 1111],
       [   3,    6,    9,   12,   33,   66,   99, 1212]])

    >>> Unfold(tensor, mode=3)
    aarray([[   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
          12],
       [  11,   22,   33,   44,   55,   66,   77,   88,   99, 1010, 1111,
        1212]])
    
    """
    output = []
    if mode == 1:
        for dim in range(tensor.shape[2]):
            for col in tensor[:, :, dim]:
                output.append(col)
        output = np.array(output).T
        return output
    elif mode == 2:
        for dim in range(tensor.shape[0]):
            for row in tensor[dim, :, :]:
                output.append(row)
        output = np.array(output).T
        return output
    elif mode == 3:
        for dim in range(tensor.shape[1]):
            for dep in tensor[:, dim, :].T:
                output.append(dep)
        output = np.array(output).T
    else:
        raise ValueError("Invalid mode. Supported modes are 1, 2, and 3.")
    return output

def mult(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """
    Perform mode multiplication between a tensor and a matrix along a specified mode.

    Parameters:
    - tensor (numpy.ndarray): The input tensor.
    - matrix (numpy.ndarray): The input matrix.
    - mode (int): The mode along which the multiplication should be performed.
        - 1: Multiply along the first mode (columns).
        - 2: Multiply along the second mode (rows).
        - 3: Multiply along the third mode (depth).
        
    Input Shapes:
    - Tensor (tensor): (z, x, y)
    - Matrix (matrix):
        - Mode 1:(q, x)
        - Mode 2:(q, y)
        - Mode 3:(q, z)

    Output Shapes:
    - Mode 1: (z, q, y)
    - Mode 2: (z, x, q)
    - Mode 3: (q, x, y)
    
    Returns:
    - numpy.ndarray: The result of the mode multiplication.

    Raises:
    - ValueError: If an invalid mode is provided.

    Example:
    >>> tensor = np.array([[[4,2],[-3,0],[0,-6],[-1,1]],[[0,-2],[1,-1],[3,4],[5,-4]],[[10,1],[5,2],[-1,4],[-2,0]]])
    >>> matrix1 = np.array([[5, 9, 2, 4],
       [7, 8, 4, 7],
       [5, 4, 6, 2],
       [5, 8, 2, 8],
       [6, 1, 6, 1]])
    >>> matrix2 = np.array([[2, 7],
       [3, 6],
       [2, 3],
       [8, 7],
       [2, 2],
       [1, 4]])
    >>> matrix3 = np.array([[4, 7, 2],
       [1, 3, 7],
       [7, 8, 6],
       [7, 7, 2],
       [2, 7, 5],
       [1, 3, 5],
       [2, 3, 6]])
    
    >>> mult(tensor, matrix1, mode=1)
    array([[[-11,   2],
         [ -3,  -3],
         [  6, -24],
         [-12,   6],
         [ 20, -23]],
 
        [[ 35, -27],
         [ 55, -34],
         [ 32,   2],
         [ 54, -42],
         [ 24,   7]],
 
        [[ 85,  31],
         [ 92,  39],
         [ 60,  37],
         [ 72,  29],
         [ 57,  32]]])

    >>> mult(tensor, matrix2, mode=2)
    array([[[ 22,  24,  14,  46,  12,  12],
        [ -6,  -9,  -6, -24,  -6,  -3],
        [-42, -36, -18, -42, -12, -24],
        [  5,   3,   1,  -1,   0,   3]],

       [[-14, -12,  -6, -14,  -4,  -8],
        [ -5,  -3,  -1,   1,   0,  -3],
        [ 34,  33,  18,  52,  14,  19],
        [-18,  -9,  -2,  12,   2, -11]],

       [[ 27,  36,  23,  87,  22,  14],
        [ 24,  27,  16,  54,  14,  13],
        [ 26,  21,  10,  20,   6,  15],
        [ -4,  -6,  -4, -16,  -4,  -2]]])

    >>> mult(tensor, matrix3, mode=3)
    array([[[ 36,  -4],
        [  5,  -3],
        [ 19,  12],
        [ 27, -24]],

       [[ 74,   3],
        [ 35,  11],
        [  2,  34],
        [  0, -11]],

       [[ 88,   4],
        [ 17,   4],
        [ 18,  14],
        [ 21, -25]],

       [[ 48,   2],
        [ -4,  -3],
        [ 19,  -6],
        [ 24, -21]],

       [[ 58,  -5],
        [ 26,   3],
        [ 16,  36],
        [ 23, -26]],

       [[ 54,   1],
        [ 25,   7],
        [  4,  26],
        [  4, -11]],

       [[ 68,   4],
        [ 27,   9],
        [  3,  24],
        [  1, -10]]])
    """
    if mode == 1:
        ten = (matrix @ Unfold(tensor, 1))
        matrix_indices = [[(i) * tensor.shape[0] + dep for i in range(tensor.shape[2])] for dep in range(tensor.shape[0])]
        ten = np.concatenate([ten[:, indices].T for indices in matrix_indices], axis=1)
        ten = ten.T.reshape(tensor.shape[0], matrix.shape[0], tensor.shape[2])
        return ten
    
    elif mode == 2:
        ten = (matrix @ Unfold(tensor, 2))
        matrix_indices = [[(i + dep * tensor.shape[1]) for i in range(tensor.shape[1])] for dep in range(tensor.shape[0])]
        ten = np.concatenate([ten[:, indices].T for indices in matrix_indices], axis=0)
        ten = ten.reshape(tensor.shape[0], tensor.shape[1], matrix.shape[0])
        return ten
    
    elif mode == 3:
        ten = (matrix @ Unfold(tensor, 3))
        ten = [np.array(ten[dep, :]).reshape(tensor.shape[1], tensor.shape[2]) for dep in range(matrix.shape[0])]
        ten = np.concatenate(ten, axis=0)
        ten = ten.reshape(matrix.shape[0], tensor.shape[1], tensor.shape[2])
        return ten
    else:
        raise ValueError("Invalid mode. Supported modes are 1, 2, and 3.")

def HOSVD(tensor: np.ndarray) -> tuple:
    """
    Perform Higher Order Singular Value Decomposition (HOSVD) on the given tensor.

    Parameters:
    - tensor (numpy.ndarray): The input tensor.

    Returns:
    - tuple: A tuple containing the core tensor S and the mode matrices U1, U2, U3.

    Example:
    >>> tensor = np.random.rand(3, 4, 5)
    >>> S, U1, U2, U3 = HOSVD(tensor)
    """
    U1, _, _ = np.linalg.svd(Unfold(tensor, 1), full_matrices=False)
    U2, _, _ = np.linalg.svd(Unfold(tensor, 2), full_matrices=False)
    U3, _, _ = np.linalg.svd(Unfold(tensor, 3), full_matrices=False)

    S = tensor.copy()
    S = mult(S, U1.T, 1)
    S = mult(S, U2.T, 2)
    S = mult(S, U3.T, 3)

    return S, U1, U2, U3
