import numpy as np

from misc import ensure_correct_slae, Number


def solve_simple_iterations(a: np.ndarray, f: np.ndarray, x0: np.ndarray = None, tau: Number = None, iters=10000):
    """
    Решает систему уравнение ax=f относительно x методом простых итераций
    Args:
        a: матрица СЛАУ
        f: свободный коэффициент
        x0: начальное приближение
        tau: шаг итераций
        iters: число итераций

    Returns:
        Вектор решений
    """

    ensure_correct_slae(a, f)

    if tau is None:
        ls = np.linalg.eigvals(a)
        if np.any(np.abs(ls) >= 1):
            raise ValueError('Метод простых итераций не сойдётся')

        tau = 2 / (np.max(ls) + np.min(ls))

    if x0 is None:
        x0 = f.copy()

    B = np.eye(*a.shape) - tau * a
    b = tau * f
    xn = x0

    for i in range(iters):
        xn = B @ xn + b

    return xn


def solve_yacoby(a: np.ndarray, f: np.ndarray, iters=10000):
    """
    Решает систему уравнение ax=f относительно x методом Якоби

    x_{n+1} = -D^(-1)(L + U)x_n + D^(-1)f

    Args:
        a: матрица СЛАУ
        f: свободный коэффициент
        iters: число итераций

    Returns:
        Вектор решений
    """

    ensure_correct_slae(a, f)

    D = np.diag(np.diag(a))
    L = np.tril(a) - D
    U = np.triu(a) - D

    b = np.linalg.inv(D)
    B = -b @ (L + U)
    b = b @ f

    ls = np.linalg.eigvals(B)
    if np.any(np.abs(ls) >= 1):
        raise ValueError('Метод Якоби не сойдётся')

    xn = np.zeros(*f.shape)

    for i in range(iters):
        xn = B @ xn + b

    return xn


def solve_zeidel(a: np.ndarray, f: np.ndarray, iters=10000):
    """
    Решает систему уравнение ax=f относительно x методом Зейделя

    x_{n+1} = -(L + D)^(-1)Ux_n + (L + D)^(-1)f

    Args:
        a: матрица СЛАУ
        f: свободный коэффициент
        iters: число итераций

    Returns:
        Вектор решений
    """

    ensure_correct_slae(a, f)

    D = np.diag(np.diag(a))
    L = np.tril(a) - D
    U = np.triu(a) - D

    b = np.linalg.inv(L + D)
    B = -b @ U
    b = b @ f

    ls = np.linalg.eigvals(B)
    if np.any(np.abs(ls) >= 1):
        raise ValueError('Метод Якоби не сойдётся')

    xn = np.zeros(*f.shape)

    for i in range(iters):
        xn = B @ xn + b

    return xn


if __name__ == '__main__':
    from lesson_2 import get_vector_norm_3
    import matplotlib.pyplot as plt

    k = 20
    iters = 700
    np.random.seed(32832)
    a = np.random.random((k, k))
    d = np.diag(np.diag(a)) * 19
    a -= d
    a += np.eye(*a.shape)

    x = np.array(range(1, a.shape[0] + 1))
    f = a @ x

    # Якоби
    D = np.diag(np.diag(a))
    L = np.tril(a) - D
    U = np.triu(a) - D

    b = np.linalg.inv(D)
    B = -b @ (L + U)
    b = b @ f

    xn = np.zeros(*f.shape)

    r = []

    for i in range(iters):
        xn = B @ xn + b
        r.append(get_vector_norm_3(x - xn))
    plt.plot(range(1, len(r) + 1), r, label='Якоби')

    # Зейдель
    D = np.diag(np.diag(a))
    L = np.tril(a) - D
    U = np.triu(a) - D

    b = np.linalg.inv(L + D)
    B = -b @ U
    b = b @ f

    xn = np.zeros(*f.shape)

    r = []
    for i in range(iters):
        xn = B @ xn + b
        r.append(get_vector_norm_3(x - xn))

    plt.plot(range(1, len(r) + 1), r, label='Зейдель', color='red')
    plt.yscale('log')
    plt.legend()
    plt.show()
