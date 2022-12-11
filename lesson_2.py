import numpy as np


def ensure_square_matrix(a: np.ndarray):
    """
    Проверяет, является ли матрица a квадратной. Если нет, то вызывает исключение ValueError
    Args:
        a: матрица
    """

    if len(a.shape) != 2:
        ValueError('Массив не двумерный')

    w, h = a.shape
    if w != h:
        raise ValueError('Не квадратная матрица')


def get_matrix_norm_1(a: np.ndarray):
    """
    Вычисляет матричную норму (максимальную сумму элементов в столбце)

    Args:
        a: матрица
    """

    return a.sum(axis=0).max()


def get_vector_norm_1(a: np.ndarray):
    """
    Вычисляет векторную норму (максимальный элемент), подчинённую матричной норме get_matrix_norm_1

    Args:
        a: вектор
    """

    return np.abs(a).max()


def get_matrix_norm_2(a: np.ndarray):
    """
    Вычисляет матричную норму (максимальную сумму элементов в строке)

    Args:
        a: матрица
    """

    return a.sum(axis=1).max()


def get_vector_norm_2(a: np.ndarray):
    """
    Вычисляет векторную норму (сумму модулей элементов), подчинённую матричной норме get_matrix_norm_2

    Args:
        a: вектор
    """

    return np.abs(a).sum()


def get_matrix_norm_3(a: np.ndarray):
    """
    Вычисляет матричную норму (корень из максимального собственное число aa*)

    Args:
        a: матрица
    """

    ensure_square_matrix(a)
    return np.sqrt(np.max(np.linalg.eigvals(a @ a.conj())))


def get_vector_norm_3(a: np.ndarray):
    """
    Вычисляет векторную норму (евклидову), подчинённую матричной норме get_matrix_norm_3

    Args:
        a: вектор
    """

    return np.sqrt(np.square(a).sum())


def get_number_of_conditionality(a, matrix_norm=get_matrix_norm_3):
    """
    Возвращает число обусловленности матрицы

    Args:
        a: матрица
        matrix_norm: матричная норма
    """
    ensure_square_matrix(a)
    return matrix_norm(a) * matrix_norm(np.linalg.inv(a))


def kramer_solve(a: np.ndarray, f: np.ndarray):
    """
    Решает систему уравнение ax=f относительно x методом Крамера (x = a^(-1)f)
    Args:
        a: матрица СЛАУ
        f: свободный коэффициент

    Returns:
        Вектор решений
    """
    ensure_square_matrix(a)
    if a.shape[0] != f.shape[0]:
        raise ValueError('Размер матрицы a не соответствуют размерам свободного коэффициента f')
    return np.linalg.inv(a) @ f


def make_upper_triangle(matrix: np.ndarray):
    """
    Приводит матрицу к верхнетреугольному виду элементарными преобразованиями

    Args:
        matrix: матрица (в процессе модифицируется)
    """
    for i in range(np.min(matrix.shape)):

        # ищем первую строку с ненулевым элементом в i-ом столбце
        for j in range(i, matrix.shape[0]):
            divider = matrix[j, i]

            if divider != 0:

                # если нашли, то обмениваем её с i-ой строкой при необходимости
                if j != i:
                    old_row = matrix[i, i:].copy()
                    matrix[i, i:] = matrix[j, i:]
                    matrix[j, i:] = old_row

                matrix[i] /= divider
                # теперь надо вычесть приведённую строку из всех нижележащих строчек
                for lower_row in matrix[i + 1:]:
                    factor = lower_row[i]  # элемент строки в колонке i
                    lower_row -= factor * matrix[i]  # вычитаем, чтобы получить ноль в колонке i

    return matrix


def gauss_backward(matrix):
    # перебор строк в обратном порядке
    for i in range(len(matrix) - 1, 0, -1):
        row = matrix[i]
        for upper_row in matrix[:i]:
            factor = upper_row[i]
            # вычитать строки не нужно, так как в row только два элемента отличны от 0:
            # в последней колонке и на диагонали

            # вычитание в последней колонке
            upper_row[-1] -= factor * row[-1]
            # вместо вычитания 1*factor просто обнулим коэффициент в соотвествующей колонке.
            upper_row[i] = 0
    return matrix


def gauss_solve(a: np.ndarray, f: np.ndarray):
    """
    Решает систему уравнение ax=f относительно x методом Гаусса
    Args:
        a: матрица СЛАУ
        f: свободный коэффициент

    Returns:
        Вектор решений
    """

    ensure_square_matrix(a)
    if a.shape[0] != f.shape[0]:
        raise ValueError('Размер матрицы a не соответствуют размерам свободного коэффициента f')

    a = np.c_[a.copy(), f]
    a = make_upper_triangle(a)

    if np.any(np.diag(a) == 0):
        raise ValueError("Система уравнений не совместна")

    a = gauss_backward(a)
    return a[:, -1]


def make_upper_triangle_pivot(matrix):
    for i in range(len(matrix)):
        # np.argmax возвращает номер строки с максимальным элементом в уменьшенной матрице
        # которая начинается со строки i
        pivot = i + np.argmax(abs(matrix[i:, i]))

        # переставляем строки
        if pivot != i:
            matrix[i], matrix[pivot] = matrix[pivot], np.copy(matrix[i])

        row = matrix[i]
        # диагональный элемент
        divider = row[i]

        if abs(divider) < 1e-10:
            # Почти ноль на диагонали => неустойчивость
            raise ValueError("Система уравнений несовместна")

        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in matrix[i + 1:]:
            factor = lower_row[i]  # элемент строки в колонке i
            lower_row -= factor * row  # вычитаем, чтобы получить ноль в колонке i

    return matrix


def gauss_pivot_solve(a, f):
    """
    Решает систему уравнение ax=f относительно x методом Гаусса c выбором главного элемента
    Args:
        a: матрица СЛАУ
        f: свободный коэффициент

    Returns:
        Вектор решений
    """

    ensure_square_matrix(a)
    if a.shape[0] != f.shape[0]:
        raise ValueError('Размер матрицы a не соответствуют размерам свободного коэффициента f')

    a = np.c_[a.copy(), f]
    a = make_upper_triangle_pivot(a)

    if np.any(np.diag(a) == 0):
        raise ValueError("Система уравнений не совместна")

    a = gauss_backward(a)
    return a[:, -1]
