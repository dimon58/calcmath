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


def ensure_correct_slae(a: np.ndarray, f: np.ndarray):
    ensure_square_matrix(a)

    if len(f.shape) != 1:
        raise ValueError('Свободный коэффициент должен быть вектор-столбцом')

    if a.shape[0] != f.shape[0]:
        raise ValueError('Размер матрицы a не соответствуют размерам свободного коэффициента f')
