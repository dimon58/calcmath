import math

from misc import Number, DEFAULT_STEP_OF_NUMERICAL_DIFFERENTIATION, EPS


def get_optimal_step_of_numerical_differentiation(m0: Number = None, m2: Number = None, eps: Number = EPS) -> Number:
    """
    Возвращает оптимальный шаг численного дифференцирования, если m0 или m2 None, то возвращает шаг по умолчанию

    Args:
        m0: максимум функции
        m2: максимум второй производной
        eps: машинная точность
    """
    if m0 is not None and m2 is not None:
        return 2 * math.sqrt(m0 * m2 * eps)

    return DEFAULT_STEP_OF_NUMERICAL_DIFFERENTIATION


def get_forward_derivative(func, x: Number, m0=None, m2=None) -> Number:
    """
    Производная вперёд от функции func в точке x

    Вычисляется по формуле f'(x) = (f(x+h) - f(x))/h

    Args:
        func: функция, от которой берём производную

        x: точка, в которой берём производную

        m0: максимум функции

        m2: максимум второй производной
    """

    h = get_optimal_step_of_numerical_differentiation(m0, m2)

    return (func(x + h) - func(x)) / h


def get_backward_derivative(func, x: Number, m0=None, m2=None) -> Number:
    """
    Производная назад от функции func в точке x

    Вычисляется по формуле f'(x) = (f(x) - f(x-h))/h

    Args:
        func: функция, от которой берём производную

        x: точка, в которой берём производную

        m0: максимум функции

        m2: максимум второй производной
    """

    h = get_optimal_step_of_numerical_differentiation(m0, m2)

    return (func(x) - func(x - h)) / h


def get_central_derivative(func, x: Number, m0=None, m2=None) -> Number:
    """
    Центральная производная от функции func в точке x

    Вычисляется по формуле f'(x) = (f(x+h) - f(x-h))/(2h)

    Args:
        func: функция, от которой берём производную

        x: точка, в которой берём производную

        m0: максимум функции

        m2: максимум второй производной
    """

    h = get_optimal_step_of_numerical_differentiation(m0, m2)

    return (func(x + h) - func(x - h)) / (2 * h)
