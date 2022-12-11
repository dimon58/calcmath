import unittest

import numpy as np

import lesson_3
from tests.common import SLAETestCaseMixin


class TestSLAE(SLAETestCaseMixin, unittest.TestCase):

    def test_simple_iterations(self):
        """
        Тестирует решение СЛАУ методом простых итераций
        """

        self._test_slae_all('Метод простых итераций', lesson_3.solve_simple_iterations)

    def test_yacoby(self):
        """
        Тестирует решение СЛАУ методом Якоби
        """

        A = np.array([
            [1., 1., 1.],
            [0., 1., 1.],
            [0., 0., 1.],
        ])
        f = np.array([1, 3, 5])

        self._test_slae_all('Метод Якоби', lesson_3.solve_yacoby)
        np.testing.assert_almost_equal(lesson_3.solve_yacoby(A, f), np.linalg.inv(A) @ f,
                                       err_msg='Ошибка в методе Якоби')

    def test_zeidel(self):
        """
        Тестирует решение СЛАУ методом Зейделя
        """

        A = np.array([
            [1., 1., 1.],
            [0., 1., 1.],
            [0., 0., 1.],
        ])
        f = np.array([1, 3, 5])

        np.testing.assert_almost_equal(lesson_3.solve_zeidel(A, f), np.linalg.inv(A) @ f,
                                       err_msg='Ошибка в методе Зейделя')


if __name__ == "__main__":
    unittest.main()
