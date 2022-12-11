import unittest

import numpy as np
import numpy.testing as np_testing

import lesson_2
from tests.common import SLAETestCaseMixin


class TestNorm(unittest.TestCase):

    def test_matrix_norm_1(self):
        self.assertAlmostEqual(
            lesson_2.get_matrix_norm_1(np.array(([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]))), 18,
            msg="Ошибка вычисления матричной нормы 1"
        )

    def test_matrix_norm_2(self):
        self.assertAlmostEqual(
            lesson_2.get_matrix_norm_2(np.array(([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]))), 24,
            msg="Ошибка вычисления матричной нормы 2"
        )

    def test_matrix_norm_3(self):
        self.assertAlmostEqual(
            lesson_2.get_matrix_norm_3(np.array(([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]))), 16.11684396980705,
            msg="Ошибка вычисления матричной нормы 3"
        )

    def test_vector_norm_1(self):
        self.assertAlmostEqual(
            lesson_2.get_vector_norm_1(np.array([1, 2, 3])), 3,
            msg="Ошибка вычисления векторной нормы 1"
        )

    def test_vector_norm_2(self):
        self.assertAlmostEqual(
            lesson_2.get_vector_norm_2(np.array([1, 2, 3])), 6,
            msg="Ошибка вычисления векторной нормы 2"
        )

    def test_vector_norm_3(self):
        self.assertAlmostEqual(
            lesson_2.get_vector_norm_3(np.array([1, 2, 3])), 3.7416573867739413,
            msg="Ошибка вычисления векторной нормы 3"
        )


class TestSLAE(SLAETestCaseMixin, unittest.TestCase):

    def test_make_upper_triangular(self):
        """
        Тест приведения к верхнетреугольному виду
        """
        a = np.array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.],
        ]), np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ])

        b = np.array([
            [1., 2., 3.],
            [4., 8., 6.],
            [7., 8., 9.],
        ]), np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
        ])

        c = np.array([
            [1., 2., 3.],
            [4., 8., 6.],
            [7., 8., 9.],
            [7., 8., 9.],
        ]), np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ])

        d = np.array([
            [1., 2., 3.],
            [4., 8., 6.],
        ]), np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, -6.0],
        ])

        e = np.array([
            [0., 1., 1.],
            [0., 0., 1.],
            [0., 0., 0.],
        ]), np.array([
            [0., 1., 1.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])

        for matrix, upper in (a, b, c, d, e):
            np.testing.assert_almost_equal(
                lesson_2.make_upper_triangle(matrix.copy()), upper,
                err_msg=f'Неправильное приведение к верхнетреугольному виду'
            )

    def test_kramer(self):
        """
        Тестирует решение СЛАУ методом Крамера
        """
        self._test_slae_all('Метод Крамера', lesson_2.kramer_solve)

    def test_gauss(self):
        """
        Тестирует решение СЛАУ методом Гаусса
        """
        self._test_slae_all('Метод Гаусса', lesson_2.gauss_solve)

    def test_gauss_pivot(self):
        """
        Тестирует решение СЛАУ методом Гаусса с выбором главного элемента
        """
        self._test_slae_all('Метод Гаусса с выбором главного элемента', lesson_2.gauss_pivot_solve)


if __name__ == "__main__":
    unittest.main()
