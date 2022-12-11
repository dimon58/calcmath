import random
import unittest

import numpy as np
import numpy.testing as np_testing

import lesson_2


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
            lesson_2.get_matrix_norm_1(np.array([1, 2, 3])), 3,
            msg="Ошибка вычисления векторной нормы 1"
        )

    def test_vector_norm_2(self):
        self.assertAlmostEqual(
            lesson_2.get_matrix_norm_2(np.array([1, 2, 3])), 6,
            msg="Ошибка вычисления векторной нормы 2"
        )

    def test_vector_norm_3(self):
        self.assertAlmostEqual(
            lesson_2.get_matrix_norm_3(np.array([1, 2, 3])), 3.7416573867739413,
            msg="Ошибка вычисления векторной нормы 3"
        )


class TestSLAE(unittest.TestCase):

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

    def __test_not_square_matrix(self, solver_name, solver_function):
        """
        Тестирует неквадратную матрицу

        Args:
            solver_name: название метода решения
            solver_function: решающая функция (метод Гаусса, Крамера и т.д.)
        """
        a = np.array(([
            [1, 2, 3],
            [4, 5, 6],
        ]))

        with self.assertRaises(ValueError) as context:
            solver_function(a, np.array([1, 2]))

        self.assertTrue(f'{solver_name} не корректно обрабатывает неквадратные матрицы')

    def __test_inconsistent_dimensions(self, solver_name, solver_function):
        """
        Тестирует несогласованность размеров матрицы и свободного коэффициента

        Args:
            solver_name: название метода решения
            solver_function: решающая функция (метод Гаусса, Крамера и т.д.)
        """
        a = np.array(([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]))

        with self.assertRaises(ValueError):
            solver_function(a, np.array([1, 2]))

        self.assertTrue(
            f'{solver_name} не корректно обрабатывает несогласованность размеров матрицы и свободного коэффициента')

    def __test_slae(self, solver_name, solver_function):
        """
        Тест решения систем уравнений

        Args:
            solver_name: название метода решения
            solver_function: решающая функция (метод Гаусса, Крамера и т.д.)
        """
        with self.subTest('Проверка обычной матрицы'):
            a = np.diag([1., 1., 1.])
            f = np.array([1., 2., 3.]).T
            x = np.array([1., 2., 3.]).T

            np.testing.assert_almost_equal(solver_function(a, f), x, err_msg=f'{solver_name} неправильно решает СЛАУ')

        with self.subTest('Нули на диагонали'):
            a = np.array([
                [0., 1., 1.],
                [1., 0., 1.],
                [1., 1., 0.],
            ])
            x = np.array([1., 2., 3.]).T

            f = a @ x
            print(f)
            np.testing.assert_almost_equal(solver_function(a, f), x, err_msg=f'{solver_name} неправильно решает СЛАУ')

    def __test_random_slae(self, solver_name, solver_function):
        """
        Тест решения систем уравнений со случайными матрицами

        Args:
            solver_name: название метода решения
            solver_function: решающая функция (метод Гаусса, Крамера и т.д.)
        """
        for i in range(20):
            size = random.randint(2, 50)

            a = np.random.random((size, size))
            f = np.random.random((size,))
            x = np.linalg.inv(a) @ f

            np.testing.assert_almost_equal(solver_function(a, f), x, err_msg=f'{solver_name} неправильно решает СЛАУ')

    def __test_all(self, solver_name, solver_function):
        """
        Тест решения систем уравнений

        Args:
            solver_name: название метода решения
            solver_function: решающая функция (метод Гаусса, Крамера и т.д.)
        """

        with self.subTest('Проверка неквадратных матриц'):
            self.__test_not_square_matrix(solver_name, solver_function)

        with self.subTest('Проверка несогласованность размеров матрицы и свободного коэффициента'):
            self.__test_inconsistent_dimensions(solver_name, solver_function)

        with self.subTest('Решение СЛАУ'):
            self.__test_slae(solver_name, solver_function)

        with self.subTest('Решение случайных СЛАУ'):
            self.__test_random_slae(solver_name, solver_function)

    def test_kramer(self):
        """
        Тестирует решение СЛАУ методом Крамера
        """
        self.__test_all('Метод Крамера', lesson_2.kramer_solve)

    def test_gauss(self):
        """
        Тестирует решение СЛАУ методом Гаусса
        """
        self.__test_all('Метод Гаусса', lesson_2.gauss_solve)

    def test_gauss_pivot(self):
        """
        Тестирует решение СЛАУ методом Гаусса с выбором главного элемента
        """
        self.__test_all('Метод Гаусса с выбором главного элемента', lesson_2.gauss_pivot_solve)


if __name__ == "__main__":
    unittest.main()
