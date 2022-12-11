import random

import numpy as np
import numpy.testing as np_testing


class SLAETestCaseMixin:
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

    def _test_slae_all(self, solver_name, solver_function):
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
