import math
import unittest
from math import pi

import numpy as np

import lesson_1


class TestDerivativeCalculation(unittest.TestCase):

    def __test_derivative_polynomial(self, name, derivative_formula):
        test_range = np.linspace(-100, 100, 1000)

        min_error = float('inf')
        max_error = 0

        for deg in range(0, 21):
            polynomial = np.poly1d(np.random.random(deg + 1))
            polynomial_derivative = polynomial.deriv()
            for x in test_range:
                derivative = derivative_formula(polynomial, x)
                real_derivative = polynomial_derivative(x)
                deviation = abs(derivative - real_derivative) / (abs(real_derivative) + 1e-10)

                min_error = min(min_error, deviation)
                max_error = max(max_error, deviation)

                self.assertLess(
                    deviation, 1e-2,
                    f'Производная от функции {name} в точке {x} ({derivative})'
                    f'сильно отличается от истины ({real_derivative}) ({deviation})'
                )

        print(f'{name}: минимальное отклонение {min_error * 100:.4g}%, максимальное {max_error * 100:.4g}%')

    def test_forward_derivative_polynomial(self):
        """
        Тест формулы производной вперёд на полиноме
        """
        self.__test_derivative_polynomial("Производная вперёд", lesson_1.get_forward_derivative)

    def test_backward_derivative_polynomial(self):
        """
        Тест формулы производной назад на полиноме
        """
        self.__test_derivative_polynomial("Производная назад", lesson_1.get_backward_derivative)

    def test_central_derivative_polynomial(self):
        """
        Тест формулы центральной производной назад на полиноме
        """
        self.__test_derivative_polynomial("Центральная производная", lesson_1.get_central_derivative)

    def __test_derivative(self, name, derivative_formula):
        functions = (
            ("sin", math.sin, math.cos, np.linspace(-2 * pi, 2 * pi, 100)),
            ("cos", math.cos, lambda x: -math.sin(x), np.linspace(-2 * pi, 2 * pi, 100)),
            ("tan", math.tan, lambda x: 1 / math.cos(x) ** 2, np.linspace(-0.45 * pi, 0.45 * pi, 100)),
            ("atan", math.atan, lambda x: 1 / (1 + x ** 2), np.linspace(-10, 10, 1000)),
            ("exp", math.exp, math.exp, np.linspace(-10, 10, 100)),
            ("log", math.log, lambda x: 1 / x, np.linspace(0.1, 200, 100)),
            ("sqrt", math.sqrt, lambda x: 1 / (2 * math.sqrt(x)), np.linspace(0.1, 200, 100)),
            ("sinh", math.sinh, math.cosh, np.linspace(-20, 20, 1000)),
            ("cosh", math.cosh, math.sinh, np.linspace(-20, 20, 1000)),
        )

        min_error = float('inf')
        max_error = 0

        for func_name, func, func_derivative, values in functions:
            for x in values:
                derivative = derivative_formula(func, x)
                real_derivative = func_derivative(x)
                deviation = abs(derivative - real_derivative) / (abs(real_derivative) + 1e-10)

                min_error = min(min_error, deviation)
                max_error = max(max_error, deviation)

                self.assertLess(
                    deviation, 1e-2,
                    f'Производная от функции {func_name} в точке {x} ({derivative})'
                    f'сильно отличается от истины ({real_derivative}) ({deviation})'
                )

        print(f'{name}: минимальное отклонение {min_error * 100:.4g}%, максимальное {max_error * 100:.4g}%')

    def test_forward_derivative(self):
        """
        Тест формулы производной вперёд
        """
        self.__test_derivative("Производная вперёд", lesson_1.get_forward_derivative)

    def test_backward_derivative(self):
        """
        Тест формулы производной назад
        """
        self.__test_derivative("Производная назад", lesson_1.get_backward_derivative)

    def test_central_derivative(self):
        """
        Тест формулы центральной производной назад
        """
        self.__test_derivative("Центральная производная", lesson_1.get_central_derivative)


if __name__ == "__main__":
    unittest.main()
