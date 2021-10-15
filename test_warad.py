import inspect
from math import sin, pi, cos
import sympy
import pytest

from warad import differentiable, differentiate


def test_run():
    @differentiable
    def f(x):
        return 2 * x

    assert inspect.isfunction(f)
    assert inspect.isfunction(f.d)

def test_constant():
    @differentiable
    def f(x):
        return 1

    assert f.d(10) == 0


def test_multivar():
    @differentiable
    def f(x, y):
        return x

    assert f.d(10, 10, _dv=0) == 1
    assert f.d(10, 10, _dv=1) == 0


def test_addition():
    @differentiable
    def f(x, y):
        return x + y

    assert f.d(10, 11) == 1


def test_power():
    @differentiable
    def f(x, y):
        return x ** y

    assert f.d(10, 3) == (3 * (10 ** 2))


def test_scope():
    c = 3

    @differentiable
    def f(x):
        return x * c

    assert f.d(1) == 3

def test_call():
    c = 5

    @differentiable
    def f(x):
        return x * c

    @differentiable
    def g(x):
        return f(2 * x)

    assert g.d(10) == 2 * c


def test_parse_text():
    f = differentiate("def f(x): return 2 * x")
    assert f(5) == 2


@pytest.mark.parametrize('expression', [
    "x",
    "2 + x",
    "x + 2",
    "x + x",
    "5 - x",
    "x - 3"
    "2 * x",
    "x * 10",
    "x * x",
    "2 * x * 2 * x",
    "x / 10",
    "10 / (x + 0.1)",
    "x ** 2",
    "(2 * x) ** 2",
    "x * (2 / (x + 0.1)) + (5 * x / 2) ** 5",
    "2 ** x", # TODO :) 
    "sin(x)",
    "sin(2 * x)",
    "cos(x)",
    "cos( 2 * sin(x))",
    "x ** 1 + x ** 3 + x ** 5",
])
def test_symbolic(expression):
    assert "\n" not in expression, "No new lines, please"

    points = [ (i - 20) / 2 for i in range(40) ]

    x = sympy.symbols("x")
    symbolic = sympy.diff(sympy.sympify(expression), x)
    automatic = differentiate("def f(x): return " + expression)
    for p in points:
        assert symbolic.subs(x, p).evalf() == pytest.approx(automatic(p)), "Results don't match for '%s' at %s " % (expression, p)
