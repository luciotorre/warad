import ast
from collections import namedtuple
import copy
import inspect
import math
import textwrap

# FIXME: fix getsource of computed derivatives

_delta = "δ"

class Dual:
    def __init__(self, value, diff=0):
        self.value = value
        self.diff = diff

    @classmethod
    def force(cls, value):
        if not isinstance(value, cls):
            return cls(value, 0)
        return value

    def __repr__(self):
        return "Dual(%s + %sϵ)" % (self.value, self.diff)

    def __add__(self, other):
        other = Dual.force(other)

        return Dual(
            self.value + other.value,
            self.diff + other.diff
        )
    __radd__ = __add__

    def __sub__(self, other):
        other = Dual.force(other)

        return Dual(
            self.value - other.value,
            self.diff - other.diff
        )

    def __rsub__(self, other):
        other = Dual.force(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = Dual.force(other)

        return Dual(
            self.value * other.value, 
            other.value * self.diff + other.diff * self.value
        )
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual.force(other)
        f = self.value
        df = self.diff
        g = other.value
        dg = other.diff

        return Dual(
            f / g,
            (df * g  - dg * f) / (g ** 2)
        )

    def __rtruediv__(self, other):
        other = Dual.force(other)
        g = self.value
        dg = self.diff
        f = other.value
        df = other.diff

        return Dual(
            f / g,
            (df * g  - dg * f) / (g ** 2)
        )
    
    def __pow__(self, other):
        if isinstance(other, Dual):
            assert other.diff == 0, "We dont support 'c ** x' yet"
            other = other.value

        d = Dual(
            self.value ** other, 
            other * (self.value ** (other - 1)) * self.diff
        )
        return d

    

class RemoveDifferentiableDecorator(ast.NodeTransformer):

    def visit_FunctionDef(self, node):
        return ast.FunctionDef(
            name=_delta + node.name, 
            args=node.args,
            body=node.body, 
            decorator_list=[], 
            returns=node.returns,
            type_comment=node.type_comment
        )
    
def remove_decorator_ast(tree):
    return ast.fix_missing_locations(RemoveDifferentiableDecorator().visit(copy.deepcopy(tree)))
    
    
class DifferentiateAST(ast.NodeTransformer):
    def visit_Call(self, node):
        return ast.Call(
            func=ast.Name(id=_delta), 
            args=[node.func] + [differentiate_ast(n) for n in node.args], 
            keywords=node.keywords
        )

    def visit_For(self, node):
        return node


def differentiate_ast(tree):
    return ast.fix_missing_locations(DifferentiateAST().visit(copy.deepcopy(tree)))

derivatives = {
    math.sin: math.cos,
    math.cos: lambda x: -math.sin(x),
}


def call_wrapper(f, *args, **kwargs):
    """Create a dual for f(g())"""
    assert not kwargs, "No multivariate support"
    assert len(args) == 1, "No multivariate support"
    df = derivatives.get(f, None)
    if df is None:
        assert hasattr(f, "d"), "%s is not  differentiable" % (f,)
        df = f.d

    g = args[0]
    return Dual(value=f(g.value), diff=df(g.value) * g.diff)


def box(args, _dv):
    return [Dual(arg, int(_dv==i)) for i, arg in enumerate(args)]


def unbox(value):
    if isinstance(value, Dual):
        return value.diff
    return 0


def boxunbox(f):
    def _inner(*args, _dv=0):
        bargs = box(args, _dv=_dv)
        v = f(*bargs)
        return unbox(v)
    return _inner


def differentiate(f):
    if isinstance(f, str):
        fast = ast.parse(f)
        scope = dict((k, getattr(math, k)) for k in dir(math))

        assert isinstance(fast, ast.Module)
        assert isinstance(fast.body[0], ast.FunctionDef)
        
        fname = fast.body[0].name
    else:
        assert inspect.isfunction(f)
        source = textwrap.dedent(inspect.getsource(f))
        fname = f.__name__
        scope = dict(f.__globals__, **inspect.getclosurevars(f).nonlocals)
        fast = ast.parse(source)

    cfast = remove_decorator_ast(fast)
    dast = differentiate_ast(cfast)
    scope[_delta] = call_wrapper
    new_code = ast.unparse(dast)
    co = compile(new_code, "<dual>", 'exec')
    exec(co, scope, scope)
    return boxunbox(scope[_delta + fname])
    

def differentiable(f):
    assert inspect.isfunction(f), "'differentiable' is a decorator, %s is not a function type" % (type(f), )
    f.d = differentiate(f)
    return f


if __name__ == "__main__":
    import pytest 

    def f(x): 
        return x * 2

    df = differentiate(f)
    
    assert df(10) == 2

    def g(x):
        r = 0
        for i in range(6):
            if i % 2 == 1:
                r = r + x ** i
        return r        

    e = 0.0001
    dg = differentiate(g)
    x = 1
    assert g(x + e) == pytest.approx(g(x) + e * dg(x))
    