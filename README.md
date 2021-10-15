THIS IS A TOY
=============

*War, huh, yeah*

*What is it good for?*

*Absolutely nothing, uhh*

WARAD: The WAR Automatic Differentiation python library.
========================================================

What does it do?
----------------

```python

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

```

Current status
--------------

 * Can differentiate functions (the idea worked!)
 * Woefully incomplete
 * Very easy to write code that will give wrong derivatives
