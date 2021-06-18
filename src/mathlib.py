import platform
_impl = None


o_system = platform.platform()
if o_system == 'Darwin':
    import metal_math as metal
    _impl = metal
elif o_system == 'Linux':
    import numpy as np
    _impl = np
elif o_system == 'Windows':
    import numpy as np
    _impl = np
else:
    import numpy as np
    _impl = np


def array(x):
    return _impl.array(x)


def square(x, y):
    return _impl.square(x, y)


def T(x):
    return array(x).T


def dot(x, y):
    return _impl.dot(x, y)


def add(x, y):
    return _impl.add(x, y)
