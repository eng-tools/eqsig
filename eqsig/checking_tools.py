def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    """
    To check equality of floats
    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return:
    """
    if a is None or b is None:
        return False
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
