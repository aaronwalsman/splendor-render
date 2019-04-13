def bezier3(t, a, b, c, d):
    return ((1-t)**3 * a +
            3 * (1-t)**2 * t * b +
            3 * (1-t) * t**2 * c +
            t**3 * d)

def crapterp(t, x):
    return (t**(1/x) * (1-t) + t**x * t)

def softish_step(t, a):
    if t <= 0:
        return 0
    elif t >= 1:
        return 1
    
    if t < 0.5:
        return 0.5 * (2*t)**a
    
    else:
        return -0.5 * (2*(1-t))**a + 1
