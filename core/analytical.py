import torch
import numpy as np

def oscillator(d, w0, t, tic=0.0, xic=1.0, vic=0.0):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-(d*xic+vic)/(w*xic)) - w*tic
    print("phi=", phi)
    A = xic/(2*np.exp(-d*tic)*np.cos(w*tic+phi))
    print("A=", A)
    cos = torch.cos(phi+w*t)
    sin = torch.sin(phi+w*t)
    exp = torch.exp(-d*t)
    x  = exp*2*A*cos
    return x


def oscillator2(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    print("phi=", phi)
    A = 1/(2*np.cos(phi))
    print("A=", A)
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

def oscillator3(d, w0, x, tic=0.0, xic=1.0):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    print("phi=", phi)
    A = xic/(2*np.cos(phi))
    print("A=", A)
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y