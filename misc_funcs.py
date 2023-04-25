import numpy as np
import time
import jax.numpy as jnp
from jax import grad as jgrad


def calculate_pend_z_with_mag_mass(l, L, M, m, g, b):
    return (b * l**2) / ( 2 * jnp.sqrt(g * (m*l + M*L) * (m*l**2 + M*L**2)))

def calculate_x_final_with_damping(b, a, m, tf):
    return (a * m**2)/(b**2) * jnp.exp((-b)/(m)*tf) + (a * m)/(b) * tf - (a * m**2)/(b**2)

def calculate_mu_k_from_final_time(xf, tf, th, g):
    return ((-2 * xf) / (g * tf**2) + jnp.sin(th)) / jnp.cos(th)

def calculate_init_ramp_acc(th, mu_k, g):
    return g * (jnp.sin(th) - mu_k * jnp.cos(th))

def NRmethod(func, guess):
    # solve using newton raphson method
    # x1 = x0 - (f(x0) / f'(x0))
    grad_func = jgrad(func)(guess)
    guess_next = guess - (func(guess)/grad_func)
    return guess_next
