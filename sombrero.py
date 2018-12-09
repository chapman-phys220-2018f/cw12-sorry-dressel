#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###
# Name: Gwyneth Casey and Amelia Rosetto
# Student ID: 2286584
# Email: gcasey@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW12
###

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

"""This module will plot different curves of the sombrero potential to investigate its attributes."""


@nb.jit
def d(t, g, F):
    """
    This function will solve for the derivatives of the equation of motion and       output them into an array.
    """
    # F is the force
    # t is the parameter

    return (np.array([g[1], -0.25*g[1] + g[0] - g[0]**3 + F*np.cos(t)])) # omega = 1 so it does not show in equation


def runge4thorder(F, x, y, N):
    """
    This function will use Runge-Kutta 4th order integration method.
    """

    deltat = 0.001
    t = np.arange(0, 2*np.pi*N, deltat)
    Approx = np.zeros((len(t), 2))
    Approx[0] = np.array([x, y])

    for u in range(len(t)-1):

        k1 = deltat*(d(t[u], Approx[u,:], F))
        k2 = deltat*(d(t[u] + deltat/2, Approx[u,:] + (k1/2), F))
        k3 = deltat*(d(t[u] + deltat/2, Approx[u,:] + (k2/2), F))
        k4 = deltat*(d(t[u] + deltat, Approx[u,:] + k3, F))
        Approx[u+1,:] = Approx[u,:] + ((k1 + 2*k2 + 2*k3 + k4)/6)
    return t, Approx


