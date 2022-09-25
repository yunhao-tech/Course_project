#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Etudiant
@author: Michael Allouche, Emmanuel Gobet
MAP556, Methodes de Monte Carlo: du lineaire au non-linéiare
"""

import numpy as np
import sys

def fonction_test(x, label=None):
    """
    input: current point x as np.array
           label describing the test function (not used by studends only by teachers)
    output: numerical value of the function "label" at point "x"
    """
    try:
        assert x.shape[0] == 5  # les gaussiennes doivent etre de dimension 5
    except AssertionError:
        print("Attention, la dimension des donnees est trop grande. Veuillez respecter la consigne.")
        sys.exit(1)

    # exemple fictif de fonction test: sin(x0 + x1**2 + x2**3 + x3**4 + x4**5)
    s=0
    for i in range(5):
        s += x[i]**(i+1)

    return (s)



