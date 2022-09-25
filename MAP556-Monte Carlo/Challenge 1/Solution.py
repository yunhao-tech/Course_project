#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Etudiant
@author: Xinkai TANG, Yunhao CHEN
MAP556, Methodes de Monte Carlo: du lineaire au non-linéiare
"""

import numpy as np
import Fonction_A_Tester as FAT
import argparse
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--LabelFonction", help="numero label fonction", default="fake")
args = parser.parse_args()
labelfonction= args.LabelFonction

N_appel_max_fonction=400 # le nombre maximal d'appel a la fonction test pour un array de dimension dimension_gaussienne
dimension_gaussienne=5

# algo MCMC a faire
import scipy.stats as stats

def frac(x):
    return x-np.floor(x)

def Torus(_N):
    '''using torus algorithm to generate a sequence of variables in [0,1]

    return:
    X : nd.array,shape of (5,_N)
    '''
    P = [2,3,5,7,11]
    P = np.sqrt(P)
    X = []
    for i in range(1,_N+1):
        x = [frac(i*p) for p in P]
        X.append(x)
    return np.array(X).T

def generate_gaussien(_N):
    '''given a uniform variable, generate a gaussien variable

    Parameters:
    _X : np.array, shape=(5,N)

    return:
    G : np.array,shape=(5,N)
    '''
    X = Torus(_N)
    return stats.norm.ppf(X)


def gradient_uM(_theta,_G,_fG):
    '''compute the gradient of uM(theta)

    Parameters
    ----------
    _theta : np.array, shape=(5,1)

    _G : np.array, shape=(5,N_simulation)
        Each column of G is a 5-dimension gaussien vector

    _fG : np.array, shape=(N_simulation,1)
        Application of function f on G

    Return
    ------- 
    gradient : np.array, shape=theta.shape

    '''
    exponential = np.exp(-np.dot(_G.T,_theta)) # Mx1
    up = np.dot(_G,_fG**2 * exponential) # 5*1
    down = np.sum(_fG**2 * exponential)
    gradient = _theta - up / down
    return gradient

def second_gradient_uM(_theta,_G,_fG):
    '''compute the gradient of uM(theta)

    Parameters
    ----------
    _theta : np.array, shape=(5,1)

    _G : np.array, shape=(5,N_simulation)
        Each column of G is a 5-dimension gaussien vector

    _fG : np.array, shape=(N_simulation,1)
        Application of function f on G

    Return
    ------- 
    hessian : np.array, shape=(5,5)
        hessian matrice

    '''

    Id = np.identity(dimension_gaussienne)
    exponential = np.exp(-np.dot(_G.T,_theta)) # Mx1, M = _G.shape[1]
    tmp = _fG**2 * exponential # Mx1
    up1 = np.dot(_G,_G.T * tmp)
    down1 = np.sum(_fG**2 * exponential)
    upleft = np.dot(_G, _fG**2 * exponential) # 5*1
    upright = np.dot((_fG**2 * exponential).T, _G.T) # 1*5
    up2 = upleft.dot(upright) # 5*5
    down2 = down1**2
    hessian = Id + up1/down1 - up2/down2
    return hessian

def newton(times,_theta,_G,_fG):
    '''using newton method to find optimal solution for theta

    Parameters
    ----------
    times : int
        Number of iterations for Newton method

    _theta : np.array, shape=(5,1)

    _G : np.array, shape=(5,N_simulation)
        Each column of G is a 5-dimension gaussien vector

    _fG : np.array, shape=(N_simulation,1)
        Application of function f on G

    Return
    ------- 
    theta_new : np.array, shape=(5,1)
        the optimal solution theta

    '''
    theta_new = _theta
    for _ in range(times):
        theta_new = theta_new - np.dot(np.linalg.inv(second_gradient_uM(theta_new,_G,_fG)), gradient_uM(theta_new,_G,_fG))
    return theta_new


### Aiming at employing the method Control Variates, compute the optimal vector beta
def Z_beta(_X,_Z):
    '''compute the beta that minimize the variance of (_X - beta * _Z) 

    Parameters
    ----------
    _X : the underlying random variable

    _Z : A random vector of d dimensions, which has a strong correlation with _X 

    Return
    ------- 
    Z_beta : variable de controle. Z_beta = beta * _Z

    '''
    assert _X.shape[0] == _Z.shape[1], 'Shape is not corresponding'
    first_part = np.linalg.inv(_Z.dot(_Z.T))
    second_part = _Z.dot(_X)
    beta = first_part * second_part
    Z_beta = _Z.T.dot(beta)
    return Z_beta


N_newton = 20
N_for_theta = 80

theta_init = (0.5*np.ones(dimension_gaussienne)).reshape((-1,1)) # 5x1

G_ = generate_gaussien(N_appel_max_fonction - N_for_theta)
fG = FAT.fonction_test(G_[:,:N_for_theta])[:,None] # N_simulation_for_theta x 1
theta_opt = newton(N_newton,theta_init, G_[:,:N_for_theta], fG) # 5*1

fG_theta = FAT.fonction_test(G_ + theta_opt)[:,None]
norm_theta = np.linalg.norm(theta_opt)**2 / 2
temp = fG_theta * np.exp(-np.dot(G_.T,theta_opt) - norm_theta)
estimateur = np.mean( temp - Z_beta(temp, G_) )


variance = np.var( temp - temp - Z_beta(temp, G_))
width_IC = 1.96 * np.sqrt(variance / G_.shape[1])
msg_estim = "La valeur d'estimation de E(f(G)) est donnée par : {0:.4f}".format
print(msg_estim(estimateur))
msg_estim = "La variance est donnée par : {0:.4f}".format
print(msg_estim(variance))
msg_IC = "L'intervalle de confiance est [{0:.4f},{1:.4f}]".format
print(msg_IC(estimateur-width_IC, estimateur+width_IC))



with open('ma_reponse_{}.txt'.format(labelfonction), "w") as file_result:
    file_result.write("{:.6f}".format(NumericalResult))



