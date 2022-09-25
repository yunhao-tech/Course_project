# ipynb-set
The homework in form of Jupyter notebook

List:
* MAP556 Monte Carlo Methods:
    * TP1: Simulation of random variables + Law of large numbers + Central limit theorem
    * TP2: Serveral methods of variance reduction
         * control variates
         * antithetic sampling
         * stratification
    * TP3: Variance reduction through importance sampling
    * TP5: Using Empirical Regression to approximate conditional expectation (in a context of finance)
    * TP6: Generative Adversarial Network (GAN)
    * TP8: Simulate processes of Brownian motion (eg. process of Ornstein-Uhlenbeck) and their Euler scheme 
    * TP9: Multi-level Monte-Carlo method (MLMC)
    * Challenge 1: simulate E(f(G)), f is reasonably regular, G is a 5-dimensional gaussian vector with I5 the covariance matrix. Attention: The number to call function f is limited to 400. 
    * Challenge 2: To play Angry Bird! Try to give a control on velocity to the bird facing a random wind. The law of wind follows Brownian motion. We can find more details in 
         - AngryBirds_Env.ipynb provides an environment based on this problem. We can generate the random wind and train a deep learning model.
         - ventETU.csv provides an example of wind format
         - savedModel is a trained model which takes time and position as input and outputs the control we should take.
         - control_etudiant.py and mon_control.py aim to provide the framework (write the result to file...)
         - Rapport.pdf is a resume of our ideas and methods. 

* MAP553 Machine learning:
    * TP1: The aim of this TP is to implement several optimization algorithms for linear regression and logistic regression models, with ridge penalization. The list of algorithms implemented:
        * gradient descent (GD)
        * accelerated gradient descent (AGD)
        * coordinate gradient descent (CGD)
        * stochastic gradient descent (SGD)
        * stochastic average gradient descent (SAG)
        * stochastic variance reduced gradient descent (SVRG) 
   * Project: This project is based on a classic dataset of tree cover type classification. Auto machine learning is used to automatically choose model and fine tune hyperparameters. The accuracy can arrive at 0.869, a second position in Kaggle competition. The details are in report. 

* INF580 Large scale mathematical optimization:
   * Project: combine random projection and linear programming. Retrieve solutions from projected problem and dual projected problem, compute primal solution. Compare their feasibility error and compute time.

* MAP433 statistiques: 
    * EN1: Estimation parametrique. Loi de Poisson pour modéliser le nombre de buts marqués par une équipe de football.
    * EN2: Test de Cramer-von Mises
    * EN3: Transformation de stabilisation de la variance
    * DM1: La variable a expliquer est la concentration en ozone noté O3 et les variables explicatives sont la temperature noté T12, le vent noté Vx et la nébulosité noté Ne12.
    * DM2: Mettre en oeuvre un test asymptotique sur les coefficients de regression.
    * DM3: Classification par la methode **des k plus proches voisins**.

* MAP432 Markov & martingale
    * Projet: L'algorithme du **recuit simulé**, pour résoudre des problèmes d'optimisation non convexe. On s'intéresse ici à une application au problème du voyageur de commerce.


* MAP435 optimisation 
    * optimisation **sans contrainte**: 
         * Algorithme de gradient à pas fixe
         * Algorithme du gradient pas optimal (le cas de fonction quadratic)
         * Algorithme de Nesterov (fonctions convexes)
         * Algorithme de Nesterov (fonctions fortement convexes)
         * Algorithme du gradient conjugué
         * Algorithme de Newton
         * Analyse de vitesse de convergence et comparer les algos
         * Quelques contre-exemples
    * optimisation **avec contraintes**: différents algorithmes d'optimisation sous contraintes dans un cas simple (minimisation d'une fonction quadratique sous contraintes affines)
         * Algorithme du gradient projeté
         * Algorithme d'Uzawa
         * Méthode de pénalisation
         * Algorithme du Lagrangien augmenté
