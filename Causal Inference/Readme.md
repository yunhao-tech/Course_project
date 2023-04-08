# Course Summary

In machine learning, there has been great progress in obtaining powerful predictive models, but these models rely on correlations between variables and do not allow for an understanding of the underlying mechanisms or how to intervene on the system for achieve a certain goal. The concepts of causality are fundamental to have levers for action, to formulate recommendations and to answer the following questions: "**what would happen if**" we had acted differently?

The questions of causal inference arise in many areas (socio-economics, politics, psychology, medicine, etc.): depending on the context which drug to use to improve the patient's health? what marketing strategy for product placement should be used to influence consumer buying behavior, etc. The formalism of causal inference makes it possible to study these questions as a problem of classical statistical inference. **The gold standard for estimating the effect of treatment is a randomized controlled trial (RCT)** which is, for example, mandatory for the authorization of new drugs in pharmaceutical and medical research. **However, RCTs are generally very expensive in terms of time and financial costs, and in some areas such as economics or political science, it is often not possible to implement an RCT**, for example to assess the effectiveness of a given policy.

The aim of this course is to present  the available methods to perform causal inference from observational data.

Below is a summarization of pratical Python codes both during lectures and labs. You can also find the my `Course summary`. The `Reference book` is a very good introduction to causal inference, which may interest you. The [tutorial online](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) is also useful.

---

# Lecture 1: RCT

1.1 Neyman Rubin model

- use `graphviz` package to draw DAG.
- simulation of nonlinear samples; simulation with intervention of treatment

1.2 Randomized Control Treatment

- Check balance and characteristics of control group and treament group. Use `describe()`
- Use `sns.boxplot` to check the distribution of each confounding variable.
- `stats.ks_2samp` to test whether two samples are drawn from the same distribution. Reject the null hypothesis if p-value < 0.05. It usually used to test **the goodness of fit**. Here to compare the distribution of variables in C/T group.

# Lab1: DM and OLS estimator in RCT context

Ex 1:

- Difference in means (DM) estimator & its confidence interval.
- `stats.ttest_ind` to **check the Control group and Treatment group do have the difference in target value**. (In accordance with the obtained ATE which is significantly not 0)
- OLS estimator: Fit linear  Difference in predicted means is ATE
- Attention to the variance of DM and OLS (When to divide by n)
- Repeat the estimation multi-times, visualize the statistics of ATE, compare the variance of DM and OLS estimator. We discover: The variance of OLS estimator is smaller than DM.

Ex 2:

- Use `sns.distplot` to highlight distributions between C/T groups
- Use `sns.catplot` : boxplot for each value of a given covariate, for both C/T groups. In order to observe the influence of the given covariate on Y, in both C/T groups.
- Use package `TableOne` to plot a so-call “Table 1”: A global statistics of the different covariates. It is one habit of econometrics and medical papers.
- To estimate confidence interval, two methods: **asymptotic variance** and those based on the **bootstrap**.

---

# Lecture 2: Propensity score

- Logistic regression to **estimate our propensity score**. Other classification models (like Neural network or Random Forest) are possible.
    
    Goal of propensity score estimation is to **make sure to include all the confounding variables**.
    
- important to **check for propensity score overlap** between the treated and untreated population : There is no case that propensity score equals to 0 or 1.

$$
0 < \eta < e(x) < 1- \eta
$$

- **Inverse Propensity Weighting (IPW)**. For individuals in the treatment group, $w = \frac{1}{e(x)}$, whereas for individuals in the control group, $w = \frac{1}{1 - e(x)}$. Then run a weighted Linear Regression (use `smf.wls`). If you use LinearRegression in sklearn, there is no standard error of coefficients, nor confidence interval for ATE. We can compute it with **Bootstrap** method.

# Lab 2: OLS and IPW, confoundings, parametric and non parametric

- Display `sns.histplot` to show distribution of the covariates with respect to the treatment (to check if the data is balanced).
- According to experimental studies, The DM estimator is not appropriate to estimate the ATE in **observational studies**.
- Note: The confounding variables: related to both the allocation of the treatment and the outcome model.
- OLS estimator: mean of difference between estimated targets for C/T groups.
- Comments:
    The parametric IPW and OLS when using all confounding factors are unbiased. The variance of the IPW estimator is larger; this is expected as this estimator is highly variable as its variance depends of the inverse of the propensity score. Consequently, when overlap is not large or when some probabilities to be treated or non treated are close to 1, the estimator is unstable. Normalized IPW can be an alternative to stabilize the estimator. When adding the variables, for OLS estimator, one can see that the variance decreases. When the models are mispecified (identifiability assumptions are not met as we do not include all the confounders), there is a large bias for both estimators.

---

# Lab 3: Doubly robust estimation

- Compare OLS, IPW and AIPW (Augmented Inverse Propensity Weighting) estimator.
- Show the strength of Doubly robust estimation: Even one of the outcome model and propensity score model is mispecified, AIPW performs well. You can afford to be wrong on one model.

---

# Lab 4: Causal effect using Dowhy

---

# Lab5: Causal discovery

- To infer the causal graph structure (skeleton and causal-consequence relation), using several causal discovery methods : **PC algorithm, LinGAM and ANM**
- PC algorithm: make use of conditional independence to discover the causal graph model.
- LiNGAM: Estimation of **Linear, Non-Gaussian Acyclic causal Model** from observed data. It assumes **non-Gaussianity of the noise terms** in the causal model.
- ANM: Additive noise model
