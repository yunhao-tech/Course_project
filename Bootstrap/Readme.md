# Brief summary of TD

- TD1: use **Jackknife** to estimate the asymptotic variance (Ex.1) and bias (Ex.2) of estimators

- TD2: use the **bootstrap** to estimate the bias (Ex.2) and variance (Ex.1, possibly to use Boostrap of Boostrap) of an estimator/statistic

- TD3: apply the theory for the bootstrap (Ex.1); see the limitations of the bootstrap (Ex.2); a weighted bootstrap scheme (Ex.3). 

# Brief summary of lectures

> A more detailed summary in Chinese is available on [Zhihu](https://zhuanlan.zhihu.com/p/619250283)

## Jackknife

**Automatically estimate Bias and Variance of an estimator**

Applications:

- An estimator is biased and we want to reduce its bias
- The asymptotic variance of a statistic has a difficult expression and we want to avoid a difficult computations

### Bias estimation

Let $\hat \theta=G_n(Z_1,...,Z_n)$ be a statistic used to estimate $\theta_0$

The bias of the estimator is defined as $b=\mathbb{E}[\hat \theta-\theta_0]$

The Jackknife estimator of this bias b is :

$$
b_{\text{jack}}=(n-1)(\bar{\hat \theta} - \hat \theta)
$$

where

$$
\bar{\hat \theta}=\frac{1}{n} \sum_{i=1}^n \hat \theta_{n-1,i} \ \ \text{ and } \ \hat \theta_{n-1,i}=G_{n-1}(Z_1,...,Z_{i-1}, Z_{i+1},...,Z_n)
$$

i.e. Each $\hat \theta_{n-1,i}$ is the statistic based on all observations except the i-th (leave-one-out).

The jackknife corrected estimator is:  $\hat \theta - b_{\text{jack}} = n \hat \theta - (n-1)\bar{\hat \theta}$

### Variance estimation

The Jackknife variance estimator is 

$$
v_{\text{jack}} = \frac{n-1}{n}\sum_{i=1}^n [\hat \theta_{n-1,i}-\bar{\hat \theta}]^2 \ \ \text{ where } \ \bar{\hat \theta}=\frac{1}{n}\sum_{i=1}^n \hat \theta_{n-1,i}
$$

we can show that $v_{jack} / \sigma_n^2 \rightarrow 1$ in probability (where $\sigma_n^2$   is the asymptotic variance of $\hat \theta$ ) so that:

$$
\frac{\hat \theta - \theta_0}{\sqrt{v_{jack}}} \rightarrow \mathcal{N}(0,1)
$$

We can then use it to construct confidence interval. 95% level confidence interval for $\theta_0$ is : 

$$
[\hat \theta - q_{0.975} * \sqrt{v_{jack}}, \ \ \hat \theta + q_{0.975} * \sqrt{v_{jack}}]
$$

---

## Bootstrap

Let $\hat \theta=G_n(Z_1,...,Z_n)$ be a statistic used to estimate $\theta_0$

Ex.  Want to know $\mathbb{E}[Z]$ and estimate it by sample mean ($\hat \theta= \bar Z$)

### Introduction

Construct Confidence intervals: use distribution of $\hat \theta$

- Case 1: $(Z_1,...,Z_n)$ iid with $Z_i \sim \mathcal{N}(\theta_0, V)$ and V is known. We have the **exact distribution**: $\hat \theta \sim \mathcal{N}(\theta_0, V/n)$ such that $\frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{V}} \sim \mathcal{N}(0, I)$

  Just use the quantile of standard normal distribution to construct the exact CI:  

  $$
  [\hat \theta - q_{1-\alpha/2} * \sqrt{\frac{V}{n}}, \ \ \hat \theta - q_{\alpha/2} * \sqrt{\frac{V}{n}}]
  $$

- Case 2: $Z_i \sim \mathcal{N}(\theta_0, V)$ may not hold. We can then apply CLT (Central limit theorem) to obtain the asymptotic distribution:

  $$
  \frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{V}} \sim \mathcal{N}(0, I) \ \ \text{when} \ n \rightarrow +\infty 
  $$

  We can also use a plug-in estimator of variance $\hat V$:

  $$
  \frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{\hat V}} \sim \mathcal{N}(0, I) \ \ \text{when} \ n \rightarrow +\infty 
  $$

  In other words, 

  $$
  P(q_{\alpha/2} \le \frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{\hat V}} \le q_{1- \alpha/2}) \rightarrow 1- \alpha \ \ \text{when} \ n \rightarrow +\infty 
  $$

  In this case, the CI is like:

  $$
  [\hat \theta - q_{1-\alpha/2} * \sqrt{\frac{\hat V}{n}}, \ \ \hat \theta - q_{\alpha/2} * \sqrt{\frac{\hat V}{n}}]
  $$

  There are several shortcomings:

  1. The estimator of variance might be difficult to construct or have a complex expression
  2. In case of limited samples, the approximation error is significant and confidence interval is not accurate.

Bootstrap is a good tool to **estimate the true distribution of statistic (**$\sqrt{n} (\hat \theta - \theta_0)$ or $\frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{\hat V}}$**) by resampling**, thus to construct CI. [Resampling means to generate artificial samples using the sample data].

### Pairwise Bootstrap

Look at how to apply Pairwise Bootstrap on the previous example:

1. Draw with replacement n observations from $(Z_1,...,Z_n)$ to get $(Z_1^*,...,Z_n^*)$. [In the following, star * means quantities computed using boostrap samples]

2. Use $(Z_1^*,...,Z_n^*)$ to compute statistic in Bootstrap world:  $\hat t^* = \frac{(\hat \theta^* - \hat \theta) }{\hat \sigma_n^*}$ [In this example, $\hat \theta^*$ is the mean of Bootstrap samples; $\hat \sigma_n^* = \frac{1}{\sqrt{n}} \ \text{or} \ \sqrt{\frac{\hat V^*}{n}}$ ]

3. Repeat the 2 previous steps B times to get $(\hat t_1^*, …, \hat t_B^*)$.

4. Compute the quantile of statistic $\hat t^* = \frac{(\hat \theta^* - \hat \theta) }{\hat \sigma_n}$: $\hat q_{\beta}$ and $\hat q_{1- \alpha}$.

5. Plug them in (because the distribution of $\hat t = \frac{(\hat \theta - \theta_0) }{\hat \sigma_n}$ is approximated by the distribution of $\hat t^*$ )

   $$
   P(\hat q_{\beta} \le \frac{(\hat \theta - \theta_0) }{\hat \sigma_n} \le \hat q_{1- \alpha}) = 1- \alpha-\beta 
   $$

One remaining question: how to choose $\hat \sigma_n$ ?

Conclusion: if the asymptotic distribution of statistic $\hat t$ does not depend on the distribution of data, it is called **asymptotically pivotal**. In this case, **Boostrap could provide refinements** compared to asymptotic approximation. 

If $\hat \sigma_n = \frac{1}{\sqrt{n}}$, $\hat t = \sqrt{n}(\hat \theta - \theta_0)  \sim \mathcal{N}(0, \hat V)$. In this case, $\hat t$ is not asymptotically pivotal. Nothing is lost by boostrapping, but nothing is gained compared to asymptotic approximation.

If $\hat \sigma_n = \sqrt{\frac{\hat V}{n}}$, $\hat t = \frac{\sqrt{n} (\hat \theta - \theta_0) }{\sqrt{\hat V}} \sim \mathcal{N}(0, I)$. In this case, $\hat t$ is asymptotically pivotal and Boostrap provides more accurate confidence interval compared to asymptotic method.

Thus, if possible (the asymptotic variance is not too complex), it’s better to use a statistic which is asymptotically pivotal.

### Delta method

In the previous example, we proved that $\hat t^* = \sqrt{n}(\hat \theta^* - \hat \theta)$  estimates consistently the distribution of $\hat t = \sqrt{n}(\hat \theta - \theta_0)$. 

By using delta method, we can prove that given a smooth function $\phi$, we have $\sqrt{n}(\phi(\hat \theta^*) - \phi(\hat \theta))$ estimates consistently the distribution of $\sqrt{n}(\phi(\hat \theta) - \phi(\theta_0))$. 

#### Example: OLS estimators

We observe $\{Y_i, Z_i\}_{i=1}^n$ , given the regression model $Y_i = Z^T \beta_0 + \epsilon \ \  \text{with} \ \mathbb{E}[\epsilon Z]=0$, we are interested in constructing CI for $\beta_0 = (\mathbb{E}[ZZ^T])^{-1} \mathbb{E}[ZY]$

OLS estimator of $\beta_0$ is $\hat \beta = (\overline{ZZ^T})^{-1} \overline{ZY}$. We have:

$$
\sqrt{n}(\hat \beta - \beta_0) \sim \mathcal{N}(0, \Sigma) \ \ \text{with} \ \Sigma=(\mathbb{E}[ZZ^T])^{-1}(\mathbb{E}[\epsilon^2ZZ^T])(\mathbb{E}[ZZ^T])^{-1}
$$

This time, the estimator $\hat \beta$ is a smooth transformation of sample mean. Thus, with delta method, we obtain:

$$
P(\hat q_{\beta} \le \sqrt{n}(\hat \beta - \beta_0) \le \hat q_{1- \alpha}) = 1- \alpha-\beta 
$$

with $\hat q_{\beta}$ and $\hat q_{1- \alpha}$ are quantiles of boostrap statistic $\sqrt{n}(\hat \beta^* - \hat \beta)$

We can also prove the consistency of **asymptotically pivotal statistic** $\hat \Sigma^{-1/2} \sqrt{n}(\hat \beta - \beta_0)$:

$$
P(\hat q_{\beta}' \le \hat \Sigma^{-1/2}\sqrt{n}(\hat \beta - \beta_0) \le \hat q_{1- \alpha}') = 1- \alpha-\beta 
$$

with $\hat q_{\beta}'$ and $\hat q_{1- \alpha}'$ are quantiles of statistic $(\hat \Sigma^*)^{-1/2} \sqrt{n}(\hat \beta^* - \hat \beta)$

#### Example: 2SLS estimators

### Bootstrap for kernel density estimation

Context: Given samples $\{X_i\}_{i=1}^n$, use kernel functions to estimate the density **at one fixed point x**.

$\hat f(x) = \frac{1}{nh} \sum_{i=1}^n K(\frac{x_i - x}{h})$

Use boostrap to construct CI for true density f(x):

1. Draw with replacement n observations from $\{X_i\}_{i=1}^n$, noted as $\{X_i^*\}_{i=1}^n$

2. Compute the bootstrap statistic $\hat t^* = \frac{\sqrt{nh} (\hat f^*(x) - \hat f(x))}{ \hat s^*(x) }$, where $\hat s(x)^2 = \frac{1}{nh} \sum_{i=1}^n K(\frac{x_i - x}{h})^2 - h*[\hat f(x)]^2$

3. Repeat two previous steps B times, getting $\{ \hat t^*_i \}_{i=1}^B$

4. Using B bootstrap statistics to compute its quantiles

5. Getting CI for f(x) with ($1-\alpha -\beta$) level:

   $$
   [\hat f(x) - \frac{\hat s(x)}{\sqrt{nh}} \hat q_{1-\alpha}, \ \ \hat f(x) - \frac{\hat s(x)}{\sqrt{nh}} \hat q_{\beta}]
   $$

---

### Other applications:

- Bootstrap for M-estimators
- Bootstrap for Generalized method of Moments (GMM)

### Other resampling scheme

- Residual bootstrap
- wild bootstrap