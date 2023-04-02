                                ## Function to use in the Simulations/Data Implementations

## Section 1. The functions below have been written for the OLS model. They work with "data" in "matrix" format or "data.frame" format # nolint

betahat.fun <- function(data){ ## OLS estimator # nolint
  lmodel <- lm(y ~ x1 + x2, data = data)
  betahat <- as.vector(lmodel$coefficients)
  return(betahat)
}

T_i <- function(Tn, data, i){ ## This computes the "leave-one-out" version of the estimator "Tn"
  ## NB: here "data" must be "data.frame" or "matrix" format
  data_i <- data[-i, ] #" This drops the ith observation from the data
  T_i <- Tn(data = data_i)
  return(T_i)
}

T.tilde.fun <- function(Tn, n, data){ ## This computes the sequence of "leave-one-out" estimators; 
  T.tilde <- t(sapply(1:n, function(e) T_i(Tn=Tn, data=data , i=e))) # This matix has at the ith row the estimator omitting the ith observation
  return(T.tilde)
}

# tjack.fun <- function(n, data){ ## t-statistic for OLS with "jackknifed" variance
#   betahat_OLS <- betahat.fun(data = data)
#   T.tilde <- T.tilde.fun(Tn = betahat.fun, n = n, data = data)
#   vjack <- ((n - 1) / n) * (n - 1) * cov(T.tilde)
#   # sigma_jack <- solve(a=diag(vjack) * pracma::eye(length(betahat_OLS))) # return the inverse of a
#   sigma_jack <- solve(a=vjack) # return the inverse of a
#   tjack <- sqrt(sigma_jack) %*% (betahat_OLS - 1) ## %*% means matrix multiplication
#   # The true parameters are (1, 1, 1)
#   return(tjack)
# }

# tOLS.fun <- function(n, data){ ## t-statistic for OLS with Heteroskedasticity robust estimator; 
#   lmodel <- lm(y ~ x1 + x2, data = data)
#   WHE <- sandwich::vcovHC(lmodel, type = "HC0") ## White-Huber HCCE ; type="HC0" for heteroskedasticity robust estimator
#   # sigma_OLS <- solve(a=diag(WHE) * pracma::eye(nrow(WHE))) # return the inverse of a
#   sigma_OLS <- solve(a=WHE) # return the inverse of a
#   tOLS <- sqrt(sigma_OLS) %*% (as.vector(lmodel$coefficients)-1) # true parameters are (1, 1, 1)
#   return(tOLS)
# }

DGP_B <- function(n, gamma) {## Data Generating Process; gamma= magnitude of the heteroskedasticity
    x1 <- rnorm(n, 0, 1)
    x2 <- rnorm(n, 0, 1)
    v1 <- rnorm(n, 0, 1)
    v2 <- rnorm(n, 0, 1)
    u <- rnorm(n, 0, 2)
    eps <- u + v1 * x1 * gamma + v2 * x2 * gamma
    y <- 1 + x1 + x2 + eps
    data.frame(y = y, x1 = x1, x2 = x2)
}

sigma.jack.fun <- function(n,data){ ## t-statistic for OLS with "jackknifed" variance; 
  T.tilde <- T.tilde.fun(Tn=betahat.fun, n=n, data=data) # sequence of "leave-one-out" estimators
  vjack <- ((n-1)/n) * (n-1)*cov(T.tilde) # (n-1)*cov(T.tilde) = sum([T_{n-1,i} - mean(T_{n-1,i})]^2)
  sigma_jack <-  sqrt(diag(vjack)) # Only take standard error of parameters, drop the correlation terms
  return(sigma_jack)
}