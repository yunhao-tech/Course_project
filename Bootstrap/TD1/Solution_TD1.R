                                      ## SOLUTION TO EXERCISE 1
setwd('D:/VScode/Bootstrap/TD1')
source("bits.R")

library(pracma)
library(foreach)
library(doParallel)
library(doRNG)
library(compiler)

n <- 50 # number of observations
gamma <- 1 ## benchmark gamma=1

## Item 1)
data <- DGP_B(n=n, gamma=gamma) ## This extracts a sample fom the DGP
betahat <- as.vector(betahat.fun(data=data))

## Item 2) Construct the CI with the White estimator
lmodel <- lm(y ~ x1 + x2, data = data)
WHE <- sandwich::vcovHC(lmodel, type = "HC0")
sdWHE <- sqrt(diag(WHE)) # Standard error of three parameters in regression

CIAsy.mat <- matrix(rep(NA, 2*length(betahat)), nrow=length(betahat), ncol=2)
CIAsy.mat[,1] <- betahat - sdWHE * qnorm(0.975)
CIAsy.mat[,2] <- betahat + sdWHE * qnorm(0.975)
# CI Asymp is in form: [estimate of parameter -(+) standard error of parameter * qnorm(0.975)]
rownames(CIAsy.mat) <- c("CI Asy for beta0", "CI Asy for beta1" , "CI Asy for beta2")
print(CIAsy.mat)

## Item 3) CI based on the Jaccknife estimator of the Asymptotic variance
sigma.jack <- sigma.jack.fun(n=n,data=data) ## This a a function that you had to construct. it is now contained in the "bits.R" source file
CIJack.mat <- matrix(rep(NA, 2*length(betahat)), nrow=length(betahat), ncol=2)
CIJack.mat[,1] <- betahat - sigma.jack * qnorm(0.975)
CIJack.mat[,2] <- betahat + sigma.jack * qnorm(0.975)
rownames(CIJack.mat) <- c("CI Jack for beta0", "CI Jack for beta1" , "CI Jack for beta2")
print(CIJack.mat)

## Item 4) We should obtain that the CI based on the White estimator are tighter than the CI based on the Jackknife variance estimator.
## Now, the tighter a CI the higher the probability that a t-test based on it will reject the null hypothesis. This is connected with what 
## we saw in class: the t-test based on the White estimator was rejecting a way more than the nominal level, 
## while the t-test based on the Jackknife was rejecting with a proportion close to the nominal level.
## So, the worse behavior of the test based on the White estimator comes from the fact that the White estimator of the asymptotic variance
## does not behave well. We fix this by using the Jackknife variance estimator





                                           ## SOLUTION TO EXERCISE 2
##  Solutions to Item 1) Since thta_0=exp(E(x)) and mean(x) (i.e. the sample mean of x) estimates consistently E(x), 
##  an estimator of theta_0 is thetahat=exp(mean(x))
## An R function for thetahat is as follows:

thetahat.fun <- function(data){ ## here data must be in vector format
  thetahat <- exp(mean(data))
  return(thetahat)
}

## Solution to Item 2) The expression of the jackknife bias estimator is in expression 1.1 page 5, with 
##   T_n given by the first display in page 10 and with g()=exp(). The resulting Jackknife corrected bias estimator is in 
## Equation 1.2 page 6.
#" The R function for it is given by
thetahat.jack.fun <- function(data, n){ ## It returns a Jakckknife bias-corrected estimator; here data must be in vector format
  thetahat.i <- t(sapply(1:n, function(e) thetahat.fun(data=data[-e])))
  thetahat <- thetahat.fun(data=data)
  bias.jack <- (n-1)*(mean(thetahat.i)- thetahat)
  thetahat.bc <- thetahat - bias.jack
  return(thetahat.bc)
}

## Solution to Item 3) The Monte Carlo experiment to compare the performances of thetahat and thetahat_jack is as follows:

library(pracma)
library(foreach)
library(doParallel)
library(doRNG)
library(compiler)
n <- 30 ## number of observations
M <- 5000 ## number of Monte Carlo replications for the simulation


set.seed(1234567)
nrcore <- 5 ## number of cores of the computer that will be used here. Leave at least one core free for your computer 
cl <- makeCluster(mc <- getOption("cl.cores", nrcore))
registerDoParallel(cl)


tic()
Bias_simul <- foreach(m=1:M, .combine=cbind)%dorng%{ ## this is the "for" loop that runs the simulations. Notice that it is parallelized

  data <- rnorm(n, 0, sd=sqrt(6)) #" mu=0, so theta0:=exp(mu)=1
  thetahat <- thetahat.fun(data=data)
  thetahat.jack <-  thetahat.jack.fun(data=data, n=n)
  return(c(thetahat, thetahat.jack))
}
toc()

stopCluster(cl)

res <- matrix(rep(NA,4), nrow=2, ncol=2)
res[,1] <- rowMeans(Bias_simul - 1) ## This computes the bias of each estimator and takes the mean
res[,2] <- c(mean((Bias_simul[1,]-1)^2), mean((Bias_simul[2,]-1)^2)) #" This computes the MSE of each estimator
rownames(res) <- c("No Bias correction", "Jackknife bias correction" )
colnames(res) <- c("Bias", "MSE")
print(res)