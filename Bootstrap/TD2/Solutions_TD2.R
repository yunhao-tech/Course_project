setwd("D:/VScode/Bootstrap/TD2")
source("bits.R")
                                     
                                                      ## Exercise 1

## Item 1: A bootstrap estimator of the variance of betahat is mean((betahat.star - mean(betahat.star))^2), 
## where betahat.star is a vector whose generic component is the OLS estimator computed with a bootstrap sample
## and the number of components of betahat.star is equal to the number of bootstrap replications

## Item 2: 
varboot.fun <- function(n,data, B){ ## This function computes the t-stat with a bootstrap estimator of the variance 
  betahat.mat <- matrix(rep(NA, B*3), nrow=3, ncol=B) ## 3=number of OLS coefficients (including the intercept)
  for(b in 1:B){
    datab <- data[sample(n, size = n, replace = TRUE), ] ## single bootstrap extraction
    betahat.boot <- betahat.fun(data=datab)
    betahat.mat[,b] <- betahat.boot
  }
  varhat.boot <- ((B-1)/B)*cov(t(betahat.mat))
  return(varhat.boot)
}

## Item 3:
n <- 500 # number of observations
gamma <- 1 ## benchmark gamma=1
B <- 100 ## number of bootstrap replications
data <- DGP_B(n=n, gamma=gamma)
betahat <- as.vector(betahat.fun(data=data))
var.boot <- varboot.fun(n=n, data=data, B=B)
sigma.boot <- sqrt(diag(var.boot))

alpha <- 0.05 ## nominal size of the test
q_1.minus.alpha <- qnorm(1-(alpha)/2, mean=0, sd=1) #" This is the quantile we will compare our statistic in EQUATION 4 to get the confidence interval
CI.sigmaboot.mat <- matrix(rep(NA, 2*length(betahat)), nrow=length(betahat), ncol=2 )
CI.sigmaboot.mat[,1] <- betahat - sigma.boot * q_1.minus.alpha
CI.sigmaboot.mat[,2] <- betahat + sigma.boot * q_1.minus.alpha
rownames(CI.sigmaboot.mat) <- c("CI for beta0 based on the bootstrap var. est.", "CI for beta1 based on the bootstrap var. est." , 
                                "CI for beta2 based on the bootstrap var. est.")
print(CI.sigmaboot.mat)

## Item 4: The bootstrap of an asyptotically pivotal statistic gives refinement compared to the asymptotic distribution.
## Differently, an asymptotic approximation or bootstrapping a statistic that is not asymptotically pivotal give less accurate
## approximations of the true distribution of the statistic


                                                  ## Exercise 2

## Item 1: a bootstrap estimator of the bias of thetahat is mean(thetahat.star)-thetahat, where thetahat.star is a vector whose generic 
## component is the estimator thetahat computed with a bootstrap sample, and thetahat.star has as many components as the number 
## of bootstrap replications

## Item 2: theta.hat.boot = theta.hat - [mean(thetahat.star)-thetahat]

## Item 3: 
## The function computing the uncorrected estimator is
thetahat.fun <- function(data){ ## here data must be in vector format
  thetahat <- exp(mean(data))
  return(thetahat)
}
## The function computing the bootstrap bias corrected estimator is the following:
thetahat.bc.boot.fun <- function(data, n, B){ ## It returns a Bootstrap bias-corrected estimator;; here "data" must be in vector format
  thetahat <- thetahat.fun(data=data)
  thetahat.boot.vec <- rep(NA, B)
  for(b in 1:B){
    datab <- data[sample(n, size = n, replace = TRUE) ] ## single bootstrap extraction
    thetahat.boot <- thetahat.fun(data=datab)
    thetahat.boot.vec[b] <- thetahat.boot
  }
  boot.bias <- mean(thetahat.boot.vec) - thetahat
  thetahat.bc <- thetahat - boot.bias ## Bias corrected estimator
  return(thetahat.bc)
}


## Item 4:
library(pracma)
library(foreach)
library(doParallel)
library(doRNG)
library(compiler)

n <- 30 ## number of observations
B <- 599
M <- 1000 ## number of Monte Carlo replications for the simulation


set.seed(1234567)
nrcore <- 5 ## number of cores of the computer that will be used here. Leave at least one core free for your computer 
cl <- makeCluster(mc <- getOption("cl.cores", nrcore))
registerDoParallel(cl)


tic()
Bias_simul <- foreach(m=1:M, .combine=cbind)%dorng%{ ## this is the "for" loop that runs the simulations. Notice that it is parallelized
  data <- rnorm(n,0,sd=sqrt(6)) #" mu=0, so theta0:=exp(mu)=1
  thetahat <- thetahat.fun(data=data)
  thetahat.boot <-  thetahat.bc.boot.fun(data=data,n=n,B=B)
  return(c(thetahat, thetahat.boot))
}
toc()

stopCluster(cl)

res <- matrix(rep(NA,4), nrow=2,ncol=2)
res[,1] <- rowMeans(Bias_simul - 1)
res[,2] <- c(  mean((Bias_simul[1,]-1)^2) , mean((Bias_simul[2,]-1)^2)  ) ## The population value of the parameter is theta0=1
rownames(res) <- c("No Bias correction", "Bootstrap bias correction")
colnames(res) <- c("Bias" , "MSE")
print(res)

## Item 5: From the results of the experiment we realize that the bootstra bias corrected estimator is 
## less biased and more precise than the uncorrected estimator (i.e. it has a lower Mean Squared Error)

