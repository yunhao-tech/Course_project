##### TP d’Introduction au langage R #####
##### Author: Yunhao CHEN ################
##### Date: 2022/10/11 ###################

# Exercice 1 - Efficient P rogrammation
zeta <- function(n, s) {
    return(sum((1 / (1:n))^s))
}

ppositive <- function(x) {
    return(pmax(x, 0))
}

deriv1 <- function(f, a, b, h) {
    return(diff(f(seq(a, b, h))) / h)
}

# Exercice 2 - Simple Importation of data
# 1. read data
x <- read.table("https://www.nicolasbaradel.fr/R/donnees/N_n.txt", header = FALSE, sep=";", dec = ",")
# 2. replace NA by 0
x[is.na(x)] <- 0
# 3. transform data to matrix
x <- matrix(unlist(x), nrow = 6, ncol = 6)
is.matrix(x)
# 4. compute mean by rows
rowMeans(x)

# Exercice 3 - Simulation and Monte Carlo
# 1.
n <- 15
N <- rpois(n, lambda = 5)
# 2.
M <- matrix(0, nrow = n, ncol = max(N))
for (i in 1:n) {
    M[i, 1:N[i]] <- rlnorm(N[i], meanlog = 11, sdlog = 2) + 10^5
}
# 3. difficult (efficient programmation)

# 4.
S <- rowSums(M)
# 5.
quantile_S <- function(n) {
    N <- rpois(n, lambda = 5)
    M <- matrix(0, nrow = n, ncol = max(N))
    for (i in 1:n) {
        if (N[i] > 0) {
            M[i, 1:N[i]] <- rlnorm(N[i], meanlog = 11, sdlog = 2) + 10^5
        }
    }
    return(quantile(rowSums(M), 0.995))
}
print(quantile_S(10^5))
