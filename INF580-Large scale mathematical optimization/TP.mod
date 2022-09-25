#project model
#TP model

# dimension of A,B,c and x
param m integer ,>0;
param n integer ,>0;
param k integer ,>0;

set M := 1..m;
set N := 1..n;
set K := 1..k;

param A {M,N} default 0;
param B {M} default 0;
param T {K,M} default 0;
param c {N} default 0;

param TA {K,N} default 0;
param TB {K} default 0;

var x{N} >=0;

minimize objectTP: sum{i in N} c[i]*x[i];

subject to constraintTP {i in K}: sum{j in N} TA[i,j]*x[j] = TB[i] ;