#project model
#P model

# dimension of A,B,c and x
param m integer ,>0;
param n integer ,>0;

set M := 1..m;
set N := 1..n;

#not used
param k;
set K := 1..k;
param T {K,N} default 0;

param A {M,N} default 0;
param B {M} default 0;
param c {N} default 0;

param TA {K,N} default 0;
param TB {K} default 0;

var x{N} >=0;

minimize objectP: sum{i in N} c[i]*x[i];

subject to constraintP {i in M}: sum{j in N} A[i,j]*x[j] = B[i] ;