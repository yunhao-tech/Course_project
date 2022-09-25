#project model
#TD model

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

var u{K};

maximize objectTD0: sum{i in K} u[i]*TB[i];

subject to constraintTD0 {i in N}: sum{j in K} u[j]*TA[j,i] <= c[i] ;