// see fit_mod_spline_comb_iid_race_more_knots.r for model description

#include <TMB.hpp>
using namespace density;
using Eigen::SparseMatrix;
#include "lcar_strmat.hpp"
#include "prior_param_struct.hpp"
#include "prior_penalty.hpp"

// Objective function
template<class Type>
  Type objective_function<Type>::operator() () {
    
    // Define data and inputs
    DATA_VECTOR(Y);     // deaths
    DATA_VECTOR(N);     // population
    DATA_MATRIX(X);     // covariates (at minimum, a column for the intercept)
    DATA_IVECTOR(J);    // area indicator
    DATA_IVECTOR(T);    // year indicator
    DATA_IVECTOR(A);    // age indicator
    DATA_IVECTOR(R);    // race indicator
    
    DATA_SPARSE_MATRIX(graph_j);  // neighborhood structure
    DATA_SPARSE_MATRIX(graph_t);
    DATA_SPARSE_MATRIX(graph_a);
    
    
    // Define parameters
    // fixed effects (intercept, covariate effects)
    PARAMETER_VECTOR(B);
    
    // RE2: area-level random intercept
    PARAMETER_VECTOR(re1); 
    PARAMETER(re1_log_sigma);
    Type sigma_1 = exp(re1_log_sigma);
    PARAMETER(logit_rho_1);
    Type rho_1 = invlogit(logit_rho_1);
    DATA_STRUCT(re1_prior_param, prior_type_sigma);
    
    // NLL contribution from random effects
    Type nll = 0;
    max_parallel_regions = omp_get_max_threads();
    
    // RE2: area-level random intercept
    SparseMatrix<Type> K_2 = lcar_strmat(graph_j, rho_1); 
    PARALLEL_REGION nll += SCALE(GMRF(K_2), sigma_1)(re1);
    
    Type pen1 = eval_prior_sigma(re1_prior_param, re1_log_sigma);
    
    PARALLEL_REGION nll -= pen1;
    
    REPORT(pen1);
    
    PARALLEL_REGION nll -= dnorm(logit_rho_1, Type(0), Type(1.5), true);
    
    // NLL contribution from data
    // predictions
    vector<Type> log_m = X * B;
    for(size_t i = 0; i < Y.size(); i++) {
      log_m[i] += re1[J[i]];
    }
    
    
    vector<Type> m = exp(log_m);
    
    
    // data likelihood
    for(size_t i = 0; i < Y.size(); i++)
      PARALLEL_REGION nll -= dpois(Y[i], N[i]*m[i], true);
    
    return nll;
  }
