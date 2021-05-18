#### Test model using normalize()

library(R.utils)
library(data.table)
library(TMB)
library(optimx)
library(splines)

set.seed(98121)

## settings ------------------------------------------------------------------------------------

sex <- 1 # either 1 or 2

if (!dir.exists("outputs")) {
  dir.create("outputs")
}

## Format data for TMB -----------------------------------------------------------------------------
cat("\n\n***** Format data\n"); flush.console()

load("re_graphs.rdata")
data <- readRDS("nvss_dummy_data_race_ethn.rds")
setnames(data, "mcnty", "area")
data <- data[sex == get("sex", .GlobalEnv), ]
data[, race := as.integer(as.factor(race)) - 1]

data[, int := 1L]

num_j <- max(data$area) + 1
num_t <- max(data$year) + 1
num_a <- max(data$age) + 1
num_r <- max(data$race) + 1

prior_type <- "pc"
prior_list <- list(c(5,0.05,-5))

j <- 1
assign(paste0("re",j,"_par1"), prior_list[[j]][1])
assign(paste0("re",j,"_par2"), prior_list[[j]][2])
assign(paste0("re",j,"_log_sigma"), prior_list[[j]][3])

ii <- which(data$pop > 0)
tmb_data <- list(
  Y = data$deaths[ii],
  N = data$pop[ii],
  J = data$area[ii],
  A = data$age[ii],
  T = data$year[ii],
  R = data$race[ii],
  X = as.matrix(data[ii, c("int"), with = F]),
  graph_j = graph_j,
  graph_t = graph_t,
  graph_a = graph_a,
  re1_prior_param = list(type = prior_type, par1 = re1_par1, par2 = re1_par2))

## Set parameters for TMB --------------------------------------------------------------------------
cat("\n\n***** Set parameters\n"); flush.console()
tmb_par <- list(
  B = rep(0, 1), # intercept & covariate effects
  
  re1 = rep(0, num_j), # area-level LCAR random intercept
  re1_log_sigma = re1_log_sigma,
  logit_rho_1 = 0)

map <- NULL

message("Done initializing data")

## Fit model ---------------------------------------------------------------------------------------
# compile CPP code for objective function
TMB::compile("USHD_fast.cpp")
dyn.load(dynlib("USHD_fast"))
config(tape.parallel = 0, DLL = "USHD_fast")

# make objective function
cat("\n\n***** Make objective function\n"); flush.console()
tmb_data$flag <- 1  ## include data
obj <- MakeADFun(tmb_data, tmb_par, random = c("B", paste0("re", 1:1)), DLL = "USHD_fast", map = map)
obj <- normalize(obj, flag = "flag")

saveRDS(obj$report(),  file = paste0("outputs/fast_model_fit_OPT_1_", sex, ".rds"))

# optimize objective function
cat("\n\n***** Optimize objective function\n"); flush.console()
opt_time <- proc.time()
nlminb(obj$par, obj$fn, obj$gr)

(opt_time <- proc.time() - opt_time)

# get standard errors
cat("\n\n***** Extract standard errors\n"); flush.console()
se_time <- proc.time()
saveRDS(obj$report(),  file = paste0("outputs/fast_model_fit_race_test_OPT_", sex, ".rds")) # temporarily save this so that we can see the covariance matrix

out <- sdreport(obj, getJointPrecision = T)
(se_time <- proc.time() - se_time)

# save model output
cat("\n\n***** Save model output\n"); flush.console()
saveRDS(out, file = paste0("outputs/fast_model_fit_race_test_", sex, ".rds"))
saveRDS(rbind(se_time, opt_time), file = paste0("outputs/fast_model_fit_time_race_test_", sex, ".rds"))
