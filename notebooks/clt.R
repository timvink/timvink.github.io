
#### Setup ### 
# Note: Save images in 700x300px for blogpost

library(tidyverse)
library(ggplot2)
library(gridExtra)
library(brms)
set.seed(1337) 
options(scipen=10)

#### Central Limit Theorem #### 

population <- rweibull(10e5, shape = 1.5, scale = 1)

# Generate data
generate_sample_means <- function(population, sample_size, n_samples) {
  rerun(n_samples, 
        population %>% sample(sample_size) %>% mean) %>% 
    unlist
}
sample_means <- generate_sample_means(population, sample_size = 100, n_samples = 1000) 

# Visualize sample means
p1 <- tibble(x = population) %>% 
  ggplot(aes(x = x)) + geom_histogram(bins = 100) + 
  ggtitle(glue::glue("'True' population with mean {round(mean(population),4)}"), 
          subtitle = glue::glue("100k samples from Weibull(λ = 1, k = 1.5)"))
p2 <- tibble(x = sample_means) %>% 
  ggplot(aes(x = x)) + geom_histogram(bins = 50) +
  #geom_density(aes(y=..density..*5), linetype = 2) +
  ggtitle(glue::glue("Mean of means {round(mean(sample_means), 4)}"), 
          subtitle = glue::glue("Means of 1000 samples of size 100"))
grid.arrange(p1, p2, nrow = 1)

# Visualize effect of sample size
tibble(sample_size = c(50,100,500,1000)) %>% 
  mutate(sample_means = map(sample_size, generate_sample_means, 1000)) %>% 
  unnest %>%
  ggplot(aes(x = sample_means)) +
  geom_histogram(bins = 100) +
  coord_cartesian(xlim = c(.6, 1.2)) + 
  facet_wrap(~ sample_size, nrow = 1, scales = "free_y")

# p-value calc
pnorm(abs(1.33333), lower.tail = F)


#### Linear regression from stratch ####

# The data
X <- mtcars %>% 
  select(wt, hp, qsec, am) %>% 
  mutate(intercept = 1) %>% 
  as.matrix
y <- as.matrix(mtcars$mpg)

# Estimate coefficients using the normal equation
beta <-  matlib::inv(t(X) %*% X) %*% (t(X) %*% y)
custom_model <- tibble(feature = c("wt","hp","qsec","am", "intercept"), 
       coefficients = as.numeric(beta))
custom_model

## Calculate standard error by hand
y_hat <- X %*% beta
rss <- sum((y - y_hat)^2)
degrees_of_freedom <- (nrow(mtcars) - 4 - 1)
RSE <- sqrt(rss / degrees_of_freedom) # Residual standard error 
RSE
#> 2.435 

# Calculate the z-score and the p-values
matrixinv <- matlib::inv(t(X) %*% X) 
diagonals <- diag(matrixinv)
zscores <- beta / (RSE * sqrt(diagonals))
custom_model <- custom_model %>% 
  mutate(z_score = as.numeric(zscores),
         p_value = 2 * pnorm(abs(z_score), lower.tail = F))
custom_model

# Validate custom LR with R's lm()
model <- lm(mpg ~ wt + hp + qsec + factor(am), data = mtcars)
model$coefficients
summary(model) # we should drop horsepower hp


#### Bayesian Linear Regression ####

custom_mtcars <- mtcars %>% 
  mutate(am = factor(am))

#### Approach using MCMCPack
# https://rdrr.io/cran/MCMCpack/man/MCMCregress.html
library(MCMCpack)

posterior <- MCMCregress(mpg ~ wt + hp + qsec + am, 
                          b0=0, B0 = 0.1,
                          sigma.mu = 5, sigma.var = 25, data=custom_mtcars, verbose=1000)

# Diagnostics on model
plot(posterior)
raftery.diag(posterior)
autocorr.plot(posterior)
summary(posterior)

#### Approach using brms

fit <- brm(mpg ~ wt + hp + qsec + am,
           data = custom_mtcars, family = gaussian(), chains = 2)
# Show estimates
fixef(fit)

# Get MCMC Samples
samples <- fit %>% 
  brms::as.mcmc() %>% 
  as.matrix() %>% 
  as_tibble()

# Visualize
plot_distribution <- function(data, feature_name) {
  data %>% 
    pull(feature_name) %>% 
    tibble(feature = .) %>% 
    ggplot(aes(x=feature))+
    geom_density(color="darkblue", fill="lightblue") +
    ggtitle(glue::glue("Posterior distribution of {feature_name}"))
}
samples %>% plot_distribution("b_wt")

# Now define priors and fit a new model
priors <- c(
  set_prior("normal(25, 15)", class = "Intercept"),
  set_prior("normal(0, 10)", class = "b", coef = "wt"),
  set_prior("normal(0, 10)", class = "b", coef = "hp"),
  set_prior("normal(0, 10)", class = "b", coef = "qsec"),
  set_prior("normal(0, 10)", class = "b", coef = "am1")
)

fit2 <- brm(mpg ~ wt + hp + qsec + am,
           data = custom_mtcars, 
           prior = priors,
           family = gaussian(), chains = 2)
fixef(fit2)

samples_fit2 <- fit2 %>% 
  brms::as.mcmc() %>% 
  as.matrix() %>% 
  as_tibble()

# Visualize shift in estimates after specifying priors
tibble(
  "flat prior" = samples$b_wt,
  "N(0,10) prior" = samples_fit2$b_wt
) %>% 
  tidyr::gather(prior, sample) %>% 
  ggplot(aes(x=sample)) +
  geom_density(aes(fill=prior), alpha = .3) + 
  ggtitle("Shift in posterior density of weight wt")

# plot 
plot(fit2)
plot(fit2, pars = c("wt", "hp", "qsec", "am1"))
plot(marginal_effects(fit2, effects = "wt"))



