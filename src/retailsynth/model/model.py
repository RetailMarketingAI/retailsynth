import numpy as np
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import mask
from scipy import stats


# Generate some synthetic data
np.random.seed(0)
N = 10000 # number of observations
D = 3 # number of features
n_products = 2# number of products

X = np.random.randn(N, D)

# Generate synthetic true parameters
true_beta_Y0 = np.random.randn(1, D)
true_intercept_Y0 = 10*np.random.randn(1)
logits_Y0 = np.dot(X, true_beta_Y0.T).flatten() + true_intercept_Y0
probabilities_Y0 = 1 / (1 + np.exp(-logits_Y0))
y0 = np.random.binomial(1, probabilities_Y0)

true_beta_Y1_given_Y0 = np.random.randn(n_products, D)
true_intercept_Y1_given_Y0 = np.random.randn(n_products)
X_Y0 = X[y0 == 1]

logits_Y1_given_Y0 = np.dot(X_Y0, true_beta_Y1_given_Y0.T) + true_intercept_Y1_given_Y0
probabilities_Y1_given_Y0 = 1 / (1 + np.exp(-logits_Y1_given_Y0))
y1 = np.zeros((N, n_products), dtype=int)
y1[y0 == 1] = np.random.binomial(1, probabilities_Y1_given_Y0)

# Convert data to JAX arrays
X = jnp.array(X)
y0 = jnp.array(y0)
y1 = jnp.array(y1)

def logit_choice_model(X, name_prefix, n_outputs):
    n_features = X.shape[1]
    beta = numpyro.sample(f'{name_prefix}_beta', dist.Normal(jnp.zeros((n_outputs, n_features)), jnp.ones((n_outputs, n_features))))
    intercept = numpyro.sample(f'{name_prefix}_intercept', dist.Normal(jnp.zeros(n_outputs), 1.))
    linear_combination = jnp.einsum('ij,kj->ik', X, beta) + intercept
    return jax.nn.sigmoid(linear_combination)

def simple_model(X, n_products, y0=None, y1=None):
    # Model P(Y0 | X)
    P_Y0 = logit_choice_model(X, 'Y0', 1).squeeze()

    # Sample Y0
    y0_sample = numpyro.sample('y0', dist.Bernoulli(P_Y0), obs=y0)  

    # Model P(Y1 | Y0 = 1, X)
    P_Y1_given_Y0 = logit_choice_model(X, 'Y1_given_Y0', n_products)  

    with numpyro.plate('products', n_products, dim=-1):
        with numpyro.plate('data_y1', X.shape[0]):
               numpyro.sample('y1', dist.Bernoulli(P_Y1_given_Y0), obs=y1)

def masked_model(X, n_products, y0=None, y1=None):
    # Model P(Y0 | X)
    P_Y0 = logit_choice_model(X, 'Y0', 1).squeeze()

    # Sample Y0
    y0_sample = numpyro.sample('y0', dist.Bernoulli(P_Y0), obs=y0)  

    # Masking to filter out Y1 calculations when Y0 is 0
    mask_array = (y0_sample == 1)[:, None]

    # Model P(Y1 | Y0 = 1, X)
    P_Y1_given_Y0 = logit_choice_model(X, 'Y1_given_Y0', n_products)  

    with numpyro.plate('products', n_products, dim=-1):
        with numpyro.plate('data_y1', X.shape[0]):
            with mask(mask=mask_array):
               numpyro.sample('y1', dist.Bernoulli(P_Y1_given_Y0), obs=y1)



def get_predictive_posterior_samples(model):          
    # Define the NUTS sampler
    nuts_kernel = NUTS(model)

    # Run MCMC to sample from the posterior
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), X, n_products, y0, y1)
    mcmc.print_summary()
    return mcmc.get_samples()

    # # Predictive posterior sampling
    # predictive = Predictive(combined_model, posterior_samples=mcmc.get_samples())
    # predictions = predictive(jax.random.PRNGKey(1), X, n_products)

    # # Extract predictions and compute mean predicted probabilities
    # predicted_y0_probs = predictions['y0'].mean(axis=0)
    # predicted_y1_probs = predictions['y1'].mean(axis=0)


samples1 = get_predictive_posterior_samples(simple_model)

samples2 = get_predictive_posterior_samples(masked_model)

# def compare_samples(samples1, samples2):
#     for ((samples1_param_name, samples1_param_values), (samples2_param_name, samples2_param_values)) in zip(samples1.items(), samples2.items()):
#         t_test_mu = stats.ttest_ind(samples1_param_values[:], samples2_param_values[:])
#         print(f"T-test for {samples1_param_name}: p-value={t_test_mu.pvalue}")

# compare_samples(samples1, samples1)
import ipdb; ipdb.set_trace()