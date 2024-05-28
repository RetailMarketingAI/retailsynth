import numpy as np
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import mask
from scipy import stats


def logit_choice_model(X, name_prefix, n_outputs):
    n_features = X.shape[1]
    beta = numpyro.sample(f'{name_prefix}_beta', dist.Normal(jnp.zeros((n_outputs, n_features)), jnp.ones((n_outputs, n_features))))
    intercept = numpyro.sample(f'{name_prefix}_intercept', dist.Normal(jnp.zeros(n_outputs), 1.))
    linear_combination = jnp.einsum('ij,kj->ik', X, beta) + intercept
    return jax.nn.sigmoid(linear_combination)


class SimpleModel:
    def __init__(self, n_customers, n_products):
        self.n_customers = n_customers
        self.n_products = n_products


    def product_choice_model(self, X, P, C, B_obs=None):
        """Model for product choice conditional on category purchase.

        Parameters
        ----------
            X (jnp.ndarray): customer-product observable features, shape=(n_steps, n_customers, n_products)
            P (jnp.ndarray): product prices, shape=(n_steps, n_customers, n_products)
            C (jnp.ndarray): category purchase, shape=(n_steps, n_customers)
            B (jnp.ndarray): prouct purchase, shape=(n_steps, n_customers, n_products)
        """
        n_steps, n_customers, n_products = X.shape
        
        Z_mean = numpyro.sample("Z_mean", dist.HalfNormal(1))
        Z_std = numpyro.sample("Z_std", dist.HalfNormal(1)) 
        with numpyro.plate('products', n_products):                     
            Z = numpyro.sample("Z", dist.TruncatedNormal(Z_mean, Z_std, low=0))

        with numpyro.plate('products', n_products):
            with numpyro.plate('customers', n_customers):
                    utility_beta_ui_x = numpyro.sample('utility_beta_ui_x', dist.Normal(0, 1))
                    utility_beta_ui_w = numpyro.sample('utility_beta_ui_w', dist.Normal(0, 1))
                    utility_beta_ui_z = numpyro.sample('utility_beta_ui_z', dist.Normal(0, 1))

        product_utility = (
            jnp.expand_dims(utility_beta_ui_x, axis=0) * X
            + jnp.expand_dims(utility_beta_ui_w, axis=0) * jnp.log(P)
            + jnp.expand_dims(utility_beta_ui_z, axis=0) * utility_beta_ui_z
            * jnp.expand_dims(Z, axis=[0,1])
        )
       
        product_choice_probs = jnp.exp(product_utility) / jnp.sum(jnp.exp(product_utility), axis=-1, keepdims=True)
        # Masking to filter out Y1 calculations when Y0 is 0
        cat_purchase_mask = (C == 1)

        with mask(mask=cat_purchase_mask):
            numpyro.sample('B', dist.Multinomial(probs=product_choice_probs), obs=B_obs)

# Run main method
if __name__ == "__main__":
    n_steps = 2
    n_customers, n_products = 100, 10
    model = SimpleModel(n_customers, n_products)
    X = jnp.array(np.random.rand(n_steps, n_customers, n_products))
    P = jnp.array(np.random.rand(n_steps, n_customers, n_products))
    C = jnp.array(np.random.randint(0, 2, (n_steps, n_customers)))
    B = jnp.array(np.random.randint(0, 2, (n_steps, n_customers, n_products)))
    nuts_kernel = NUTS(model.product_choice_model)
    mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, X, P, C, B)

    posterior_samples = mcmc.get_samples()