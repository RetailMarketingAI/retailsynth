import logging
from collections import defaultdict
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from tqdm import tqdm

from retailsynth.synthesizer.config import SyntheticParameters


class DataSynthesizer:
    """To generate synthetic data based on the given configuration."""

    def __init__(self, cfg_raw_data: SyntheticParameters):
        """To read coefficients from the config and save as attributes."""
        self.n_customer = cfg_raw_data.n_customer
        self.n_category = cfg_raw_data.n_category
        self.n_product = cfg_raw_data.n_product
        self.category_product_count = cfg_raw_data.category_product_count
        self.store_util_marketing_feature_mode = (
            cfg_raw_data.store_util_marketing_feature_mode
        )
        if self.store_util_marketing_feature_mode not in [
            "discount",
            "random",
            "discount_coupon",
        ]:
            raise ValueError(
                f"Invalid marketing feature mode: {self.store_util_marketing_feature_mode}. Supported modes are 'discount_coupon', 'discount' and 'random'."
            )
        self.random_seed = cfg_raw_data.random_seed
        self.random_seed_range = cfg_raw_data.random_seed_range
        # coef for discount generation
        self.discount_depth_distribution = cfg_raw_data.discount_depth_distribution
        self.transition_prob = cfg_raw_data.transition_prob
        # distribution for coupon generation
        self.coupon_distribution = cfg_raw_data.coupon_distribution
        self.coupon_redemption_distribution = (
            cfg_raw_data.coupon_redemption_distribution
        )
        # coef for product price generation
        self.price_alpha_i0 = cfg_raw_data.price_alpha_i0
        self.price_alpha_1 = cfg_raw_data.price_alpha_1
        self.lowest_price = cfg_raw_data.lowest_price
        # coef for product utility generation
        self.utility_beta_ui_x = cfg_raw_data.utility_beta_ui_x
        self.utility_beta_ui_w = cfg_raw_data.utility_beta_ui_w
        self.utility_clip_percentile = cfg_raw_data.utility_clip_percentile
        self.utility_beta_ui_z = cfg_raw_data.utility_beta_ui_z
        self.utility_error_distribution = cfg_raw_data.utility_error_distribution

        # coef for category utility generation
        self.category_choice_gamma_0j_cate = cfg_raw_data.category_choice_gamma_0j_cate
        self.category_choice_gamma_1j_cate = cfg_raw_data.category_choice_gamma_1j_cate

        # coef for store visit probability
        self.store_visit_gamma_1_store = cfg_raw_data.store_visit_gamma_1_store
        self.store_visit_gamma_2_store = cfg_raw_data.store_visit_gamma_2_store
        self.store_visit_gamma_0_store = cfg_raw_data.store_visit_gamma_0_store
        self.store_visit_theta_u = cfg_raw_data.store_visit_theta_u

        # coef for product demand
        self.purchase_quantity_gamma_0i_prod = (
            cfg_raw_data.purchase_quantity_gamma_0i_prod
        )
        self.purchase_quantity_gamma_1i_prod = (
            cfg_raw_data.purchase_quantity_gamma_1i_prod
        )

        # copy of store visit score in the current step
        # so that we could use this score when computing the store visit probability in the next time step
        self.product_category_mapping = self._initialize_product_category_mapping()
        self.dict_category_product_mapping = (
            self._initialize_dict_category_product_mapping()
        )

        # store choice probabilities
        self.choice_decision_stats: Dict = defaultdict(list)
        # store elasticities
        self.elasticity_stats: Dict = defaultdict(list)
        # cache to compute store visit probability depending on the previous step
        self.category_utility_cache = jnp.zeros((self.n_customer, self.n_category))

        # set up initial store prob to start a trajectory
        self._initialize_random_seed_generator()
        self.initial_store_visit_prob = jnp.ones((self.n_customer,))
        self.initial_store_visit = jnp.ones((self.n_customer,))
        self.discount_state = jnp.zeros((self.n_product,))
        self.product_endogenous_feature = self._initialize_product_endogenous_feature()
        self.product_price = self._initialize_product_price(
            self.product_endogenous_feature
        )

    def _initialize_random_seed_generator(self):
        """Fix a random seed generator internally to sample the trajectory."""
        self.random_seed_generator = np.random.default_rng(self.random_seed)
        self.random_seed = random.PRNGKey(self.random_seed)

    def _get_random_seed(self) -> jnp.ndarray:
        """Pop a random integer from the random seed generator and encapsulate it to a jax random key.

        Returns
        -------
            jnp.ndarray: The jax random key
        """
        seed = self.random_seed_generator.integers(low=0, high=self.random_seed_range)
        seed = random.PRNGKey(seed)
        return seed

    def _initialize_product_category_mapping(self) -> jnp.ndarray:
        """Initialize a indicator matrix to specify which category a product belongs to.

        The mapping is stored as a matrix, which will be used in multiplication with categorical transaction matrix.

        Returns
        -------
            jnp.ndarray: a 2d matrix in shape of (n_product, n_category)
        """
        mapping = np.zeros((self.n_product, self.n_category), dtype=np.int)
        idx = np.repeat(np.arange(self.n_category), self.category_product_count)
        mapping[np.arange(self.n_product), idx] = 1
        return jnp.array(mapping)

    def _initialize_dict_category_product_mapping(self) -> Dict[int, jnp.ndarray]:
        """Initialize a dictionary of category product mapping.

        The mapping is stored as a dictionary with category_nbr as the key and a list of corresponding product_nbrs as the value.
        It will be used to easily extract the product index for one category to generate product choice decision within a category

        Returns
        -------
            Dict[int, jnp.ndarray]: with category_nbr as the key and an array of product indices in the corresponding category

        """
        product_index, category_index = jnp.where(self.product_category_mapping == 1)
        return {
            category_nbr: product_index[jnp.where(category_index == category_nbr)]
            for category_nbr in range(self.n_category)
        }

    def _initialize_product_endogenous_feature(self) -> jnp.ndarray:
        """Initialize the endogenous feature of every product, which maintain static along the trajectory.

        Returns
        -------
            jnp.ndarray: product endogenous feature, in shape of (n_product, )
        """
        variable_mean = numpyro.sample(
            "endogenous_mean",
            dist.HalfNormal(1),
            sample_shape=(self.n_product,),
            rng_key=self.random_seed,
        )
        variable_std = numpyro.sample(
            "endogenous_std",
            dist.HalfNormal(1),
            sample_shape=(self.n_product,),
            rng_key=self.random_seed,
        )
        w_dist = dist.TruncatedNormal(variable_mean, variable_std, low=0)
        w = self._sample_feature(w_dist, "w")
        return w

    def _initialize_product_price(self, endogenous_feature: jnp.array) -> jnp.array:
        """Initialize the base price of every product, which maintain static along the trajectory.

        $$P_{it} = \alpha_{i0} + \alpha_1 Z_{i}$$

        Parameters
        ----------
            endogenous_feature (jnp.array): shared feature used in price generation and product utility computation

        Returns
        -------
            jnp.array: product price
        """
        assert endogenous_feature.shape == self.price_alpha_i0.shape
        base_price = self.price_alpha_i0 + self.price_alpha_1 * endogenous_feature
        return base_price.clip(self.lowest_price)

    def _initialize_feature_distribution_standard_normal(
        self,
        shape: tuple,
        variable_name: Optional[str] = "z",
        force_positive: Optional[bool] = False,
    ) -> dist.Distribution:
        """Create a standard normal distribution to sample features from.

        Parameters
        ----------
            shape (tuple): shape of the feature to generate
            variable_name (str, optional): variable name. Defaults to "z".
            force_positive (bool, optional): whether to force the feature to be positive. Defaults to False.

        Returns
        -------
            dist.Distribution: a distribution to sample features from
        """
        seed = self._get_random_seed()
        variable_mean = numpyro.sample(
            f"{variable_name}_mean",
            dist.Normal(0, 1),
            sample_shape=shape,
            rng_key=seed,
        )
        variable_std = numpyro.sample(
            f"{variable_name}_std",
            dist.HalfNormal(1),
            sample_shape=shape,
            rng_key=seed,
        )
        variable_dist = (
            dist.TruncatedNormal(variable_mean, variable_std, low=0)
            if force_positive
            else dist.Normal(variable_mean, variable_std)
        )
        return variable_dist

    def _sample_feature(
        self,
        variable_distribution: dist.Distribution,
        variable_name: Optional[str] = "z",
    ) -> jnp.ndarray:
        """Sample a random feature.

        Parameters
        ----------
            variable_distribution (dist.Distribution): feature distribution
            variable_name (str, optional): variable name. Defaults to "z".

        Returns
        -------
            jnp.ndarray: random features
        """
        seed = self._get_random_seed()
        return numpyro.sample(variable_name, variable_distribution, rng_key=seed)

    def _sample_error(
        self,
        shape: tuple,
        distribution: dist.Distribution,
    ) -> jnp.ndarray:
        """Sample error term.

        Parameters
        ----------
            shape (tuple): shape of the error to generate
            distribution (dist.Distribution): error distribution

        Returns
        -------
            jnp.ndarray: random error generated from normal distribution (0, 0.1)
        """
        seed = self._get_random_seed()
        return numpyro.sample("error", distribution, sample_shape=shape, rng_key=seed)

    def sample_discount(self, variable_name: Optional[str] = "d") -> jnp.ndarray:
        """Sample random discount.

        Parameters
        ----------
            variable_name (str, optional): name of the discount term. Defaults to "d".

        Returns
        -------
            jnp.ndarray: random discount array in shape of (n_customer, n_product)
        """
        seed = self._get_random_seed()
        shape = (self.n_product,)
        # get discount
        discount = numpyro.sample(
            variable_name,
            self.discount_depth_distribution,
            sample_shape=shape,
            rng_key=seed,
        )
        # get the flag whether to apply disocunts
        # in shape of (n_product, 2), where the first column is the next status if the current status is 0
        # and the second column is the next status if the current status is 1
        transition_to_discount = numpyro.sample(
            "transition",
            self.transition_prob,
            rng_key=seed,
        )
        self.discount_state = self._update_discount_state(transition_to_discount)
        return discount * self.discount_state

    def _update_discount_state(self, transition_sample):
        """Update the discount flag based on the transition sample.

        Parameters
        ----------
            transition_sample (jnp.ndarray): a sample from the transition probability
        """
        new_discount_state = jnp.zeros((self.n_product,))
        status_0_index = jnp.where(self.discount_state == 0)
        new_discount_state = new_discount_state.at[status_0_index].set(
            transition_sample[status_0_index, 0].flatten()
        )
        status_1_index = jnp.where(self.discount_state == 1)
        new_discount_state = new_discount_state.at[status_1_index].set(
            transition_sample[status_1_index, 1].flatten()
        )
        return new_discount_state

    def compute_product_price(self, discount: jnp.ndarray) -> jnp.ndarray:
        """Compute product price.

        $$
        P_{it} &= (1 - D_{it}) P_{it}
        $$

        Parameters
        ----------
            discount (jnp.ndarray): discount array in shape of (n_customer, n_product)
            coupon (jnp.ndarray): coupon that the customer redeemed at the current time step, in shape of (n_customer, n_product). Default to None.

        Returns
        -------
            jnp.ndarray: price array in shape of (n_product, )
            jnp.ndarray: shared feature used in price generation and product utility generation
        """
        # compute the discounted price
        assert self.product_price.shape == discount.shape
        unit_price = (1 - discount) * self.product_price

        # clip the price to be at least the lowest price
        negative_price = (unit_price <= 0).sum()
        logging.debug(
            f"Find {negative_price} negative prices (out of {self.n_product}) and clip them to 0.01."
        )
        return unit_price.clip(self.lowest_price)

    def sample_coupon(self, mode: str = "universal"):
        """Sample coupon for all customers.

        Parameters
        ----------
            mode (str, optional): coupon modes. Defaults to "universal".
                "universal" mode for coupons applied to all categories.
                "category" mode for coupons applied to specific category.

        Returns
        -------
            jnp.ndarray: coupon array in shape of (n_customer, n_product)
        """
        if mode == "universal":
            shape = (self.n_customer,)
        elif mode == "category":
            shape = (self.n_customer, self.n_category)  # type: ignore
        else:
            raise ValueError(
                f"Invalid coupon mode: {mode}. Supported modes are 'universal' and 'category'."
            )
        seed = self._get_random_seed()
        # get actual coupon
        coupon = numpyro.sample(
            "coupon",
            self.coupon_distribution,
            sample_shape=shape,
            rng_key=seed,
        )
        return coupon

    def compute_product_price_with_coupon(
        self, product_price: jnp.ndarray, coupon: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Compute the product price with coupon redemption.

        $$
        P_{it} &= (1 _ C_{uit} * Redemption_{ut}) P_{it}
        $$

        Parameters
        ----------
            product_price (jnp.ndarray): product price at the current step, in shape of (n_product, )
            coupon (jnp.ndarray, optional): coupon a customer redeemed at the current step, in shape of (n_customer, n_category) or (n_customer, ). Defaults to be None.

        Returns
        -------
            jnp.ndarray: product price with coupon redemption
        """
        seed = self._get_random_seed()

        if coupon is None:
            # broadcast to be in shape of (n_customer, n_product)
            # to make the price computation easier
            redeemed_coupon = jnp.zeros((self.n_customer, self.n_product))
        elif coupon.shape == (self.n_customer, self.n_category):
            redemption = numpyro.sample(
                "redemption",
                self.coupon_redemption_distribution,
                sample_shape=(self.n_category,),
                rng_key=seed,
            ).T  # get redemption indicator matrix in shape of (n_customer, n_category)
            redeemed_coupon = jnp.matmul(
                coupon * redemption, self.product_category_mapping.T
            )
        elif coupon.shape == (self.n_customer,):
            redemption = numpyro.sample(
                "redemption",
                self.coupon_redemption_distribution,
                rng_key=seed,
            )  # get redemption indicator matrix in shape of (n_customer, )
            redeemed_coupon = jnp.repeat(
                jnp.expand_dims(coupon * redemption, axis=1), self.n_product, axis=1
            )
        else:
            raise ValueError(
                f"Invalid coupon shape: {coupon.shape}. It should be either (n_customer, n_category) or (n_customer, )"
            )

        assert redeemed_coupon.shape == (self.n_customer, self.n_product)
        assert (redeemed_coupon >= 0).all(), "Coupon should be non-negative."
        assert (
            redeemed_coupon < 1
        ).all(), "Coupon should be in percentage, and less than one."

        price_with_coupon = (1 - redeemed_coupon) * product_price
        return price_with_coupon

    def compute_product_utility(self, price: jnp.ndarray) -> jnp.ndarray:
        """Compute product utility.

        $$
        \mu^{prod}_{uit} &= \mathbf{\beta_{ui}^x} \mathbf{X_{uit}} + \beta_{ui}^{z} Z_{i} + \beta_{ui}^w log(P_{it}) + \epsilon_{uit}
        $$

        Parameters
        ----------
            price (jnp.ndarray): product price

        Returns
        -------
            jnp.ndarray: product utility in shape of (n_customer, n_product)
        """
        x_dist = self._initialize_feature_distribution_standard_normal(
            (self.n_customer, self.n_product), "x"
        )
        x = self._sample_feature(x_dist)
        self.choice_decision_stats["x"].append(x)

        error = self._sample_error(
            (self.n_customer, self.n_product), self.utility_error_distribution
        )
        assert x.shape == error.shape
        assert self.product_endogenous_feature.shape[-1] == x.shape[-1]

        product_utility = (
            self.utility_beta_ui_x * x
            + self.utility_beta_ui_w * jnp.log(price)
            + self.utility_beta_ui_z
            * jnp.expand_dims(self.product_endogenous_feature, axis=0)
            + error
        )
        # An extreme large product utility can leads to unreasonable large demand in the following step
        # To avoid this, we clip the product utility with an upper qua
        upper_bound = jnp.percentile(product_utility, self.utility_clip_percentile)
        return jnp.clip(product_utility, a_max=upper_bound)

    def compute_product_purchase_conditional_probability(
        self, product_utility: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the product purchase probability conditional on category choice.

        $$
        p^{prod}_{uit} &= P(\mathbb{I}_{uit} = 1 | \mathbb{I}_{ujt} = 1)\\
        &= \frac{\exp{(\mu^{prod}_{uit})}}{\sum_{k \in J_c} \exp{(\mu^{prod}_{ukt}})}\\
        $$

        Parameters
        ----------
            product_utility (jnp.ndarray): product utility in shape of (n_customer, n_product)

        Returns
        -------
            jnp.ndarray: conditional purchase probability
        """
        exp_utility = jnp.exp(product_utility)
        category_utility = jnp.matmul(
            # (customer, product) VS (product, category) -> (customer, category)
            jnp.matmul(exp_utility, self.product_category_mapping),
            # (category, product)
            self.product_category_mapping.T,
        )
        product_prob = exp_utility / category_utility
        logging.debug(
            f"Global mean of product purchase probability: {product_prob.mean(): .4f}"
        )
        return product_prob

    def compute_category_utility(self, product_utility: jnp.ndarray) -> jnp.ndarray:
        """Compute category utility based on product utility.

        $$
        CV_{ujt} &= \log \sum_{k \in J_j} \exp(\mu^{prod}_{ukt})
        $$

        Parameters
        ----------
            product_utility (jnp.ndarray): product utility in shape of (n_customer, n_product)

        Returns
        -------
            jnp.ndarray:
        """
        category_utility = jnp.log(
            jnp.matmul(
                jnp.exp(product_utility),
                self.product_category_mapping,
            )
        )
        return category_utility

    def compute_category_purchase_conditional_probability(
        self, category_utility: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute category purchase probability conditional on store visit.

        $$
        p^{cate}_{ujt} &= P(\mathbb{I}_{ujt}=1 | \mathbb{I}_{ut}=1) \\
        &= \frac{\exp(\gamma_{0j}^{cate} + \gamma_{1j}^{cate} CV_{ujt})}{1 + \exp(\gamma_{0j}^{cate} + \gamma_{1j}^{cate} CV_{ujt})}
        $$

        Parameters
        ----------
            category_utility (jnp.ndarray): category utility in shape of (n_customer, n_category)

        Returns
        -------
            jnp.ndarray: conditional purchase probability
        """
        assert (
            self.category_choice_gamma_0j_cate.shape[0]
            == self.category_choice_gamma_1j_cate.shape[0]
            == category_utility.shape[1]
        )
        score = jnp.expand_dims(
            self.category_choice_gamma_1j_cate, axis=0
        ) * category_utility + jnp.expand_dims(
            self.category_choice_gamma_0j_cate, axis=0
        )
        conditional_prob = jax.nn.sigmoid(score)
        logging.debug(
            f"Global mean of category purchase probability: {conditional_prob.mean(): .4f}"
        )
        return conditional_prob

    def sample_marketing_feature(self, **kwargs) -> jnp.ndarray:
        """Get marketing feature to be used in the store utility computation.

        Parameters
        ----------
            **kwargs: keyword arguments to pass to the function.
                If self.store_util_marketing_feature_mode == "discount", the discount is required. marketing feature is the sum of the discount.
                If self.store_util_marketing_feature_mode == "discount_coupon", the discount and coupon are required. marketing feature is the sum of all product discounts and coupons.

        Raises
        ------
            NotImplementedError: raise if the marketing feature mode is not supported

        Returns
        -------
            jnp.ndarray: marketing features
        """
        if self.store_util_marketing_feature_mode == "random":
            marketing_feature_dist = (
                self._initialize_feature_distribution_standard_normal(
                    (self.n_customer,), "marketing_feature"
                )
            )
            marketing_feature = self._sample_feature(marketing_feature_dist)
        elif self.store_util_marketing_feature_mode == "discount":
            product_discount = kwargs.get("discount")
            # this marketing feature is a scaler as the same discount applies to all customers
            marketing_feature = jnp.sum(product_discount)
        elif self.store_util_marketing_feature_mode == "discount_coupon":
            product_discount = kwargs.get("discount")
            marketing_feature = jnp.sum(product_discount)

            product_coupon = kwargs.get("coupon")
            if product_coupon is None:
                raise ValueError(
                    "Coupon is required to generate marketing feature in 'discount_coupon' mode."
                )
            if product_coupon.shape == (self.n_customer,):
                # empirical coupon is the store-wide coupon multiplied by the number of products
                product_coupon_marketing_term = product_coupon * self.n_product
            elif product_coupon.shape == (self.n_customer, self.n_category):
                product_coupon_marketing_term = jnp.matmul(
                    product_coupon, self.product_category_mapping.T
                )
                # empirical coupon is to sum (category coupon * the number of products in the category)
                product_coupon_marketing_term = jnp.sum(
                    product_coupon_marketing_term, axis=1
                )
            elif product_coupon.shape == (self.n_customer, self.n_product):
                product_coupon_marketing_term = jnp.sum(product_coupon, axis=1)

            # marketing feature in shape of (n_customer, )
            # because the coupon is customer-specific
            marketing_feature = (
                jnp.sum(product_discount) + product_coupon_marketing_term
            )

        return marketing_feature

    def compute_store_utility(
        self,
        prev_category_utility: jnp.ndarray,
        prev_store_visit: jnp.ndarray,
        current_marketing_feature: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute store utility for all customers, to derive the store visit probability.

        $$
        SV_{ut} &= \log \sum_{j \in J} \exp(CV_{ujt})
        $$

        Parameters
        ----------
            prev_category_utility (jnp.ndarray): category utility at the previous time step
            prev_store_visit (jnp.ndarray): store visit decision at the previous time step
            current_marketing_feature (jnp.ndarray): marketing feature at the current time step

        Returns
        -------
            jnp.ndarray: store utility at the current step
        """
        logsum_store_utility = jnp.log(jnp.exp(prev_category_utility).sum(axis=-1))
        score = (
            self.store_visit_gamma_1_store * prev_store_visit * logsum_store_utility
            + self.store_visit_gamma_2_store * current_marketing_feature
            + self.store_visit_gamma_0_store
        )
        return score

    def compute_store_visit_probability(
        self,
        prev_category_utility: jnp.ndarray,
        prev_store_visit_prob: jnp.ndarray,
        prev_store_visit: jnp.ndarray,
        marketing_feature: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute store visit probability.

        \begin{align}
            P(\mathbb{I}_{ut}) &=
            \begin{cases}
                1 & \text{if } t = 1, \\
                (1-\theta_{u}) s_{ut} + P(\mathbb{I}_{u(t-1)}) \theta_{u} & \text{if } t > 1,
            \end{cases} \label{eq:main_equation} \\
            \intertext{where:}
            \mu^{store}_{ut} &= \gamma_0^{store} + \mathbb{I}_{u(t-1)} \gamma_1^{store} SV_{u(t-1)} + \boldsymbol{\gamma_2^{store}} \mathbf{X_{ut}^{store}} \label{eq:mu_store}\\
            s_{ut} &= \frac{exp(\mu^{store}_{ut})}{ 1 + exp(\mu^{store}_{ut})} \label{eq:s_ut}
        \end{align}

        Parameters
        ----------
            prev_category_utility (jnp.ndarray): category utility at the previous time step
            prev_store_visit_prob (jnp.ndarray): store visit probability at the previous time step
            prev_store_visit (jnp.ndarray): store visit decision at the previous time step
            marketing_feature (jnp.ndarray): marketing feature at the current time step

        Returns
        -------
            jnp.ndarray: store visit probability at the current time step
        """
        store_utility = self.compute_store_utility(
            prev_category_utility, prev_store_visit, marketing_feature
        )
        current_store_prob = jax.nn.sigmoid(store_utility)
        store_visit_prob = (
            1 - self.store_visit_theta_u
        ) * current_store_prob + self.store_visit_theta_u * prev_store_visit_prob
        logging.debug(
            f"Global mean of store visit probability: {store_visit_prob.mean(): .4f}"
        )
        return store_visit_prob

    def _sample_product_choice(self, product_prob, random_seed) -> jnp.ndarray:
        """Sample product choice decision for all categories.

        Parameters
        ----------
            product_prob (jnp.ndarray): product purchase probability conditional on category choice
            random_seed (int): random seed to get a sample from the product choice distribution

        Returns
        -------
            jnp.ndarray: product choice decision in shape of (n_customer, n_product)

        """
        complete_product_choice = jnp.zeros(product_prob.shape)
        selected_customer_product_index = (
            []
        )  # expecting (customer, product) pair to be added
        for _, product_index in self.dict_category_product_mapping.items():
            product_index_choice = dist.Categorical(
                product_prob[:, product_index]
            ).sample(random_seed)
            product_choice = product_index[product_index_choice]
            selected_customer_product_index.append(
                jnp.vstack([jnp.arange(self.n_customer), product_choice])
            )
        # selected index array in shape of (n_trx, 2), where the second dimension records customer_key and product_nbr
        selected_customer_product_index = jnp.hstack(selected_customer_product_index)
        complete_product_choice = complete_product_choice.at[
            selected_customer_product_index[0], selected_customer_product_index[1]
        ].set(1)
        return complete_product_choice

    def compute_product_demand_mean(self, product_utility: jnp.ndarray) -> jnp.ndarray:
        """Compute the mean of product demand.

        Parameters
        ----------
            product_utility (jnp.ndarray): product utility in shape of (n_customer, n_product)

        $$
        \lambda_{uit} &= \exp(\gamma^{prod}_{0i} + \gamma^{prod}_{ui}\ \mu^{prod}_{uit})
        $$

        Returns
        -------
            jnp.ndarray: mean of product demand in shape of (n_customer, n_product)
        """
        demand_mean = jnp.exp(
            self.purchase_quantity_gamma_1i_prod * product_utility
            + jnp.expand_dims(self.purchase_quantity_gamma_0i_prod, axis=0)
        )
        return demand_mean

    def _sample_decisions(
        self,
        store_visit_prob: jnp.ndarray,
        category_choice_prob: jnp.ndarray,
        product_choice_prob: jnp.ndarray,
        product_demand_prob: jnp.ndarray,
    ) -> tuple:
        """Sample store visit, category choice, product choice, and product demand.

        Parameters
        ----------
            store_visit_prob (jnp.ndarray): store visit probability
            category_choice_prob (jnp.ndarray): category purchase probability conditional on store visit
            product_choice_prob (jnp.ndarray): product purchase probability conditional on category choice
            product_demand_prob (jnp.ndarray): product demand mean conditional on product choice

        Returns
        -------
            tuple: tuple of jnp.ndarray for decision of each step
        """
        seed = self._get_random_seed()

        store_visit = dist.Bernoulli(store_visit_prob).sample(seed)  # (n_customer, )
        category_choice = dist.Bernoulli(category_choice_prob).sample(
            seed
        )  # (n_customer, n_category)
        product_choice = self._sample_product_choice(
            product_choice_prob, seed
        )  # (n_customer, n_product)
        product_demand = (
            dist.Poisson(product_demand_prob).sample(seed) + 1
        )  # (n_customer, n_product)
        # A failed case for the following assertion statement is that we have product_demand_prob super big,
        # that we generate a "infinity" large float 64.
        # When we shifted the generated number, "infinity" + 1 would give us negative "infinity"
        assert (
            product_demand > 0
        ).all(), "The generated demand is so large that it causes an integer overflow."
        return store_visit, category_choice, product_choice, product_demand

    def compute_joint_decision(
        self,
        store_visit: jnp.ndarray,
        category_choice: jnp.ndarray,
        product_choice: jnp.ndarray,
        product_demand: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute joint decision for product purchase.

        \begin{align}
            P(\mathbb{Q}_{ujt} = q) =  P(\mathbb{I}_{ut}=1) * P(\mathbb{I}_{uct} = 1 | \mathbb{I}_{ut} = 1) * P(\mathbb{I}_{ujt} = 1 | \mathbb{I}_{uct} = 1) * P(\mathbb{Q}_{ujt} = q|\mathbb{I}_{ujt} = 1)
        \end{align}

        Parameters
        ----------
            store_visit (jnp.ndarray): store visit decision
            category_choice (jnp.ndarray): category choice decision conditional on store visit
            product_choice (jnp.ndarray): product choice decision conditional on category choice
            product_demand (jnp.ndarray): product demand decision conditional on product choice

        Returns
        -------
            jnp.ndarray: joint decision on product quantity purchased

        """
        store_visit = jnp.expand_dims(store_visit, axis=-1)
        category_choice = jnp.matmul(category_choice, self.product_category_mapping.T)
        joint_decision = product_demand * category_choice * product_choice * store_visit

        return joint_decision

    def sample_transaction_one_step(
        self,
        prev_store_visit: jnp.ndarray,
        prev_store_visit_prob: jnp.ndarray,
        coupon: jnp.ndarray = None,
        compute_store_prob: bool = False,
    ) -> tuple:
        """Sample transaction for one time step.

        Parameters
        ----------
            prev_store_visit (jnp.ndarray): store visit decision at the previous time step
            prev_store_visit_prob (jnp.ndarray): store visit probability at the previous time step
            coupon (jnp.ndarray): coupon that the customer redeemed at the current time step. Default to None.
            compute_store_prob (bool): flag to tell whether to compute the store visit probability recursively based on the previous store visit prob

        Returns
        -------
            tuple: tuple of jnp.ndarray
        """
        discount = self.sample_discount()
        product_price = self.compute_product_price(discount)
        product_price_with_coupon = self.compute_product_price_with_coupon(
            product_price, coupon
        )
        product_utility = self.compute_product_utility(product_price_with_coupon)

        category_utility = self.compute_category_utility(product_utility)
        logging.debug(f"Avg. category utility: {category_utility.mean():.4f}")
        product_choice_prob = self.compute_product_purchase_conditional_probability(
            product_utility
        )
        logging.debug(
            f"Avg. product choice probability: {product_choice_prob.mean():.4f}"
        )

        category_choice_prob = self.compute_category_purchase_conditional_probability(
            category_utility
        )
        logging.debug(
            f"Avg. category choice probability: {category_choice_prob.mean():.4f}"
        )
        if compute_store_prob:
            marketing_feature = self.sample_marketing_feature(
                discount=discount, coupon=coupon
            )
            assert prev_store_visit_prob is not None
            assert prev_store_visit is not None
            store_visit_prob = self.compute_store_visit_probability(
                self.category_utility_cache,
                prev_store_visit_prob,
                prev_store_visit,
                marketing_feature,
            )
        else:
            store_visit_prob = prev_store_visit_prob

        product_demand_mean = self.compute_product_demand_mean(product_utility)

        self.category_utility_cache = category_utility
        # store probability stats
        self.choice_decision_stats["product_price"].append(product_price)
        self.choice_decision_stats["product_price_with_coupon"].append(
            product_price_with_coupon
        )
        self.choice_decision_stats["discount"].append(discount)
        self.choice_decision_stats["discount_state"].append(self.discount_state)
        self.choice_decision_stats["store_visit_prob"].append(store_visit_prob)
        self.choice_decision_stats["category_choice_prob"].append(category_choice_prob)
        self.choice_decision_stats["product_choice_prob"].append(product_choice_prob)
        self.choice_decision_stats["demand_mean"].append(product_demand_mean)
        self.choice_decision_stats["product_utility"].append(product_utility)
        self.choice_decision_stats["expected_demand"].append(
            self.compute_joint_decision(
                store_visit_prob,
                category_choice_prob,
                product_choice_prob,
                product_demand_mean,
            )
        )
        # store elasticity stats
        redeemed_coupon = 1 - product_price_with_coupon / self.product_price
        store_visit_elasticity = self.compute_store_visit_elasticity(
            store_visit_prob, discount=discount, coupon=redeemed_coupon
        )
        category_elasticity = self.compute_category_elasticity(
            product_choice_prob, category_choice_prob
        )
        product_elasticity = self.compute_product_elasticity(product_choice_prob)
        product_demand_elasticity = self.compute_product_demand_elasticity(
            product_demand_mean
        )
        overall_elasticity = self.compute_overall_elasticity(
            store_visit_elasticity,
            category_elasticity,
            product_elasticity,
            product_demand_elasticity,
        )
        self.elasticity_stats["store_visit_elasticity"].append(store_visit_elasticity)
        self.elasticity_stats["category_choice_elasticity"].append(category_elasticity)
        self.elasticity_stats["product_choice_elasticity"].append(product_elasticity)
        self.elasticity_stats["product_demand_elasticity"].append(
            product_demand_elasticity
        )
        self.elasticity_stats["overall_elasticity"].append(overall_elasticity)

        (
            store_visit,
            category_choice,
            product_choice,
            product_demand,
        ) = self._sample_decisions(
            store_visit_prob,
            category_choice_prob,
            product_choice_prob,
            product_demand_mean,
        )
        joint_decision = self.compute_joint_decision(
            store_visit, category_choice, product_choice, product_demand
        ).astype(np.int)

        return (
            joint_decision,
            store_visit,
            store_visit_prob,
            product_price,
            discount,
            product_price_with_coupon,
        )

    def sample_trajectory(
        self,
        n_week: int,
    ) -> jnp.ndarray:
        """Sample a trajectory.

        Parameters
        ----------
            n_week (int): length of the trajectory

        Returns
        -------
            tuple: trajectory, price array
        """
        store_visit_prob = self.initial_store_visit_prob
        store_visit = self.initial_store_visit
        trajectory = []
        product_price_record = []
        discount_record = []
        price_with_coupon_record = []
        for week_index in tqdm(range(n_week), desc="Synthesizing trajectory"):
            compute_store_prob = week_index > 0
            coupon = self.sample_coupon()
            (
                trx,
                store_visit,
                store_visit_prob,
                product_price,
                discount,
                product_price_with_coupon,
            ) = self.sample_transaction_one_step(
                store_visit,
                store_visit_prob,
                coupon=coupon,
                compute_store_prob=compute_store_prob,
            )
            trajectory.append(trx)
            product_price_record.append(product_price)
            discount_record.append(discount)
            price_with_coupon_record.append(product_price_with_coupon)
        trajectory = jnp.array(trajectory)
        product_price_record = jnp.array(product_price_record)
        discount_record = jnp.array(discount_record)
        price_with_coupon_record = jnp.array(price_with_coupon_record)
        return (
            trajectory,
            product_price_record,
            discount_record,
            price_with_coupon_record,
        )

    def convert_product_utility_to_df(
        self,
        product_utility: jnp.ndarray,
    ) -> pd.DataFrame:
        """Convert trajectory to dataframe."""
        time, customer, product = jnp.where(product_utility)
        df = pd.DataFrame(
            columns=[
                "week",
                "customer_key",
                "product_nbr",
                "product_utility",
            ]
        )
        df["week"] = time
        df["customer_key"] = customer
        df["product_nbr"] = product
        df["product_utility"] = product_utility[time, customer, product]
        return df

    def convert_trajectory_to_df(
        self,
        trajectory: jnp.ndarray,
        price_record: jnp.ndarray,
        discount_record: jnp.array,
        price_with_coupon_record: jnp.array,
    ) -> pd.DataFrame:
        """Convert trajectory to dataframe."""
        time, customer, product = jnp.where(trajectory > 0)
        df = pd.DataFrame(
            columns=[
                "week",
                "customer_key",
                "product_nbr",
                "item_qty",
                "unit_price",
                "sales_amt",
            ]
        )
        START_WEEK = 1
        df["week"] = time + START_WEEK
        df["customer_key"] = customer
        df["product_nbr"] = product
        df["item_qty"] = trajectory[time, customer, product]
        df["discount_portion"] = discount_record[time, product]
        df["unit_price"] = price_record[time, product]
        df["unit_price_with_coupon"] = price_with_coupon_record[time, customer, product]
        df["sales_amt"] = df["unit_price"] * df["item_qty"]
        return df.set_index(["week", "customer_key", "product_nbr"])

    def convert_customer_info_to_df(self) -> pd.DataFrame:
        """Create placeholder for customer info data frame, containing only customer_key."""
        customer_info = pd.DataFrame(columns=["customer_key"])
        customer_info["customer_key"] = jnp.arange(self.n_customer)
        return customer_info.set_index("customer_key")

    def convert_product_info_to_df(self):
        """Create placeholder for product info data frame, containing only product_nbr and category_desc."""
        n_product, _ = self.product_category_mapping.shape
        product_info = pd.DataFrame(columns=["product_nbr", "category_nbr"])
        product_info["product_nbr"] = jnp.arange(n_product)
        product_info["category_nbr"] = jnp.argmax(
            self.product_category_mapping, axis=-1
        )
        product_info["all"] = "all"
        return product_info.set_index("product_nbr")

    def compute_store_visit_elasticity(
        self, store_visit_prob: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Compute the store visit elasticity of a product regarding to the product price.

        $$
            e^{store}_{uit} &= \left. \frac{d p^{store}_{ut}} {p^{store}_{ut}} \middle/ \frac{d P_{uit}}{P_{uit}} \right.\\
            &= -\gamma_2(1-p^{store}_{ut})(1-D_{it})
        $$

        Parameters
        ----------
            store_visit_prob (jnp.ndarray): store visit probability at the current time step
            **kwargs: keyword arguments to pass to the function. Need to include "discount" if self.store_util_marketing_feature_mode == "discount"

        Returns
        -------
            jnp.ndarray: store elasticity in shape of (n_customer, )
        """
        if self.store_util_marketing_feature_mode == "discount":
            discount = kwargs.get("discount")
            store_elasticity = (
                -self.store_visit_gamma_2_store
                * (1 - jnp.expand_dims(store_visit_prob, axis=-1))
                * (1 - jnp.expand_dims(discount, axis=0))
            )
            logging.debug(
                f"Global mean of store elasticity: {store_elasticity.mean(): .4f}"
            )
            return store_elasticity
        elif self.store_util_marketing_feature_mode == "discount_coupon":
            discount = kwargs.get("discount")
            coupon = kwargs.get("coupon")
            if coupon is None:
                raise ValueError(
                    "Coupon is required to generate marketing feature in 'discount_coupon' mode."
                )
            assert coupon.shape == (
                self.n_customer,
                self.n_product,
            ), f"Coupon shape needs to be in ({self.n_customer}, {self.n_product})."
            store_elasticity = (
                -self.store_visit_gamma_2_store
                * (1 - jnp.expand_dims(store_visit_prob, axis=-1))
                * (1 - jnp.expand_dims(discount, axis=0))
                * (1 - coupon)
            )
            logging.debug(
                f"Global mean of store elasticity: {store_elasticity.mean(): .4f}"
            )
            return store_elasticity

        elif self.store_util_marketing_feature_mode == "random":
            # the store visit probability is not a function of price, thus the elasticity is zero
            return jnp.zeros(shape=(self.n_customer, self.n_product))

    def compute_category_elasticity(
        self, product_prob: jnp.ndarray, category_prob: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the category elasticity of a product.

        $$
            e^{cate}_{uit} &= \left. \frac{d p^{cate}_{ujt}} {p^{cate}_{ujt}} \middle/ \frac{d P_{uit}}{P_{uit}} \right.\\
            &= p^{prod}_{uit} (1 - p^{cate}_{ujt})
            \gamma^{cate}_{1j}\beta_{ui}^{w}
        $$

        Parameters
        ----------
            product_prob (jnp.ndarray): product choice probability
            category_prob (jnp.ndarray): category choice probability

        Returns
        -------
            jnp.ndarray: category elasticity in shape of (n_customer, n_category)
        """
        category_prob_reshaped = jnp.matmul(
            category_prob, self.product_category_mapping.T
        )
        category_scaler_reshaped = jnp.matmul(
            self.category_choice_gamma_1j_cate, self.product_category_mapping.T
        )
        category_elasticity = (
            self.utility_beta_ui_w
            * category_scaler_reshaped
            * (1 - category_prob_reshaped)
            * product_prob
        )
        logging.debug(
            f"Global mean of category elasticity: {category_elasticity.mean(): .4f}"
        )
        return category_elasticity

    def compute_product_elasticity(self, product_prob: jnp.ndarray) -> jnp.ndarray:
        """Compute the product elasticity for the step of product choice.

        $$
            e^{prod}_{u(ii)t} &= \left. \frac{d p^{prod}_{uit}} {p^{prod}_{uit}} \middle/ \frac{d P_{uit}}{P_{uit}} \right.\\
            &= (1 - p^{prod}_{uit}) \beta_{ui}^{w}
        $$

        Parameters
        ----------
            product_prob (jnp.ndarray): product choice probability

        Returns
        -------
            jnp.ndarray: product elasticity in shape of (n_customer, n_product)
        """
        product_elasticity = (1 - product_prob) * self.utility_beta_ui_w
        logging.debug(
            f"Global mean of product elasticity: {product_elasticity.mean(): .4f}"
        )
        return product_elasticity

    def compute_product_demand_elasticity(
        self, product_demand: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the product elasticity for the step of purchased quantity.
        
        $$
            e^{quant}_{u(ii)t} &= \left. \frac{d (\lambda_{uit} + 1)} {\lambda_{uit} + 1} \middle/ \frac{d P_{uit}}{P_{uit}} \right.\\
            &= (\frac{\lambda_{uit}}{1 + \lambda_{uit}})\gamma_{ui}^{prod}\beta_{ui}^{w}
        $$

        Parameters
        ----------
            product_demand (jnp.ndarray): mean of product demand

        Returns
        -------
            jnp.ndarray: demand elasticity in shape of (n_customer, n_product)
        """
        demand_elasticity = (
            (1 - 1 / (product_demand + 1))
            * self.purchase_quantity_gamma_1i_prod
            * self.utility_beta_ui_w
        )
        logging.debug(
            f"Global mean of product demand elasticity: {demand_elasticity.mean(): .4f}"
        )
        return demand_elasticity

    def compute_overall_elasticity(
        self,
        store_visit_elasticity: jnp.ndarray,
        category_elasticity: jnp.ndarray,
        product_elasticity: jnp.ndarray,
        product_demand_elasticity: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the overall elasticity of the joint decision.

        $$
            e_{uit}^{overall} &= \left. \frac{d \mathbb{E}Q_{uit}}{\mathbb{E}Q_{uit}} \middle/ \frac{d P_{uit}}{P_{uit}} \right.\\
            &= e_{uit}^{store} + e_{uit}^{cate} + e_{uit}^{prod} + e_{uit}^{quant}
        $$

        Parameters
        ----------
            store_prob (jnp.ndarray): store visit probability
            category_elasticity (jnp.ndarray): category elasticity of a product
            product_elasticity (jnp.ndarray): product choice elasticity of a product
            product_demand_elasticity (jnp.ndarray): product demand elasticity of a product

        Returns
        -------
            jnp.ndarray: overall elasticity in shape of (n_customer, n_product)
        """
        overall_elasticity = (
            store_visit_elasticity
            + category_elasticity
            + product_elasticity
            + product_demand_elasticity
        )
        logging.debug(
            f"Global mean of overall elasticity: {overall_elasticity.mean(): .4f}"
        )
        return overall_elasticity
