import logging
from typing import List

import jax.numpy as jnp
import numpyro
from hydra.utils import instantiate
from jax import config, random
from jax.typing import ArrayLike
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

logging.info("Setting jax_enable_x64 to True")
config.update("jax_enable_x64", True)


class ArbitraryTypeConfig:
    arbitrary_types_allowed = True


@dataclass(config=ArbitraryTypeConfig)
class SyntheticParameters:
    """Define hyperparameter used to generate the synthetic dataset"""

    n_customer: int
    n_category: int
    n_product: int
    category_product_count: List[int]
    # we currently support two modes:
    # 1. "random" to generate marketing feature from a specified normal distribution
    # 2. "discount" to derive the marketing feature from product discount, using sum of the discount array
    store_util_marketing_feature_mode: str = "random"
    # random seeds to generate trajectory
    random_seed: int = 0
    random_seed_range: int = 100
    # discount coef
    discount_depth_distribution: numpyro.distributions.Distribution = None
    discount_state_a_01: ArrayLike = None
    discount_state_a_11: ArrayLike = None
    # coefficients to generate product price
    price_alpha_i0: ArrayLike = None  # in shape of (n_product, )
    price_alpha_1: ArrayLike = None  # in shape of (1, )
    # coefficients to generate product utility
    lowest_price: float = 0.01
    utility_beta_ui_z: ArrayLike = None  # in shape of (n_customer, n_product)
    utility_beta_ui_x: ArrayLike = None  # in shape of (n_customer, n_product)
    utility_beta_i_w: ArrayLike = None  # in shape of (n_product, )
    utility_beta_u_w: ArrayLike = None  # in shape of (n_customer, )
    utility_c: float = None  # type: ignore
    utility_clip_percentile: float = None  # type: ignore # for example, use 99 to represent 99%
    utility_error_distribution: numpyro.distributions.Distribution = None
    # coefficients to generate category purchase probability conditional on store visit
    category_choice_gamma_0j_cate: ArrayLike = None  # in shape of (n_category, )
    category_choice_gamma_1j_cate: ArrayLike = None  # in shape of (n_category, )
    # coefficients to generate store visit probability
    store_visit_theta_u: ArrayLike = None  # in shape of (n_customer, )
    store_visit_gamma_0_store: ArrayLike = None  # in shape of (1, )
    store_visit_gamma_1_store: ArrayLike = None  # in shape of (1, )
    store_visit_gamma_2_store: ArrayLike = None  # in shape of (1, )
    # coefficients to generate product demand
    purchase_quantity_gamma_0i_prod: ArrayLike = None  # in shape of (n_product, )
    purchase_quantity_gamma_1i_prod: ArrayLike = None  # in shape of (n_product, )

    def __post_init__(self):
        assert self.n_product == sum(self.category_product_count)
        assert self.n_category == len(self.category_product_count)

        self.utility_beta_ui_w = (
            self.utility_c
            * jnp.expand_dims(self.utility_beta_u_w, axis=-1)
            * jnp.expand_dims(self.utility_beta_i_w, axis=0)
        )  # in shape of (n_customer, n_product)
        self.transition_prob = jnp.stack(
            [self.discount_state_a_01, self.discount_state_a_11], axis=1
        )
        self.transition_prob = numpyro.distributions.Bernoulli(self.transition_prob)

    def __getitem__(self, index: str):
        try:
            return getattr(self, index)
        except AttributeError:
            raise KeyError(f"Invalid attribute: {index}")


def initialize_synthetic_data_setup(config_dict: DictConfig) -> SyntheticParameters:
    n_customer = config_dict["n_customer"]
    n_category = config_dict["n_category"]
    n_product = config_dict["n_product"]
    category_product_count = list(config_dict["category_product_count"])
    store_util_marketing_feature_mode = config_dict["store_util_marketing_feature_mode"]
    random_seed = config_dict["random_seed"]
    random_seed_range = config_dict["random_seed_range"]
    discount_depth_distribution = instantiate(
        config_dict["discount_depth_distribution"]
    )

    seed = random.PRNGKey(random_seed)
    discount_state_a_01 = instantiate(config_dict["discount_state_a_01"]).sample(
        key=seed, sample_shape=(n_product,)
    )
    discount_state_a_11 = instantiate(config_dict["discount_state_a_11"]).sample(
        key=seed, sample_shape=(n_product,)
    )
    price_alpha_i0 = instantiate(config_dict["price_alpha_i0"]).sample(
        key=seed, sample_shape=(n_product,)
    )
    price_alpha_1 = instantiate(config_dict["price_alpha_1"]).sample(
        key=seed, sample_shape=(1,)
    )
    lowest_price = config_dict["lowest_price"]
    utility_beta_ui_z = instantiate(config_dict["utility_beta_ui_z"]).sample(
        key=seed, sample_shape=(n_customer, n_product)
    )
    utility_beta_ui_x = instantiate(config_dict["utility_beta_ui_x"]).sample(
        key=seed, sample_shape=(n_customer, n_product)
    )
    utility_beta_u_w = instantiate(config_dict["utility_beta_u_w"]).sample(
        key=seed, sample_shape=(n_customer,)
    )
    utility_beta_i_w = instantiate(config_dict["utility_beta_i_w"]).sample(
        key=seed, sample_shape=(n_product,)
    )
    utility_c = config_dict["utility_c"]
    utility_clip_percentile = config_dict["utility_clip_percentile"]
    utility_error_distribution = instantiate(config_dict["utility_error_distribution"])
    category_choice_gamma_0j_cate = instantiate(
        config_dict["category_choice_gamma_0j_cate"]
    ).sample(key=seed, sample_shape=(n_category,))
    category_choice_gamma_1j_cate = instantiate(
        config_dict["category_choice_gamma_1j_cate"]
    ).sample(key=seed, sample_shape=(n_category,))
    store_visit_theta_u = instantiate(config_dict["store_visit_theta_u"]).sample(
        key=seed, sample_shape=(n_customer,)
    )
    store_visit_gamma_0_store = instantiate(
        config_dict["store_visit_gamma_0_store"]
    ).sample(key=seed, sample_shape=(1,))
    store_visit_gamma_1_store = instantiate(
        config_dict["store_visit_gamma_1_store"]
    ).sample(key=seed, sample_shape=(1,))
    store_visit_gamma_2_store = instantiate(
        config_dict["store_visit_gamma_2_store"]
    ).sample(key=seed, sample_shape=(1,))
    purchase_quantity_gamma_0i_prod = instantiate(
        config_dict["purchase_quantity_gamma_0i_prod"]
    ).sample(key=seed, sample_shape=(n_product,))
    purchase_quantity_gamma_1i_prod = instantiate(
        config_dict["purchase_quantity_gamma_1i_prod"]
    ).sample(key=seed, sample_shape=(n_customer, n_product))

    return SyntheticParameters(
        n_customer=n_customer,
        n_category=n_category,
        n_product=n_product,
        category_product_count=category_product_count,
        store_util_marketing_feature_mode=store_util_marketing_feature_mode,
        random_seed=random_seed,
        random_seed_range=random_seed_range,
        discount_depth_distribution=discount_depth_distribution,
        discount_state_a_01=discount_state_a_01,
        discount_state_a_11=discount_state_a_11,
        price_alpha_i0=price_alpha_i0,
        price_alpha_1=price_alpha_1,
        lowest_price=lowest_price,
        utility_beta_ui_z=utility_beta_ui_z,
        utility_beta_ui_x=utility_beta_ui_x,
        utility_beta_u_w=utility_beta_u_w,
        utility_c=utility_c,
        utility_beta_i_w=utility_beta_i_w,
        utility_clip_percentile=utility_clip_percentile,
        utility_error_distribution=utility_error_distribution,
        category_choice_gamma_0j_cate=category_choice_gamma_0j_cate,
        category_choice_gamma_1j_cate=category_choice_gamma_1j_cate,
        store_visit_theta_u=store_visit_theta_u,
        store_visit_gamma_0_store=store_visit_gamma_0_store,
        store_visit_gamma_1_store=store_visit_gamma_1_store,
        store_visit_gamma_2_store=store_visit_gamma_2_store,
        purchase_quantity_gamma_0i_prod=purchase_quantity_gamma_0i_prod,
        purchase_quantity_gamma_1i_prod=purchase_quantity_gamma_1i_prod,
    )  # type: ignore
