import jax.numpy as jnp
from jax import random
import pytest

from retailsynth.synthesizer.config import initialize_synthetic_data_setup
from retailsynth.synthesizer.data_synthesizer import DataSynthesizer


@pytest.fixture(scope="module")
def data_synthesizer(sample_config):
    synthesizer_config = initialize_synthetic_data_setup(
        sample_config.synthetic_data.synthetic_data_setup
    )
    synthesizer = DataSynthesizer(synthesizer_config)
    return synthesizer


@pytest.fixture(scope="module")
def initial_store_visit_prob(data_synthesizer):
    prob = jnp.ones(
        shape=(data_synthesizer.n_customer,),
    )
    return prob


@pytest.fixture(scope="module")
def initial_store_visit(data_synthesizer):
    obs = jnp.ones(
        shape=(data_synthesizer.n_customer,),
    )
    return obs


@pytest.fixture(scope="module")
def probability_for_each_step(
    data_synthesizer, initial_store_visit, initial_store_visit_prob
):
    discount = data_synthesizer.sample_discount()
    product_price = data_synthesizer.compute_product_price(discount)
    coupon = data_synthesizer.sample_coupon()
    product_price_with_coupon = data_synthesizer.compute_product_price_with_coupon(
        product_price, coupon
    )
    product_utility = data_synthesizer.compute_product_utility(
        product_price_with_coupon
    )
    category_utility = data_synthesizer.compute_category_utility(product_utility)

    product_choice_prob = (
        data_synthesizer.compute_product_purchase_conditional_probability(
            product_utility
        )
    )
    category_choice_prob = (
        data_synthesizer.compute_category_purchase_conditional_probability(
            category_utility
        )
    )
    marketing_feature = data_synthesizer.sample_marketing_feature(
        discount=discount, coupon=coupon
    )
    store_visit_prob = data_synthesizer.compute_store_visit_probability(
        category_utility,
        initial_store_visit_prob,
        initial_store_visit,
        marketing_feature,
    )
    product_demand_mean = data_synthesizer.compute_product_demand_mean(product_utility)

    return (
        store_visit_prob,
        category_choice_prob,
        product_choice_prob,
        product_demand_mean,
    )


@pytest.fixture(scope="module")
def decision_for_each_step(data_synthesizer, probability_for_each_step):
    return data_synthesizer._sample_decisions(*probability_for_each_step)


def check_array_between_0_1(array):
    return jnp.all((array >= 0) & (array <= 1))


def check_decision_binary(array):
    return jnp.all((array == 0) | (array == 1))


def test_attributes(data_synthesizer, sample_config):
    config = sample_config.synthetic_data.synthetic_data_setup
    n_customer = data_synthesizer.n_customer
    n_category = data_synthesizer.n_category
    n_product = data_synthesizer.n_product
    assert n_customer == config["n_customer"]
    assert n_category == config["n_category"]
    assert n_product == config["n_product"]
    # check if every coefficient are in the right shape
    assert data_synthesizer.price_alpha_i0.shape == (n_product,)
    assert data_synthesizer.price_alpha_1.shape == (1,)
    assert data_synthesizer.utility_beta_ui_z.shape == (n_customer, n_product)
    assert data_synthesizer.utility_beta_ui_x.shape == (n_customer, n_product)
    assert data_synthesizer.utility_beta_ui_w.shape == (n_customer, n_product)
    assert data_synthesizer.category_choice_gamma_0j_cate.shape == (n_category,)
    assert data_synthesizer.category_choice_gamma_1j_cate.shape == (n_category,)
    assert data_synthesizer.store_visit_theta_u.shape == (n_customer,)
    assert data_synthesizer.store_visit_gamma_0_store.shape == (1,)
    assert data_synthesizer.store_visit_gamma_1_store.shape == (1,)
    assert data_synthesizer.store_visit_gamma_2_store.shape == (1,)
    assert data_synthesizer.purchase_quantity_gamma_0i_prod.shape == (n_product,)
    assert data_synthesizer.purchase_quantity_gamma_1i_prod.shape == (
        n_customer,
        n_product,
    )


def test_data_synthesizer_initialization(sample_config):
    mode_name = "Unsupported mode"
    sample_config.synthetic_data.synthetic_data_setup[
        "store_util_marketing_feature_mode"
    ] = mode_name
    synthesizer_config = initialize_synthetic_data_setup(
        sample_config.synthetic_data.synthetic_data_setup
    )
    with pytest.raises(ValueError) as excinfo:
        _ = DataSynthesizer(synthesizer_config)
    assert f"Invalid marketing feature mode: {mode_name}" in str(excinfo.value)


def test_generate_product_price(data_synthesizer):
    discount = data_synthesizer.sample_discount()
    product_price = data_synthesizer.compute_product_price(discount)
    assert product_price.shape == (data_synthesizer.n_product,)
    assert (product_price > 0).all()


def test_generate_product_price_with_coupon(data_synthesizer):
    discount = data_synthesizer.sample_discount()
    coupon = random.uniform(
        data_synthesizer.random_seed,
        shape=(data_synthesizer.n_customer, data_synthesizer.n_category),
    )
    product_price = data_synthesizer.compute_product_price(discount)
    product_price_with_coupon = data_synthesizer.compute_product_price_with_coupon(
        product_price, coupon
    )
    assert product_price_with_coupon.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (product_price_with_coupon > 0).all()


def test_prob_for_each_step(data_synthesizer, probability_for_each_step):
    (
        store_visit_prob,
        category_choice_prob,
        product_choice_prob,
        product_demand_mean,
    ) = probability_for_each_step

    assert product_choice_prob.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert check_array_between_0_1(product_choice_prob)

    assert category_choice_prob.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_category,
    )
    assert check_array_between_0_1(category_choice_prob)

    assert store_visit_prob.shape == (data_synthesizer.n_customer,)
    assert check_array_between_0_1(store_visit_prob)

    assert product_demand_mean.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (product_demand_mean > 0).all()


def test_decision_for_each_step(data_synthesizer, decision_for_each_step):
    (
        store_visit,
        category_choice,
        product_choice,
        product_demand,
    ) = decision_for_each_step

    assert product_choice.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert check_decision_binary(product_choice)
    # drawing sample from conditional product purchase probability,
    # the result should satisfy the assumption that customer only buy one product from each product
    assert (product_choice.sum(axis=1) == data_synthesizer.n_category).all()

    assert category_choice.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_category,
    )
    assert check_decision_binary(category_choice)

    assert store_visit.shape == (data_synthesizer.n_customer,)
    assert check_decision_binary(store_visit)

    assert product_demand.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (product_demand >= 1).all()


def test_joint_decision(data_synthesizer, decision_for_each_step):
    joint_decision = data_synthesizer.compute_joint_decision(*decision_for_each_step)
    assert joint_decision.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (joint_decision >= 0).all()


def test_store_visit_elasticity(data_synthesizer, initial_store_visit_prob):
    discount = data_synthesizer.sample_discount()
    coupon = jnp.zeros((data_synthesizer.n_customer, data_synthesizer.n_product))
    store_visit_elasticity = data_synthesizer.compute_store_visit_elasticity(
        initial_store_visit_prob, discount=discount, coupon=coupon
    )
    assert store_visit_elasticity.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )


def test_category_elasticity(data_synthesizer, probability_for_each_step):
    (
        _,
        category_choice_prob,
        product_choice_prob,
        _,
    ) = probability_for_each_step
    category_elasticity = data_synthesizer.compute_category_elasticity(
        product_choice_prob, category_choice_prob
    )
    assert category_elasticity.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )


def test_product_elasticity(data_synthesizer, probability_for_each_step):
    (
        _,
        _,
        product_choice_prob,
        _,
    ) = probability_for_each_step
    product_elasticity = data_synthesizer.compute_product_elasticity(
        product_choice_prob
    )
    assert product_elasticity.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )


def test_product_demand_elasticity(data_synthesizer, probability_for_each_step):
    (
        _,
        _,
        _,
        product_demand_mean,
    ) = probability_for_each_step
    product_demand_elasticity = data_synthesizer.compute_product_demand_elasticity(
        product_demand_mean
    )
    assert product_demand_elasticity.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )


def test_overall_elasticity(data_synthesizer, probability_for_each_step):
    (
        store_visit_prob,
        category_choice_prob,
        product_choice_prob,
        product_demand_mean,
    ) = probability_for_each_step
    discount = data_synthesizer.sample_discount()
    coupon = jnp.zeros((data_synthesizer.n_customer, data_synthesizer.n_product))
    store_visit_elasticity = data_synthesizer.compute_store_visit_elasticity(
        store_visit_prob, discount=discount, coupon=coupon
    )
    category_elasticity = data_synthesizer.compute_category_elasticity(
        product_choice_prob, category_choice_prob
    )
    product_elasticity = data_synthesizer.compute_product_elasticity(
        product_choice_prob
    )
    product_demand_elasticity = data_synthesizer.compute_product_demand_elasticity(
        product_demand_mean
    )
    overall_elasticity = data_synthesizer.compute_overall_elasticity(
        store_visit_elasticity,
        category_elasticity,
        product_elasticity,
        product_demand_elasticity,
    )
    assert overall_elasticity.shape == (
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )


def test_sample_trajectory(data_synthesizer, sample_config):
    n_week = sample_config.synthetic_data.sample_time_steps
    (
        trajectory,
        price_record,
        discount_record,
        price_with_coupon_record,
    ) = data_synthesizer.sample_trajectory(
        n_week,
    )

    assert trajectory.shape == (
        n_week,
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (trajectory >= 0).all()
    assert trajectory.dtype == "int"

    assert price_record.shape == (
        n_week,
        data_synthesizer.n_product,
    )
    assert (price_record > 0).all()

    assert discount_record.shape == (
        n_week,
        data_synthesizer.n_product,
    )
    assert check_array_between_0_1(discount_record)

    assert price_with_coupon_record.shape == (
        n_week,
        data_synthesizer.n_customer,
        data_synthesizer.n_product,
    )
    assert (price_with_coupon_record > 0).all()


def test_discount_state(data_synthesizer):
    discount = jnp.array(data_synthesizer.choice_decision_stats["discount"])
    discount_state = jnp.array(data_synthesizer.choice_decision_stats["discount_state"])
    discount_state_0 = jnp.where(discount_state == 0)
    assert (discount[discount_state_0] == 0).all()


def test_df_conversion(data_synthesizer, sample_config):
    n_week = sample_config.synthetic_data.sample_time_steps
    (
        trajectory,
        price_record,
        discount_record,
        price_with_coupon_record,
    ) = data_synthesizer.sample_trajectory(
        n_week,
    )

    trx_df = data_synthesizer.convert_trajectory_to_df(
        trajectory, price_record, discount_record, price_with_coupon_record
    )
    customer_df = data_synthesizer.convert_customer_info_to_df()
    product_df = data_synthesizer.convert_product_info_to_df()

    assert set(trx_df.columns) == set(
        [
            "item_qty",
            "discount_portion",
            "unit_price",
            "sales_amt",
            "unit_price_with_coupon",
        ]
    )

    assert len(customer_df) == data_synthesizer.n_customer

    assert set(product_df.columns) == set(["category_nbr", "all"])
    assert len(product_df) == data_synthesizer.n_product
