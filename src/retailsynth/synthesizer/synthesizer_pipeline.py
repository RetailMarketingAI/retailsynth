import jax.numpy as jnp

from retailsynth.base_config import SyntheticData
from retailsynth.synthesizer.config import initialize_synthetic_data_setup
from retailsynth.synthesizer.data_synthesizer import DataSynthesizer


def run_synthesizer(
    config: SyntheticData,
):
    """Run the synthesizer pipeline.

    This function applies the synthesizer pipeline to the raw data based on the provided configuration.

    Parameters
    ----------
    config : SyntheticData
        Configuration for the synthetic datasets.

    Returns
    -------
    tuple
        A tuple containing the synthesizer customers, products, and transactions data frames.
    """
    # read config
    synthesizer_config = initialize_synthetic_data_setup(config.synthetic_data_setup)
    n_week = config.sample_time_steps

    # initialize the synthesizer
    synthesizer = DataSynthesizer(synthesizer_config)

    # run the trajectory

    (
        trajectory,
        price_record,
        discount_record,
        price_with_coupon_record,
    ) = synthesizer.sample_trajectory(
        n_week,
    )
    # prepare output dataframe
    trx_df = synthesizer.convert_trajectory_to_df(
        trajectory, price_record, discount_record, price_with_coupon_record
    )
    customer_df = synthesizer.convert_customer_info_to_df()
    product_df = synthesizer.convert_product_info_to_df()
    customer_df = customer_df.loc[trx_df.index.levels[1].values]
    product_df = product_df.loc[trx_df.index.levels[2].values]

    return (
        customer_df,
        product_df,
        trx_df,
        synthesizer,
    )

def run_synthesizer_for_model_estimation(
    config: SyntheticData,
):
    """Run the synthesizer pipeline.

    This function applies the synthesizer pipeline to the raw data based on the provided configuration.

    Parameters
    ----------
    config : SyntheticData
        Configuration for the synthetic datasets.

    Returns
    -------
    tuple
        A tuple containing the synthesizer customers, products, and transactions data frames.
    """
    # read config
    synthesizer_config = initialize_synthetic_data_setup(config.synthetic_data_setup)
    n_week = config.sample_time_steps

    # initialize the synthesizer
    synthesizer = DataSynthesizer(synthesizer_config)

    # run the trajectory

    (
        trajectory,
        _,
        _,
        price_with_coupon_record,
    ) = synthesizer.sample_trajectory(
        n_week,
    )
    x = jnp.array(synthesizer.choice_decision_stats["x"])
    return x, price_with_coupon_record, trajectory