from retailsynth.feature_eng.feature_utils import convert_trx_df_to_numpy


def test_index_mapping(sample_indices):
    assert sample_indices["time_index"].get_values([0, 1]) == [1, 2]
    assert sample_indices["time_index"].get_index([1, 2]) == [0, 1]


def test_convert_trx_df_to_numpy(raw_sample_tables, sample_indices):
    _, _, txns = raw_sample_tables

    trx_array = convert_trx_df_to_numpy(
        txns,
        "item_qty",
        sample_indices,
    )

    assert trx_array.shape == (8, 4, 4)
    assert (trx_array == 0).sum() == 8 * 4 * 4 - len(txns)
    assert trx_array.sum() == float(txns["item_qty"].sum())
