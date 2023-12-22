import pytest

from retailsynth.feature_eng.feature_loader import initialize_feature_loader


@pytest.fixture
def sample_feature_loader(sample_config):
    feature_loader = initialize_feature_loader(sample_config.paths)
    return feature_loader
