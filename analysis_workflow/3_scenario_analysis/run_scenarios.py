from copy import deepcopy
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from retailsynth.utils.dataset_viz import viz_main
from retailsynth.utils.storage import clear_cwd


def overwrite_synthetic_data_config(
    original_cfg: DictConfig, overwrite_params: DictConfig, scenario_name: str
) -> DictConfig:
    """overwrite the original config for synthetic data generation with new parameters for the scenario run

    Parameters
    ----------
        original_cfg (DictConfig): the original config for synthetic data generation
        overwrite_params (DictConfig): parameter to overwrite
        scenario_name (str): name of the scenario

    Returns
    -------
        DictConfig: new config for the scenario run
    """
    # overwrite path
    config = deepcopy(original_cfg)
    for path_name, path in config["paths"].items():
        config["paths"][path_name] = Path(scenario_name, path)
    scenario_config = OmegaConf.merge(config, overwrite_params)
    return scenario_config


@hydra.main(
    config_path="cfg",
    config_name="scenario_analysis",
)
def run_scenario_analysis(scenario_cfgs):
    for scenario_name, overwrite_params in scenario_cfgs["scenarios"].items():
        # overwrite config
        scenario_config = overwrite_synthetic_data_config(
            original_cfg, overwrite_params, scenario_name
        )
        # gather data for one scenario
        clear_cwd(scenario_config.paths.processed_data)
        viz_main(scenario_config)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="cfg"):
        original_cfg = compose(config_name="synthetic_dataset")

    run_scenario_analysis()
