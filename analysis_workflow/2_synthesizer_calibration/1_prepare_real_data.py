import hydra

from retailsynth.utils.dataset_viz import viz_main
from retailsynth.utils.storage import clear_cwd


@hydra.main(
    config_path=str("cfg"),
    config_name="real_dataset",
)
def run_real_data(cfg):
    clear_cwd(cfg.paths.processed_data)
    viz_main(cfg)


if __name__ == "__main__":
    run_real_data()
