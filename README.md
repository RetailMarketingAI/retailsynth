# RetailSynth: Synthetic Data Generation for Retail AI Evaluation
[![License](https://img.shields.io/github/license/moby/moby)](https://github.com/RetailMarketingAI/retailsynth/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Overview

RetailSynth is a sophisticated simulation environment designed to generate synthetic retail datasets. Its purpose is
the evaluation of AI systems in retail, focusing on personalized pricing promotions, product recommendation algorithms, 
and more. RetailSynth leverages a multi-stage model to simulate customer shopping behavior, incorporating price 
sensitivity and customer history. The environment is calibrated with publicly available grocery data, allowing for 
the creation of realistic shopping transactions. 

**DISCLAIMER:** RetailSynth is a research grade library and not production ready. The goal of open sourcing this library 
is to reproduce the analysis and results shown in the paper. Please cite the following paper when you use (parts of) 
this code:
> Xia Y, Arian A, Narayanamoorthy S and Mabry J (2023). RetailSynth: Synthetic Data Generation 
> for Retail AI Systems Evaluation

## Key Features

- **Multi-Stage Simulation Model**: Incorporates econometric models covering the full customer lifecycle, including 
  store visits, category choices, product choices, and purchase quantities.
- **Interpretability**: Based on utility theory and can incorporate customer-specific 
  features and treatments.
- **Price Policy Setting**: Employs a Markov model for generating realistic pricing strategies.

## Getting Started

### Prerequisites
* Python 3.10.10
* [Poetry]((https://python-poetry.org/)) for dependency management

### Installation
Clone the repository and set up the environment:

```sh
git clone git@github.com:RetailMarketingAI/retailsynth.git
cd retailsynth
poetry install  # install deps
poetry shell    # switch to the project environment
```

### Repository Structure

The code in this repository is organized into three primary directories:  
* `src`: Core `RetailSynth` library, featuring modules for data processing, feature engineering, and 
  the synthesizer itself.
* `analysis_workflow`: Contains workflows used in our research, including data analysis, synthesizer calibration, 
  and scenario analysis. These scripts and notebooks can be used to reproduce the results in the paper.
* `tests`: Unit tests ensuring code quality and reliability.

A directory tree with details of the submodules is provided below. 
```
RetailSynth
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── analysis_workflow              # Research workflows
│ ├── 1_complete_journey_eda        # Exploratory data analysis
│ ├── 2_synthesizer_calibration     # Synthesizer Calibration
│ └── 3_scenario_analysis           # Scenario Analysis
├── src                           
│  └── retailsynth                # Core RetailSynth library
│      ├── base_config.py           # Base configuration with defaults
│      ├── datasets                 # Module for processing Complete Journey Data
│      ├── feature_eng              # Feature engineering module
│      ├── synthesizer              # Core synthesizer module
│      └── utils                    # Misc utility functions
└── tests                         # Unit tests
```
 

### Configuration

In RetailSynth, we utilize [Hydra](https://hydra.cc/) for managing our project's configuration settings. Hydra's dynamic configuration
capabilities make it simple to extend and further customize RetailSynth to different business settings.

#### Configuration Structure

- **Base Configuration**: Global default configuration values are defined in `src/retailsynth/base_config.py`. This 
  base configuration serves as the foundation for all workflow-specific settings.
- **Workflow Configurations**: Each analysis workflow contains its own `cfg` subdirectory, holding configuration files
tailored for individual tasks within that workflow.

#### Example Hydra Configuration

Below is an example of a Hydra configuration in YAML format:

```yaml
# Context parameters for script execution
hydra:
  run:
    dir: analysis_workflow/1_complete_journey_eda/outputs  # Output directory
  job:
    chdir: True  # Change working directory to the job's directory

# Default settings are loaded from `src/retailsynth/base_config.py`
# The '_self_' directive ensures these defaults are overridden by this file's settings
defaults:
  - base_config
  - _self_

# Custom parameters (overrides the defaults)
n_customers_sampled: 100  # Number of customers to sample
n_workers: 4              # Number of workers to use
```

To customize configurations for your specific needs:

- Modify the YAML files in the `cfg` directory of the relevant workflow.
- Adjust the parameters as needed, ensuring they align with the structure defined in `base_config.py`.

For more detailed information on using and customizing Hydra configurations, refer to the [Hydra Documentation](https://hydra.cc/docs/intro/).

## Complete Journey Data EDA (`1_complete_journey_eda`)

Exploratory Data Analysis of The Complete Journey data and an overview of the pre-processing logic is contained 
in the `1_preprocess_analysis.ipynb` notebook. This notebook is a good entrypoint to become familiar with the 
contents and structure of The Complete Journey data. The notebook utilizes a single configuration file, 
located at `./cfg/real_dataset.yaml`. To speed up the analysis iterations, we recommend you modify 
the `n_customers_sampled` parameter within the configuration file to fit the compute resources available.


## Calibration Process (`2_synthesizer_calibration`)

We followed an iterative workflow to calibrate the synthesizer distributions to match the distributions from the 
Complete journey data. Tactically, this translated to:

* **Step 0**: [Pre-requisite] Process Complete Journey data and generate baseline distributions to compare against 
* **Step 1**: Choose a set of parameters for the Bayesian priors for the generative model that we have described 
  in the paper
* **Step 2**: Generate a batch of synthetic data
* **Step 3**: Inspect the key distributions of interest both quantitatively and qualitatively
* **Step 4**: Update the parameters and repeat till the distributions match well

Initially, our approach to calibration was manual, leveraging insights from academic literature, 
intuitive understanding, and business judgment for parameter setting and adjustments.
However, we encountered challenges in accurately replicating a few key 
target distributions. To help address these fitting issues, we shifted our approach to leverage Bayesian 
optimization for parameter sweeping using Optuna. We document both these approaches below. 

### Manual calibration workflow
We now switch to the POV of a user attempting to replicate our workflow.

To update the model parameters (**Step 1**), edit  
`./cfg/synthetic_data/synthetic_data_params.yaml`. This configuration specifies the input to the 
synthesizer and our current estimate of the parameter values for the prior distributions. This is a key file that you 
will frequently revise during calibration. Detailed documentation for each setting is provided below:
  
  ```yaml
  sample_time_steps: 53  # Duration of simulation in weeks.

  synthetic_data_setup:
    n_customer: 100  # Number of customers to be simulated.
    n_category: 3  # Number of product categories.
    n_product: 30  # Total number of products across all categories.
    category_product_count: [5, 10, 15]  # Distribution of products across categories (sum of this should be equal n_product).
    store_util_marketing_feature_mode: "random"  # Choose from ["random", "discount"]; set to "random" for calibration
    random_seed: 0  # Seed for random number generation, ensuring reproducibility.
    random_seed_range: 100  # Range of seed values for random processes.
    
    # Parameters for the prior distributions
    # details provided in Appendix A of the paper
    # coefficients to generate discount
    ...
    # coefficients to compute product price
    ...
    # coefficients to compute product utility
    ...
    # coefficients to compute category utility
    ...
    # coefficients to compute store visit probability
    ...
    # coefficients to compute product demand
    ...
  ```

Run the `synthesizer_validation.ipynb` notebook to perform **Step 0**, **Step 2** and **Step 3**. This runs the 
synthesizer using the above specified config and verifies how closely we match the target distributions. We recommend 
looking at the overlap of the key probability distributions (store visit, category, and product choice probabilities) 
as well as outcome distributions (time between visits, basket size, quantity purchased etc.)

In addition to the configs we have already introduced (i.e. `synthetic_data_params.yaml` and
`./cfg/real_dataset.yaml`), this notebook uses `synthetic_dataset.yaml`. This additional config
lets you override the default paths for output.

Once you have analyzed the notebook, you can nudge the parameters in the right direction by updating
the relevant values in `synthetic_data_params.yaml`.


### Parameter sweeping using Optuna workflow

While the calibration steps outlined above remains the same, we automate **Step 1**, **Step 2**,
and **Step 3** (partially) by defining an objective function and letting Optuna figure out the best
set of parameters. As discussed in the paper, due to structural reasons it is difficult to achieve a perfect 
match between the distributions. We provide a range of different objective functions for the fine-tuning that can optimize different target distributions.  

We use the KS-complement metric as the objective unless otherwise noted.
- **Store Visit Choice Optimization**: Optimize the fit of the store visit distribution.
- **Overall Demand Optimization**: Optimize the fit of the quantity purchased distribution.
- **Category Choice Optimization**: Optimize the fit of the category choice distribution according to a blended metric, summing the KS-complement metrics for the category choice and basket size distributions.
- **Product Choice Optimization**: Optimize the fit of the product choice distribution.
- **Combined Optimization**: Optimize the fit of the category and product choice distributions, summing their
  respective KS-complement metrics.

Begin the parameter sweeping process by processing The Complete Journey data (i.e. re-running **Step 0**). The 
configuration for this script is driven by `./cfg/real_dataset.yaml`. We suggest re-running this analysis rather than using data prepared previously, in case any parameters have been updated.  The processed data will be saved in 
the `outputs/data` directory by default. Run the script with the following command:

```sh
python ./analysis_workflow/2_synthesizer_calibration/1_prepare_real_data.py
```

Next, update `./cfg/parameter_sweeping.yaml` to specify:
- the optimization strategy
```yaml
hydra:
  sweeper:
    study_name: best_category_fit  # choose from one of ["best_category_fit", "best_product_fit", " best_overall_fit", "best_store_fit", and "best_demand_fit"]
```

- the specific parameters that requires sweeping
```yaml
hydra:
  sweeper:
    params:
      # Specify the parameters and the range of values to sweep here.
      # For example, the following two parameters are key ones to compute the category utility
      synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.loc: range(-5, -4.4, step=0.2)
      synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.scale: 0.1,0.5
```

There are additional settings that control the execution of the parameter sweep. Documentation for the key settings is provided below:
```yaml
hydra:
  run:
    dir: analysis_workflow/2_synthesizer_calibration/outputs
  job:
    chdir: True
  mode: MULTIRUN
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler # see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html for details
      seed: 0
    direction: maximize
    # specifies the optimization target
    study_name: best_category_fit  # choose from one of ["best_category_fit", "best_product_fit", " best_overall_fit", "best_store_fit", and "best_demand_fit"]
    storage: null
    n_trials: 1  # specifies the maximum number of runs to perform in a sweep
    n_jobs: 1  # specifies how many trials to run in parallel
    params:
      # Specify the parameters and the range of values to sweep here.
      # For example, the following two parameters are key ones to compute the category utility
      synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.loc: range(-5, -4.4, step=0.2)
      synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.scale: 0.1,0.5
  
  sweep:
    # sweeping report will be stored locally in the following directory
    dir: analysis_workflow/2_synthesizer_calibration/multirun/${hydra.sweeper.study_name}/${now:%Y-%m-%d_%H-%M-%S}/
    subdir: ${hydra.job.num}

paths:  # use the following to override the default
  processed_data: data/processed/synthetic_data_calib/
  txns_array_path: data/processed/synthetic_data_calib/txns_array/
  store_feature_path: data/processed/synthetic_data_calib/store_features/
  category_feature_path: data/processed/synthetic_data_calib/category_features/
  product_feature_path: data/processed/synthetic_data_calib/product_features/
```

Execute the parameter sweeping script with the following command:
```sh
python ./analysis_workflow/2_synthesizer_calibration/2_parameter_sweeping.py
```

Running this script effectively does **Step 1, 2 and 4** multiple times. The results of the sweeping process, 
including detailed reports, are stored in the 
`analysis_workflow/2_synthesizer_calibration/multirun/${hydra.sweeper.study_name}/${now:%Y-%m-%d_%H-%M-%S}/` 
directory. For example, if you set `study_name: best_category_fit` in the `parameter_sweeping.yaml` the output 
would be stored in `analysis_workflow/2_synthesizer_calibration/multirun/best_category_fit/${now:%Y-%m-%d_%H-%M-%S}/`. 
The optimal parameter values are outputted to the console and also saved in the `optimization_results.yaml` 
file in the results directory, as shown below:

```yaml
name: optuna
best_params:
  synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.loc: -4.6
  synthetic_data.synthetic_data_setup.category_choice_gamma_0j_cate.scale: 0.1
best_value: 1.4419610352479562  # KS-Complement metric (max=2 in this case as we are matching 2 distributions)
```

After performing the sweep, we can update the corresponding parameter values in `synthetic_dataset.yaml` to become our new baseline:

```yaml
  # coefficients to compute category utility
  category_choice_gamma_0j_cate:
    _target_: numpyro.distributions.Normal
    loc: -4.6
    scale: 0.1
```

Now, perform **Step 3** by running `synthesizer_validation.ipynb` notebook to verify the quantitative and 
qualitative fit of the chosen set of parameters. The process here is the same as the one described in the manual 
calibration approach.

**Note**: We've included the optimal parameter values obtained from our experiments in the `optimal` directory 
for each strategy. The provided `synthetic_data_params.yaml` file is pre-configured with these values. Simply
running `synthesizer_validation.ipynb` notebook will reproduce the calibration results presented in our paper.

## Scenario Analysis (`3_scenario_analysis`)

To replicate the scenario analysis results featured in our paper, you can run the `pricing_strategy_visualization.ipynb` notebook. This notebook executes the `run_scenarios.py` 
script to generate the synthetic data for each scenario and then generates the scenario analysis results.

The policies we investigated in our paper are detailed in the table below. 
| Policy | Effective Discount | Discount State Probability | Expected Discount Depth | $\alpha_{01}^{trans},\beta_{01}^{trans}$ | $\alpha_{11}^{trans},\beta_{11}^{trans}$ | $\alpha_d, \beta_d$ |
|--------|--------------------|----------------------------|-------------------------|-----------------------------------------|-----------------------------------------|---------------------|
| I      | 3%                 | 60%                        | 5%                      | (60,40)                                 | (60,40)                                 | (5,95)              |
| II     | 3%                 | 30%                        | 10%                     | (30,70)                                 | (30,70)                                 | (10,90)             |
| III    | 15%                | 60%                        | 25%                     | (60,40)                                 | (60,40)                                 | (25,75)             |
| IV     | 15%                | 30%                        | 50%                     | (30,70)                                 | (30,70)                                 | (50,50)             |
| V      | 24%                | 60%                        | 40%                     | (60,40)                                 | (60,40)                                 | (40,60)             |

*Configuration and Customization*:

- The setup for the synthesizer in the scenario analysis builds upon the calibration step, utilizing the optimal values from `synthetic_data_params.yaml`. However, 
  additional overrides are specified in `synthetic_dataset.yaml` to reflect our updated specification of store utility equation.
- **Policy Specification:** The policies outlined in the table above are defined in `./cfg/scenario_analysis.yaml`. For example, Policy I gets expressed in the config as:

  ```yaml
  scenarios:
    I: # high-frequency, low-discount policy
      synthetic_data:
        synthetic_data_setup:
          discount_depth_distribution:
            _target_: numpyro.distributions.Beta
            concentration1: 5 # alpha, as in the beta distribution
            concentration0: 95 # beta, as in the beta distribution
          discount_state_a_01:
            _target_: numpyro.distributions.Beta
            concentration1: 60
            concentration0: 40
          discount_state_a_11:
            _target_: numpyro.distributions.Beta
            concentration1: 60
            concentration0: 40
  ```

# Support
For support, questions, or feedback, please [file an issue](https://github.com/RetailMarketingAI/retailsynth/issues/new) on our GitHub repository.
