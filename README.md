# A Modular Framework for Backdoor Attacks and Defenses

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a modular and extensible framework for the empirical evaluation of backdoor attacks and defenses in machine learning. It was developed to support the experiments and ensure the reproducibility of our Systematization of Knowledge (SoK) paper on backdoor threats. 🔬

## Overview

The primary goal of this framework is to offer a standardized platform for researchers to implement, benchmark, and analyze backdoor vulnerabilities and countermeasures. By separating concerns into distinct modules—datasets, models, attacks, and defenses—we aim to foster fair comparisons and accelerate future research in this domain.

The framework is built with **PyTorch** and designed to handle both centralized and **federated learning (FL)** scenarios, making it suitable for a wide range of modern ML paradigms.

## Features

* **Modular Architecture**: Easily integrate new attacks, defenses, or models by extending the provided base classes.
* **Configuration-Driven Experiments**: Define and manage complex experiments using simple `.yaml` configuration files. No need to write boilerplate code for every experiment.
* **Reproducibility First**: Designed to replicate the results in our SoK paper and provide a stable foundation for new research.
* **Federated Learning Support**: Built-in modules to simulate and evaluate backdoor threats in federated settings.
* **Rich Set of Implementations**: Includes a variety of well-known algorithms out of the box:
    * **Attacks**: BadNets, Distributed Backdoor Attacks (DBA), Model Replacement, A3FL, IBA, and more.
    * **Defenses**: Krum, DP+Clip, Flame, Deepsight, and others.
    * **Models**: Standard architectures like ResNet.

## Project Structure

The repository is organized to maintain a clear separation of concerns:

* 📁 `experiments/`: Contains all the logic for running experiments.
  * 📁 `configs/`:  Stores YAML configuration files that define individual experiments.
  * 📄 `run_parallel.py`: The main entry point to execute a single experiment from a config file.
* 📁 `src/`: The core source code of the framework, organized by functionality.
  * 📁 `attacks/`:   Implementations of various backdoor attack algorithms.
  * 📁 `datasets/`:  Data loaders, transformations, and pre-processing scripts for different datasets.
  * 📁 `defenses/`:  Implementations of various backdoor defense mechanisms.
  * 📁 `fl/`:  Modules and logic specific to Federated Learning simulations.
  * 📁 `models/`:  Definitions of neural network architectures (e.g., ResNet, VGG).
  * 📄 `utils.py`: Common utility functions used across the project.
* 📄 `requirements.txt`: Lists all the Python dependencies required to run the project.



## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Ayoub-46/BackdoorFramework.git](https://github.com/Ayoub-46/BackdoorFramework.git)
    cd BackdoorFramework
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
---

## How to Run Experiments

Running experiments is a two-step process: defining the experiment in a config file and then executing it with one of the provided scripts.

### 1. Define Your Experiment

All experiments are defined in `.yaml` files located in the `experiments/configs/` directory. You can create a new file or modify an existing one. The configuration specifies the dataset, model, attack, defense, and training parameters.

### 2. Run Your Experiment

To run an experiment defined by a specific config file, use `experiments/run_parallel.py`, for instance:
```sh
python experiments/run_parallel --config experiments/configs/badnets_cifar10.yaml
```
