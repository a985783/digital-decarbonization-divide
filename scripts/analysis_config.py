import os

import yaml

REQUIRED_KEYS = [
    "treatment_main",
    "treatment_secondary",
    "outcome",
    "moderators_X",
    "controls_W",
    "years",
    "groups",
    "cv",
    "bootstrap",
    "imputation",
    "dci_components",
    "wdi_indicators",
]


def load_config(path):
    config_path = path
    if not os.path.isabs(path):
        config_path = os.path.join(os.getcwd(), path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    return cfg
