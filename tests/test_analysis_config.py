from scripts.analysis_config import load_config


def test_load_config_has_required_fields():
    cfg = load_config("analysis_spec.yaml")
    assert cfg["treatment_main"] == "DCI"
    assert "moderators_X" in cfg
    assert "controls_W" in cfg
