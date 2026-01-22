from scripts.analysis_config import load_config
from scripts.wdi_indicators import load_indicators


def test_required_wdi_codes_present():
    cfg = load_config("analysis_spec.yaml")
    indicators = load_indicators(cfg)
    for code in [
        "IT.NET.USER.ZS",
        "IT.NET.BBND.P2",
        "IT.NET.SECR.P6",
        "BX.GSR.CCIS.ZS",
    ]:
        assert code in indicators
