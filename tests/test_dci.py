import pandas as pd

from scripts.dci import build_dci


def test_build_dci_requires_components():
    df = pd.DataFrame({"Internet_users": [10, 20]})
    try:
        build_dci(df, ["Internet_users", "Fixed_broadband_subscriptions", "Secure_servers"])
    except ValueError:
        assert True
    else:
        assert False
