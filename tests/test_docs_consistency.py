from pathlib import Path


def test_no_simulated_dci_in_paper():
    text = Path("paper.tex").read_text(encoding="utf-8")
    assert "Simulated" not in text
