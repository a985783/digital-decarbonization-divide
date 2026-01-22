import os

from scripts.joblib_utils import resolve_n_jobs


def test_resolve_n_jobs_default():
    if "REBUTTAL_N_JOBS" in os.environ:
        os.environ.pop("REBUTTAL_N_JOBS")
    assert resolve_n_jobs() == 1


def test_resolve_n_jobs_env_override():
    os.environ["REBUTTAL_N_JOBS"] = "2"
    assert resolve_n_jobs() == 2
    os.environ.pop("REBUTTAL_N_JOBS")
