import os


def resolve_n_jobs(default=1):
    value = os.environ.get("REBUTTAL_N_JOBS")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
