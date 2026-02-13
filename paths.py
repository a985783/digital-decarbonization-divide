from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def data_path(filename: str) -> Path:
    return project_path("data", filename)


def result_path(filename: str) -> Path:
    return project_path("results", filename)


def toolkit_path(filename: str) -> Path:
    return project_path("policy_toolkit", filename)


def assert_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path
