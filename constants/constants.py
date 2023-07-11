from pathlib import Path
from datetime import datetime
import __main__
# REPO_PATH = Path.home() / "repositories/master_thesis_sfs"

OUTPUT_PATH_BASE = Path.home() / "Insync/OneDrive/master_thesis/repo_output"

now = datetime.utcnow()

OUTPUT_FOLDER = f"{Path(__main__.__file__).stem}_{now.strftime('%m_%d_%I_%M_%S')}"

OUTPUT_PATH = OUTPUT_PATH_BASE / OUTPUT_FOLDER


def get_output_path(now, session_name=None):
    base_path = OUTPUT_PATH_BASE / f"{Path(__main__.__file__).stem}"
    if session_name:
        base_path = base_path / session_name

    return base_path / f"{now.strftime('%m_%d_%H_%M_%S')}"


if __name__ == "__main__":
    print(OUTPUT_PATH)
    print(OUTPUT_PATH)

