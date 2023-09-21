from pathlib import Path
from datetime import datetime
import __main__

# Please adapt
OUTPUT_PATH_BASE = Path.home() / "Insync/OneDrive/master_thesis/repo_output"

# Assumes "gp_experiments.py" is the file you used to produce the main
# results with please adapt accordingly if you used a different file to
# run the main experiments with
RESULTS_PATH = OUTPUT_PATH_BASE / "gp_experiments"

now = datetime.utcnow()

OUTPUT_FOLDER = (f"{Path(__main__.__file__).stem}_"
                 f"{now.strftime('%m_%d_%I_%M_%S')}")

OUTPUT_PATH = OUTPUT_PATH_BASE / OUTPUT_FOLDER


def get_output_path(now=None, session_name=None, experiment_name=None):
    base_path = OUTPUT_PATH_BASE / f"{Path(__main__.__file__).stem}"
    if experiment_name:
        base_path = base_path / experiment_name
    if session_name:
        base_path = base_path / session_name
    if now is None:
        return base_path
    return base_path / f"{now.strftime('%m_%d_%H_%M_%S')}"


if __name__ == "__main__":
    print(OUTPUT_PATH)
    print(OUTPUT_PATH)

