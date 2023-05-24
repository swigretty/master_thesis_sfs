from pathlib import Path
from datetime import datetime
import __main__
# REPO_PATH = Path.home() / "repositories/master_thesis_sfs"

OUTPUT_PATH_BASE = Path.home() / "Insync/OneDrive/master_thesis/repo_output"

now = datetime.utcnow()

OUTPUT_FOLDER = f"{Path(__main__.__file__).stem}_{now.month}{now.day}"

OUTPUT_PATH = OUTPUT_PATH_BASE / OUTPUT_FOLDER

if __name__ == "__main__":
    print(OUTPUT_PATH)
    print(OUTPUT_PATH)

