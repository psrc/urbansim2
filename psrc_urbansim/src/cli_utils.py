import os
import glob
import pandas as pd
import orca
from pathlib import Path

timestr = pd.Timestamp.now().strftime("%Y%m%d")


def get_username():
     username = os.environ.get('USER') or os.environ.get('USERNAME') or os.getlogin()
     return username[:2].lower()


def unique_output_dir(base_dir):
     if not base_dir.exists():
          return base_dir
     for i in range(1, 10):
          candidate = base_dir.parent / (base_dir.name + '_' + str(i))
          if not candidate.exists():
               return candidate
     return base_dir.parent / (base_dir.name + '_9')


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )


def tables_in_base_year(config):
     h5store_path = Path(config['data_dir']) / config['store']
     h5store = pd.HDFStore(h5store_path, mode="r")
     store_table_names = orca.get_injectable('store_table_names_dict')
     return [t for t in orca.list_tables() if t in h5store or store_table_names.get(t, ' ') in h5store]


def find_latest_results(output_dir):
    """Find the most recent results*.h5 file across all run_* subdirectories."""
    pattern = str(Path(output_dir) / "run_*" / "results*.h5")
    h5_files = glob.glob(pattern)
    if not h5_files:
        raise FileNotFoundError(
            f"No results*.h5 files found in {output_dir}/run_*/. "
            "Specify 'store' in settings_indicators.yaml."
        )
    latest = max(h5_files, key=os.path.getmtime)
    return latest
