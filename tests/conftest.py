import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = [ROOT / "data" / "sample.csv", ROOT / "data" / "synthetic_83.csv"]


def make_server(sid, gb=100.0, cpu=2, ram=8, os_v="Ubuntu 22.04", tags="", app="App", early=False, **extra):
    s = {"Server_ID": sid, "Hostname": sid.lower(), "App_Name": app, "OS_Version": os_v, "Business_Tags": tags,
         "Allocated_vCPU": cpu, "Allocated_RAM_GB": ram, "Transfer_GB": gb, "Early_Migration_Requested": early,
         "Risk_Level": "Low", "Risk_Score": 0, "Savings_Amount": 0.0, "SCC_Id": f"SCC:{sid}",
         "WCC_Id": f"WCC:{sid}", "Community_Id": f"COM:{sid}", "Normalized_CPU_Pct": 30.0}
    s.update(extra)
    return s


@pytest.fixture
def dataset_paths():
    return DATASETS
