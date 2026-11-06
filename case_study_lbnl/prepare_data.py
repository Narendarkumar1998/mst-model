"""
Preprocessing of the LBNL Building 59 data set for the case study.

Source: N. Luo, Z. Wang, D. Blum, C. Weyandt, N. Bourassa, M. A. Piette, T. Hong,
"A three-year dataset supporting research on building energy management and occupancy
analytics", Scientific Data 9, 156 (2022), data on Dryad, doi:10.7941/D1N33Q,
licence CC BY 4.0. Office building in Berkeley, California, 2018-2020, cleaned data.

Reads the raw cleaned CSV files from the Dryad archive (folder "Bldg59_clean data")
and writes hourly means of the points used by the case study to
data/case_study_lbnl/bldg59_hourly.csv.gz (study period 2018-06-01 to 2019-02-28, the
months in which the camera occupant counts are recorded; the WiFi counts of the data set
only start in February 2020 and are not used).

Usage: python case_study_lbnl/prepare_data.py <path to "Bldg59_clean data">
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "case_study_lbnl" / "bldg59_hourly.csv.gz"
START, END = "2018-06-01", "2019-02-28 23:00"

FILES = {
    "ele": None,                     # end-use electricity, kW, 15 min
    "occ": None,                     # camera occupant counts, 3rd and 4th floor south, 1 min
    "site_weather": ["air_temp_set_1", "relative_humidity_set_1", "solar_radiation_set_1"],
    "rtu_sa_fr": None,               # supply airflow of rooftop units 1-4
    "rtu_sa_t": None,                # supply air temperature of rooftop units 1-4
    "rtu_oa_damper": None,           # outdoor-air damper position of rooftop units 1-4
    "hp_hws_temp": None,             # heat-pump heating-water supply temperature
    "zone_temp_interior": None,      # 16 interior temperature loggers (averaged)
    "zone_temp_sp_c": None,          # 41 zone cooling setpoints (averaged)
}


def load(folder: Path, name: str, cols):
    d = pd.read_csv(folder / f"{name}.csv")
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d["date"] = pd.to_datetime(d["date"], format="mixed")
    d = d.set_index("date").sort_index()
    if cols:
        d = d[cols]
    d = d.apply(pd.to_numeric, errors="coerce")
    return d.resample("h").mean()


def main():
    folder = Path(sys.argv[1])
    parts = []
    for name, cols in FILES.items():
        d = load(folder, name, cols)
        if name == "zone_temp_interior":
            d = d.mean(axis=1).to_frame("indoor_temp_mean")
        elif name == "zone_temp_sp_c":
            d = d.mean(axis=1).to_frame("cooling_setpoint_mean")
        parts.append(d)
        print(f"{name:20s} {list(d.columns)}")
    h = pd.concat(parts, axis=1, sort=True).loc[START:END]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(OUT, float_format="%.4f")
    print(h.notna().mean().round(3).to_string())
    print(f"{len(h)} hours -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
