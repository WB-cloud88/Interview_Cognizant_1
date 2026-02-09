#!/usr/bin/env python3
"""
Generate realistic oil industry transformer monitoring data.
Two files: sensor_readings.csv and maintenance_log.csv
"""

import csv
import random
import math
from datetime import date, timedelta

random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
NUM_TRANSFORMERS = 50
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)
DAYS = (END_DATE - START_DATE).days + 1  # 731 days
MISSING_RATE = 0.05

# ── Equipment metadata ─────────────────────────────────────────────────────
equipment = []
for i in range(1, NUM_TRANSFORMERS + 1):
    eid = f"TRANS_{i:03d}"
    # Install dates: 1995-2015, weighted toward older for first ~20 units
    if i <= 20:
        year = random.randint(1995, 2005)
    elif i <= 35:
        year = random.randint(2003, 2010)
    else:
        year = random.randint(2008, 2015)
    install = date(year, random.randint(1, 12), random.randint(1, 28))
    # Last inspection: somewhere in 2022-2024 before or around start
    insp_offset = random.randint(0, 700)
    last_insp = date(2022, 1, 1) + timedelta(days=insp_offset)
    if last_insp > END_DATE:
        last_insp = END_DATE - timedelta(days=random.randint(10, 60))
    # Base load profile for this transformer (some run hotter)
    base_load = random.uniform(40, 65)
    equipment.append({
        "id": eid,
        "install_date": install,
        "last_inspection": last_insp,
        "base_load": base_load,
        "install_year": year,
    })

# ── Plan failure events ────────────────────────────────────────────────────
# 10-12 failures across 2 years
# Older transformers (pre-2005) more likely; high-load periods more likely
# Types: 30% cooling, 40% insulation, 30% sudden
NUM_FAILURES = 11

# Weight equipment by age (older = higher weight)
age_weights = []
for eq in equipment:
    age = 2023 - eq["install_year"]
    w = age ** 1.5  # non-linear: much more likely for old units
    age_weights.append(w)
total_w = sum(age_weights)
age_probs = [w / total_w for w in age_weights]

# Pick which transformers fail (allow repeats but not same month)
failure_types = (
    ["cooling"] * 3 +    # 30%  ~3
    ["insulation"] * 5 +  # 40%  ~5 (rounding: 4.4→5)
    ["sudden"] * 3         # 30%  ~3
)
random.shuffle(failure_types)

# High-load months: Jun-Aug (summer cooling), Dec-Jan (winter heating)
high_load_months = [6, 7, 8, 12, 1]

failures = []
used_combos = set()
for ftype in failure_types:
    # Pick transformer weighted by age
    while True:
        idx = random.choices(range(NUM_TRANSFORMERS), weights=age_probs, k=1)[0]
        # Pick a date biased toward high-load months
        if random.random() < 0.65:
            month = random.choice(high_load_months)
        else:
            month = random.randint(1, 12)
        year = random.choice([2023, 2024])
        day = random.randint(1, 28)
        fail_date = date(year, month, day)
        combo = (idx, fail_date.month, fail_date.year)
        if combo not in used_combos and START_DATE + timedelta(days=30) <= fail_date <= END_DATE - timedelta(days=5):
            used_combos.add(combo)
            break

    failures.append({
        "equipment_idx": idx,
        "equipment_id": equipment[idx]["id"],
        "date": fail_date,
        "type": ftype,
    })

failures.sort(key=lambda f: f["date"])

# ── Build a lookup of failure effects on sensor readings ───────────────────
# For each failure, define which days before the event show anomalies
# Key: (equipment_idx, date) -> dict of sensor adjustments
sensor_adjustments = {}

for f in failures:
    idx = f["equipment_idx"]
    fd = f["date"]
    if f["type"] == "cooling":
        # Temp rise 2-3 days before failure
        for d in range(1, 4):
            adj_date = fd - timedelta(days=d)
            key = (idx, adj_date)
            ramp = (4 - d) / 3.0  # stronger closer to failure
            sensor_adjustments[key] = {
                "oil_top_add": 8 * ramp + random.uniform(-1, 1),
                "oil_bot_add": 5 * ramp + random.uniform(-0.5, 0.5),
                "winding_add": 10 * ramp + random.uniform(-1, 2),
                "fan_override": 1,  # fans running full
            }
    elif f["type"] == "insulation":
        # Gradual winding temp rise over 3-4 weeks
        ramp_days = random.randint(21, 28)
        for d in range(1, ramp_days + 1):
            adj_date = fd - timedelta(days=d)
            key = (idx, adj_date)
            progress = (ramp_days - d + 1) / ramp_days  # 0→1 toward failure
            sensor_adjustments[key] = {
                "oil_top_add": 3 * progress + random.uniform(-0.5, 0.5),
                "oil_bot_add": 1.5 * progress + random.uniform(-0.3, 0.3),
                "winding_add": 12 * progress + random.uniform(-1, 1),
                "fan_override": None,
            }
    # sudden: no precursor adjustments

# ── Generate sensor_readings.csv ───────────────────────────────────────────
print("Generating sensor_readings.csv ...")

with open("data/sensor_readings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "equipment_id",
        "oil_temp_top_celsius", "oil_temp_bottom_celsius",
        "load_mva", "ambient_temp_celsius",
        "cooling_fan_status", "winding_temp_celsius"
    ])

    for day_offset in range(DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_year = current_date.timetuple().tm_yday

        # Seasonal ambient temperature: sinusoidal, peaks in July
        # Range roughly -10 to 40°C
        season_phase = (day_of_year - 15) / 365.0 * 2 * math.pi  # peak ~July
        ambient_base = 15 + 22 * math.sin(season_phase)
        # Day-of-week load factor (weekdays higher)
        dow = current_date.weekday()
        dow_factor = 1.0 if dow < 5 else 0.82

        for eq_idx, eq in enumerate(equipment):
            # 5% missing values: skip entire row
            if random.random() < MISSING_RATE:
                continue

            # Ambient with daily noise
            ambient = ambient_base + random.gauss(0, 3)
            ambient = round(max(-15, min(45, ambient)), 1)

            # Load: base + seasonal (higher in summer/winter peaks) + noise
            seasonal_load_bump = 8 * abs(math.sin(season_phase))  # peaks summer/winter
            load = eq["base_load"] + seasonal_load_bump * dow_factor + random.gauss(0, 5)
            load = round(max(20, min(85, load)), 1)

            # Oil temps depend on load and ambient
            load_fraction = (load - 20) / 65.0  # 0-1
            oil_top = 60 + 20 * load_fraction + 0.15 * ambient + random.gauss(0, 1.5)
            oil_bot = oil_top - 8 - random.uniform(2, 6)

            # Winding temp: hotter than oil top
            winding = oil_top + 8 + 5 * load_fraction + random.gauss(0, 1.2)

            # Cooling fan: ON if oil_top > 72 or load > 65
            fan = 1 if (oil_top > 72 or load > 65) else 0
            # Random fan behavior near threshold
            if 68 < oil_top < 75:
                fan = random.choice([0, 1])

            # Apply failure precursor adjustments
            key = (eq_idx, current_date)
            if key in sensor_adjustments:
                adj = sensor_adjustments[key]
                oil_top += adj["oil_top_add"]
                oil_bot += adj["oil_bot_add"]
                winding += adj["winding_add"]
                if adj["fan_override"] is not None:
                    fan = adj["fan_override"]

            # Clamp to realistic ranges
            oil_top = round(max(55, min(95, oil_top)), 1)
            oil_bot = round(max(50, min(80, oil_bot)), 1)
            winding = round(max(65, min(110, winding)), 1)

            writer.writerow([
                current_date.isoformat(),
                eq["id"],
                oil_top, oil_bot, load, ambient, fan, winding
            ])

print(f"  sensor_readings.csv written.")

# ── Generate maintenance_log.csv ──────────────────────────────────────────
print("Generating maintenance_log.csv ...")

events = []

# Add failure events
for f in failures:
    events.append({
        "equipment_id": f["equipment_id"],
        "install_date": equipment[f["equipment_idx"]]["install_date"],
        "last_inspection": equipment[f["equipment_idx"]]["last_inspection"],
        "event_date": f["date"],
        "event_type": "FAILURE",
    })

# Add scheduled maintenance: each transformer gets 1-2 over 2 years
for eq in equipment:
    n_sched = random.choice([1, 1, 2])
    for _ in range(n_sched):
        offset = random.randint(30, DAYS - 30)
        ev_date = START_DATE + timedelta(days=offset)
        events.append({
            "equipment_id": eq["id"],
            "install_date": eq["install_date"],
            "last_inspection": eq["last_inspection"],
            "event_date": ev_date,
            "event_type": "SCHEDULED_MAINTENANCE",
        })

# Add cooling repairs: ~15 events, biased toward summer
for _ in range(15):
    eq = random.choice(equipment)
    month = random.choices(range(1, 13), weights=[1,1,2,2,3,5,5,5,3,2,1,1])[0]
    year = random.choice([2023, 2024])
    day = random.randint(1, 28)
    ev_date = date(year, month, day)
    events.append({
        "equipment_id": eq["id"],
        "install_date": eq["install_date"],
        "last_inspection": eq["last_inspection"],
        "event_date": ev_date,
        "event_type": "COOLING_REPAIR",
    })

# Add oil changes: ~20 events spread out
for _ in range(20):
    eq = random.choice(equipment)
    offset = random.randint(10, DAYS - 10)
    ev_date = START_DATE + timedelta(days=offset)
    events.append({
        "equipment_id": eq["id"],
        "install_date": eq["install_date"],
        "last_inspection": eq["last_inspection"],
        "event_date": ev_date,
        "event_type": "OIL_CHANGE",
    })

# Sort by event_date
events.sort(key=lambda e: e["event_date"])

with open("data/maintenance_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "equipment_id", "install_date", "last_inspection_date",
        "event_date", "event_type"
    ])
    for ev in events:
        writer.writerow([
            ev["equipment_id"],
            ev["install_date"].isoformat(),
            ev["last_inspection"].isoformat(),
            ev["event_date"].isoformat(),
            ev["event_type"],
        ])

print(f"  maintenance_log.csv written.")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nSummary:")
print(f"  Failures: {len(failures)}")
for f in failures:
    age = 2023 - equipment[f['equipment_idx']]['install_year']
    print(f"    {f['equipment_id']} ({age}yr old) - {f['type']:12s} on {f['date']} ")
print(f"  Total maintenance events: {len(events)}")
