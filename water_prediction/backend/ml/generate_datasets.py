import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "waternet_dummy_data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_drinking_dataset(n_samples=500):
    # Features: ph, turbidity, temperature, do, ec, tds
    # Calibrated distributions across clean, moderate, and degraded water bodies
    
    # 45% clean/good, 35% poor/very poor, 20% unfit/extreme
    n_clean = int(n_samples * 0.45)
    n_mod = int(n_samples * 0.35)
    n_poor = n_samples - n_clean - n_mod

    # Clean water
    ph_clean = np.random.normal(7.2, 0.3, n_clean).clip(6.5, 8.2)
    turb_clean = np.random.exponential(1.5, n_clean).clip(0.1, 4.8)
    temp_clean = np.random.normal(24.0, 2.5, n_clean).clip(18.0, 30.0)
    do_clean = np.random.normal(7.8, 0.8, n_clean).clip(6.0, 11.0)
    ec_clean = np.random.normal(380, 80, n_clean).clip(150, 580)
    tds_clean = (ec_clean * np.random.uniform(0.62, 0.68, n_clean)).clip(100, 400)

    # Moderate water (borderline/poor)
    ph_mod = np.where(np.random.rand(n_mod) > 0.5, np.random.normal(6.2, 0.3, n_mod), np.random.normal(8.6, 0.3, n_mod))
    turb_mod = np.random.uniform(5.5, 18.0, n_mod)
    temp_mod = np.random.normal(26.0, 3.0, n_mod).clip(19.0, 33.0)
    do_mod = np.random.normal(4.8, 0.9, n_mod).clip(3.5, 6.2)
    ec_mod = np.random.normal(950, 180, n_mod).clip(600, 1400)
    tds_mod = (ec_mod * np.random.uniform(0.63, 0.69, n_mod)).clip(400, 950)

    # Poor / Unfit water
    ph_poor = np.where(np.random.rand(n_poor) > 0.5, np.random.normal(5.4, 0.5, n_poor), np.random.normal(9.3, 0.4, n_poor))
    turb_poor = np.random.uniform(18.0, 55.0, n_poor)
    temp_poor = np.random.normal(28.5, 3.5, n_poor).clip(20.0, 36.0)
    do_poor = np.random.normal(2.5, 1.0, n_poor).clip(0.5, 4.2)
    ec_poor = np.random.normal(1800, 400, n_poor).clip(1300, 3200)
    tds_poor = (ec_poor * np.random.uniform(0.64, 0.72, n_poor)).clip(850, 2200)

    ph = np.concatenate([ph_clean, ph_mod, ph_poor]).round(2)
    turb = np.concatenate([turb_clean, turb_mod, turb_poor]).round(2)
    temp = np.concatenate([temp_clean, temp_mod, temp_poor]).round(1)
    do = np.concatenate([do_clean, do_mod, do_poor]).round(2)
    ec = np.concatenate([ec_clean, ec_mod, ec_poor]).round(1)
    tds = np.concatenate([tds_clean, tds_mod, tds_poor]).round(1)

    # Calculate Weighted Arithmetic Water Quality Index (WQI)
    # Standard values (Sn): pH=8.5 (ideal 7.0), Turbidity=5, DO=5 (ideal 14.6), EC=300, TDS=500
    # Weights wi: pH=0.22, Turb=0.18, DO=0.28, EC=0.16, TDS=0.16
    q_ph = 100 * np.abs(ph - 7.0) / (8.5 - 7.0)
    q_turb = 100 * (turb / 5.0)
    q_do = 100 * np.abs(14.6 - do) / (14.6 - 5.0)
    q_ec = 100 * (ec / 300.0)
    q_tds = 100 * (tds / 500.0)

    weights = [0.22, 0.18, 0.28, 0.16, 0.16]
    wqi_raw = (q_ph * 0.22 + q_turb * 0.18 + q_do * 0.28 + q_ec * 0.16 + q_tds * 0.16)
    wqi = wqi_raw.round(2)

    grades = []
    suitables = []
    for val, p, t, d in zip(wqi, ph, turb, do):
        if val < 50 and 6.5 <= p <= 8.5 and t <= 5.0 and d >= 6.0:
            grades.append("Excellent")
            suitables.append(1)
        elif val < 100 and 6.5 <= p <= 8.5 and t <= 8.0 and d >= 5.0:
            grades.append("Good")
            suitables.append(1)
        elif val < 180:
            grades.append("Poor")
            suitables.append(0)
        elif val < 280:
            grades.append("Very Poor")
            suitables.append(0)
        else:
            grades.append("Unfit")
            suitables.append(0)

    df = pd.DataFrame({
        "ph": ph,
        "turbidity": turb,
        "temperature": temp,
        "do": do,
        "ec": ec,
        "tds": tds,
        "wqi": wqi,
        "grade": grades,
        "suitable_drinking": suitables
    })

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_file = os.path.join(DATA_DIR, "drinking_water_dataset.csv")
    df.to_csv(out_file, index=False)
    print(f"Created {out_file} with {len(df)} rows. Suitability distribution: {dict(df['suitable_drinking'].value_counts())}")
    return df


def generate_irrigation_dataset(n_samples=500):
    # FEATURES: ph, ec, tds, na_meq, ca_meq, mg_meq, chloride_meq, bicarbonate_meq, sar, sodium_percent
    # LABELS: grade, suitable_irrigation
    
    n_good = int(n_samples * 0.52)
    n_bad = n_samples - n_good

    # Good irrigation water
    ph_good = np.random.normal(7.3, 0.4, n_good).clip(6.5, 8.3)
    ec_good = np.random.uniform(150, 1100, n_good)
    tds_good = ec_good * np.random.uniform(0.63, 0.68, n_good)
    ca_good = np.random.uniform(1.8, 9.0, n_good)
    mg_good = np.random.uniform(0.8, 4.5, n_good)
    na_good = np.random.uniform(0.5, 7.5, n_good)
    cl_good = np.random.uniform(1.0, 7.0, n_good)
    hco3_good = np.random.uniform(1.5, 6.0, n_good)

    # Poor irrigation water (High salinity/sodicity)
    ph_bad = np.where(np.random.rand(n_bad) > 0.5, np.random.normal(8.7, 0.4, n_bad), np.random.normal(6.1, 0.3, n_bad))
    ec_bad = np.random.uniform(1200, 3800, n_bad)
    tds_bad = ec_bad * np.random.uniform(0.64, 0.72, n_bad)
    ca_bad = np.random.uniform(4.0, 14.0, n_bad)
    mg_bad = np.random.uniform(2.5, 9.0, n_bad)
    na_bad = np.random.uniform(12.0, 38.0, n_bad)
    cl_bad = np.random.uniform(8.0, 26.0, n_bad)
    hco3_bad = np.random.uniform(5.0, 14.0, n_bad)

    ph = np.concatenate([ph_good, ph_bad]).round(2)
    ec = np.concatenate([ec_good, ec_bad]).round(2)
    tds = np.concatenate([tds_good, tds_bad]).round(2)
    ca = np.concatenate([ca_good, ca_bad]).round(2)
    mg = np.concatenate([mg_good, mg_bad]).round(2)
    na = np.concatenate([na_good, na_bad]).round(2)
    cl = np.concatenate([cl_good, cl_bad]).round(2)
    hco3 = np.concatenate([hco3_good, hco3_bad]).round(2)

    # Exact chemical SAR & Sodium% formulas:
    # SAR = Na / sqrt((Ca + Mg) / 2)
    sar = (na / np.sqrt((ca + mg) / 2.0)).round(2)
    # Sodium% = (Na / (Ca + Mg + Na)) * 100
    na_pct = ((na / (ca + mg + na)) * 100).round(1)

    grades = []
    suitables = []
    for e, s, npct, p in zip(ec, sar, na_pct, ph):
        if e < 350 and s < 6.0 and npct < 40 and (6.5 <= p <= 8.2):
            grades.append("Excellent")
            suitables.append(1)
        elif e < 1150 and s < 10.0 and npct < 58 and (6.2 <= p <= 8.5):
            grades.append("Good")
            suitables.append(1)
        elif e < 2100 and s < 18.0 and npct < 72:
            grades.append("Poor")
            suitables.append(0)
        elif e < 3200 or s < 26.0:
            grades.append("Very Poor")
            suitables.append(0)
        else:
            grades.append("Unfit")
            suitables.append(0)

    df = pd.DataFrame({
        "ph": ph,
        "ec": ec,
        "tds": tds,
        "na_meq": na,
        "ca_meq": ca,
        "mg_meq": mg,
        "chloride_meq": cl,
        "bicarbonate_meq": hco3,
        "sar": sar,
        "sodium_percent": na_pct,
        "grade": grades,
        "suitable_irrigation": suitables
    })

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_file = os.path.join(DATA_DIR, "irrigation_water_dataset.csv")
    df.to_csv(out_file, index=False)
    print(f"Created {out_file} with {len(df)} rows. Suitability distribution: {dict(df['suitable_irrigation'].value_counts())}")
    return df


def generate_live_readings():
    # 240 rows: 5 monitoring points N1-N5, 48 readings each (hourly)
    # node_id,timestamp,latitude,longitude,ph,turbidity,temperature,do,ec,tds
    # Specific scenarios:
    # - Contamination event on N3 (spikes at hours 22 to 34)
    # - Stuck turbidity sensor on N2 (constant value 7.16 for hours 10 to 32)
    # - Missing pH values on N5 (empty strings on specific timestamps)
    
    start_time = datetime(2026, 9, 28, 6, 0, 0)
    nodes = [
        {"id": "N1", "lat": 17.3850, "lon": 78.4867, "name": "Hussain Sagar Monitoring Station"},
        {"id": "N2", "lat": 17.4065, "lon": 78.4772, "name": "Sanjeevaiah Lake Point"},
        {"id": "N3", "lat": 17.4500, "lon": 78.3800, "name": "Durgam Cheruvu Canal"},
        {"id": "N4", "lat": 17.3600, "lon": 78.5500, "name": "Mir Alam Tank Basin"},
        {"id": "N5", "lat": 17.5000, "lon": 78.5200, "name": "Fox Sagar Inflow Point"}
    ]

    all_rows = []

    for t_idx in range(48):
        cur_time = start_time + timedelta(hours=t_idx)
        time_str = cur_time.strftime("%Y-%m-%d %H:%M:%S")

        for node in nodes:
            nid = node["id"]
            lat = node["lat"]
            lon = node["lon"]

            # Diurnal temperature cycle
            temp = round(24.5 + 4.5 * np.sin((t_idx - 8) * np.pi / 12) + np.random.normal(0, 0.4), 1)

            if nid == "N1": # Generally clean
                ph = round(7.20 + 0.15 * np.sin(t_idx * np.pi / 24) + np.random.normal(0, 0.05), 2)
                turb = round(2.4 + np.random.uniform(-0.4, 0.6), 2)
                do = round(6.8 + np.random.normal(0, 0.2), 2)
                ec = round(480 + np.random.normal(0, 15), 1)
                tds = round(ec * 0.65, 1)

            elif nid == "N2": # Stuck turbidity sensor
                ph = round(7.10 + np.random.normal(0, 0.08), 2)
                # Stuck value 7.16 between t=10 and t=32
                if 10 <= t_idx <= 32:
                    turb = 7.16
                else:
                    turb = round(6.5 + np.random.uniform(-0.5, 0.8), 2)
                do = round(5.9 + np.random.normal(0, 0.25), 2)
                ec = round(980 + np.random.normal(0, 25), 1)
                tds = round(ec * 0.65, 1)

            elif nid == "N3": # Severe contamination event at t=20..34
                if 20 <= t_idx <= 34:
                    severity = np.sin((t_idx - 20) * np.pi / 14)
                    ph = round(8.40 + 0.8 * severity + np.random.normal(0, 0.05), 2)
                    turb = round(12.0 + 35.0 * severity + np.random.normal(0, 1.0), 2)
                    do = round(max(1.2, 5.5 - 4.2 * severity + np.random.normal(0, 0.15)), 2)
                    ec = round(1350 + 1200 * severity + np.random.normal(0, 30), 1)
                    tds = round(ec * 0.67, 1)
                else:
                    ph = round(7.60 + np.random.normal(0, 0.08), 2)
                    turb = round(11.5 + np.random.normal(0, 0.8), 2)
                    do = round(5.4 + np.random.normal(0, 0.2), 2)
                    ec = round(1380 + np.random.normal(0, 30), 1)
                    tds = round(ec * 0.66, 1)

            elif nid == "N4": # Excellent clean reservoir
                ph = round(7.05 + np.random.normal(0, 0.06), 2)
                turb = round(3.4 + np.random.uniform(-0.3, 0.5), 2)
                do = round(7.4 + np.random.normal(0, 0.15), 2)
                ec = round(610 + np.random.normal(0, 15), 1)
                tds = round(ec * 0.65, 1)

            elif nid == "N5": # Degraded with missing pH values
                # Missing pH on t_idx in [0, 11, 24, 38]
                if t_idx in [0, 11, 24, 38]:
                    ph = ""
                else:
                    ph = round(7.85 + np.random.normal(0, 0.1), 2)
                turb = round(17.5 + np.random.normal(0, 1.2), 2)
                do = round(4.2 + np.random.normal(0, 0.2), 2)
                ec = round(1980 + np.random.normal(0, 40), 1)
                tds = round(ec * 0.66, 1)

            all_rows.append({
                "node_id": nid,
                "timestamp": time_str,
                "latitude": lat,
                "longitude": lon,
                "ph": ph,
                "turbidity": turb,
                "temperature": temp,
                "do": do,
                "ec": ec,
                "tds": tds
            })

    df = pd.DataFrame(all_rows)
    out_file = os.path.join(DATA_DIR, "live_sensor_readings.csv")
    df.to_csv(out_file, index=False)
    print(f"Created {out_file} with {len(df)} rows across {len(nodes)} nodes.")
    return df

if __name__ == "__main__":
    generate_drinking_dataset(500)
    generate_irrigation_dataset(500)
    generate_live_readings()
