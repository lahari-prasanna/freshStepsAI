from flask import Flask, render_template, request, redirect, jsonify, url_for
from flask_cors import CORS
import pandas as pd
import os, uuid, datetime as dt, difflib, csv, re

app = Flask(__name__)
CORS(app)

# ---------------- PATHS ----------------
DATASETS_DIR = "datasets"
SUBMISSIONS_FILE = "submissions.csv"

# ---------------- Ensure submissions.csv exists with consistent columns (include coins) ----------------
# If the CSV does not exist, create it with a stable header that includes "coins" so we can persist coin updates.
if not os.path.exists(SUBMISSIONS_FILE):
    pd.DataFrame(columns=[
        "submission_id","timestamp","fullName","age",
        "city_input","city_matched",
        "from_location","to_location",
        "indoor","outdoor","work",
        "conditions","other","notes",
        "latitude","longitude",
        "pm2_5_latest","pm2_5_avg",
        "pm10_latest","pm10_avg",
        "co2_latest","co2_avg",
        "AT_latest","RH_latest",
        "station_name","state",
        "suggestion_pm25","suggestion_pm10","suggestion_co2",
        "raw_user",
        "coins"
    ]).to_csv(SUBMISSIONS_FILE, index=False)

# ---------------- DATA ----------------
def load_all_datasets():
    frames = []
    if not os.path.isdir(DATASETS_DIR):
        return pd.DataFrame()

    for f in os.listdir(DATASETS_DIR):
        if f.lower().endswith(".csv"):
            path = os.path.join(DATASETS_DIR, f)
            try:
                df = pd.read_csv(path, on_bad_lines="skip")
            except Exception as e:
                print("Failed to read", path, e)
                continue
            df.columns = [c.strip() for c in df.columns]
            if "local_time" in df.columns:
                df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
            if "city" in df.columns:
                df["city_clean"] = df["city"].astype(str).str.strip().str.lower()
            if "station_name" in df.columns:
                df["station_name_clean"] = df["station_name"].astype(str).str.strip().str.lower()
            df["_source_file"] = f
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

DATA = load_all_datasets()
print("Loaded dataset rows:", len(DATA))

# ---------------- AQI ----------------
def calculate_aqi(pm25, pm10):
    # PM2.5 breakpoints
    pm25_bp = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,350,401,500)]
    # PM10 breakpoints
    pm10_bp = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,500,401,500)]

    def sub_aqi(value, breakpoints):
        for C_low,C_high,I_low,I_high in breakpoints:
            if C_low <= value <= C_high:
                return round(((I_high-I_low)/(C_high-C_low))*(value-C_low)+I_low)
        return None

    aqi_pm25 = sub_aqi(pm25, pm25_bp)
    aqi_pm10 = sub_aqi(pm10, pm10_bp)
    if aqi_pm25 is None and aqi_pm10 is None: return None
    if aqi_pm25 is None: return aqi_pm10
    if aqi_pm10 is None: return aqi_pm25
    return max(aqi_pm25, aqi_pm10)

# ---------------- HELPERS ----------------

def val_or_none(row, key):
    return row[key] if key in row and pd.notna(row[key]) else None

def find_best_city_match(city_input: str):
    if not city_input or "city_clean" not in DATA.columns:
        return None
    key = str(city_input).strip().lower()
    cities = DATA["city_clean"].dropna().unique().tolist()
    if key in cities:
        return key
    match = difflib.get_close_matches(key, cities, n=1, cutoff=0.6)
    if match: return match[0]
    for c in cities:
        if key in c: return c
    return None

def pm25_health_message(v):
    try: v = float(v)
    except: return "PM2.5 data unavailable."
    if v <= 15:  return "Excellent air (WHO 24h guideline)."
    if v <= 35:  return "Good — low risk for most people."
    if v <= 55:  return "Moderate — sensitive groups limit outdoor exertion."
    if v <= 150: return "Unhealthy — wear a mask outdoors."
    return "Very unhealthy — avoid outdoor exposure & use purifier."

def pm10_health_message(v):
    try: v = float(v)
    except: return "PM10 data unavailable."
    if v <= 45:   return "Excellent (WHO 24h guideline)."
    if v <= 100:  return "Moderate — sensitive groups take care."
    if v <= 250:  return "Unhealthy — consider limiting outdoor activity."
    return "Very unhealthy — stay indoors if possible."

def co2_message(v):
    try: v = float(v)
    except: return "CO₂ data unavailable."
    if v < 600:   return "Very good ventilation."
    if v < 1000:  return "Acceptable indoor levels."
    if v < 1500:  return "Poor ventilation — open windows."
    return "High CO₂ — ventilate immediately."

def get_exposure(pm25, co2):
    if pm25 is None or co2 is None:
        return "No data"

    if pm25 <= 50 and co2 <= 600:
        exposure_level = "Low"
    elif pm25 <= 100 and co2 <= 1000:
        exposure_level = "Medium"
    else:
        exposure_level = "High"

    return exposure_level

def get_health_advice(exposure_level):
    advice = {
        "Low": "Your exposure today is low. Maintain your routine and stay safe!",
        "Medium": "Your exposure today is medium. Wash hands frequently and avoid crowded places.",
        "High": "Your exposure today is high. Wear a mask, sanitize regularly, and minimize outdoor activity."
    }
    return advice.get(exposure_level, "No advice available.")

# ---------------- In-memory submissions (and persistence) ----------------
submissions = {}

# Load existing submissions into memory on startup so submission_id and coins survive restarts
if os.path.exists(SUBMISSIONS_FILE):
    try:
        df_existing = pd.read_csv(SUBMISSIONS_FILE, on_bad_lines="skip")
        # normalize column names
        df_existing.columns = [c.strip() for c in df_existing.columns]
        for _, r in df_existing.iterrows():
            sid = r.get("submission_id")
            if pd.notna(sid):
                # convert row to normal dict, convert NaN to None
                record = {}
                for k, v in r.to_dict().items():
                    if pd.isna(v):
                        record[k] = None
                    else:
                        record[k] = v
                # ensure coins present and numeric
                coins_val = record.get("coins", 0)
                try:
                    record["coins"] = int(float(coins_val)) if coins_val is not None else 0
                except:
                    record["coins"] = 0
                submissions[str(sid)] = record
    except Exception as e:
        print("Warning: failed to load existing submissions into memory:", e)

def _save_all_submissions_to_csv():
    """
    Persist the in-memory submissions dict back to SUBMISSIONS_FILE.
    This function writes all current submission records to disk so updates (like coin changes)
    survive server restarts.
    """
    if not submissions:
        # write empty file with header to keep format
        pd.DataFrame(columns=[
            "submission_id","timestamp","fullName","age",
            "city_input","city_matched",
            "from_location","to_location",
            "indoor","outdoor","work",
            "conditions","other","notes",
            "latitude","longitude",
            "pm2_5_latest","pm2_5_avg",
            "pm10_latest","pm10_avg",
            "co2_latest","co2_avg",
            "AT_latest","RH_latest",
            "station_name","state",
            "suggestion_pm25","suggestion_pm10","suggestion_co2",
            "raw_user",
            "coins"
        ]).to_csv(SUBMISSIONS_FILE, index=False)
        return

    # Determine superset of keys across all submissions for CSV columns
    all_keys = set()
    for rec in submissions.values():
        all_keys.update(rec.keys())
    # make sure coins is present
    if "coins" not in all_keys:
        all_keys.add("coins")
    fieldnames = list(all_keys)
    try:
        with open(SUBMISSIONS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in submissions.values():
                # ensure the writer gets plain python types (no numpy types)
                row = {}
                for k in fieldnames:
                    v = rec.get(k)
                    if v is None:
                        row[k] = ""
                    else:
                        row[k] = v
                writer.writerow(row)
    except Exception as e:
        print("Error saving submissions to CSV:", e)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/submit_form", methods=["POST"])
def submit_form():
    # Accept JSON or form-encoded
    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict(flat=True)

    fullName = data.get("fullName")
    age = data.get("age")
    city_input = data.get("city")
    from_location = data.get("from_location") or ""
    to_location   = data.get("to_location") or ""
    indoor = float(data.get("indoor") or 0)
    outdoor = float(data.get("outdoor") or 0)
    work = float(data.get("work") or max(0, 24 - (indoor + outdoor)))
    conditions = data.get("condition") or ""
    other = data.get("other") or ""
    notes = data.get("notes") or ""
    latitude = data.get("latitude") or ""
    longitude = data.get("longitude") or ""

    # match city and select latest
    city_key = find_best_city_match(city_input)
    if not city_key:
        return jsonify({"error": "City not found in dataset"}), 404

    city_df = DATA[DATA["city_clean"] == city_key].copy().sort_values("local_time")
    if city_df.empty:
        return jsonify({"error": "No rows for the selected city"}), 404

    # ------------------- Filter by route (from_location → to_location) -------------------
    from_location = str(from_location).strip().lower()
    to_location   = str(to_location).strip().lower()

    route_df = city_df.copy()

    if from_location and to_location and "station_name_clean" in city_df.columns:
        stations = city_df["station_name_clean"].unique().tolist()

        # fuzzy match for both stations
        from_match = difflib.get_close_matches(from_location, stations, n=1, cutoff=0.6)
        to_match   = difflib.get_close_matches(to_location, stations, n=1, cutoff=0.6)

        if from_match and to_match:
            # filter only rows belonging to those stations
            route_df = city_df[
                city_df["station_name_clean"].isin([from_match[0], to_match[0]])
            ].copy()

    # ------------------- Now compute metrics -------------------
    if not route_df.empty:
        pm2_5_latest = val_or_none(route_df.iloc[-1], "PM2_5")
        pm10_latest  = val_or_none(route_df.iloc[-1], "PM10")
        co2_latest   = val_or_none(route_df.iloc[-1], "CO2")
        at_latest    = val_or_none(route_df.iloc[-1], "AT")
        rh_latest    = val_or_none(route_df.iloc[-1], "RH")

        pm2_5_avg = round(route_df["PM2_5"].mean(skipna=True), 2) if not route_df.empty else None
        pm10_avg  = round(route_df["PM10"].mean(skipna=True), 2) if not route_df.empty else None
        co2_avg   = round(route_df["CO2"].mean(skipna=True), 2) if not route_df.empty else None

    else:
        pm2_5_latest = pm10_latest = co2_latest = at_latest = rh_latest = None
        pm2_5_avg = pm10_avg = co2_avg = None

    submission_id = str(uuid.uuid4())
    timestamp = dt.datetime.now().isoformat()

    row = {
        "submission_id": submission_id,
        "timestamp": timestamp,
        "fullName": fullName,
        "age": age,
        "city_input": city_input,
        "city_matched": city_key,
        "from_location": from_location,
        "to_location": to_location,
        "indoor": indoor,
        "outdoor": outdoor,
        "work": work,
        "conditions": conditions,
        "other": other,
        "notes": notes,
        "latitude": latitude,
        "longitude": longitude,
        "pm2_5_latest": pm2_5_latest,
        "pm2_5_avg": pm2_5_avg,
        "pm10_latest": pm10_latest,
        "pm10_avg": pm10_avg,
        "co2_latest": co2_latest,
        "co2_avg": co2_avg,
        "AT_latest": at_latest,
        "RH_latest": rh_latest,
        "station_name": route_df["station_name"].iloc[-1] if not route_df.empty else None,
        "state": route_df["state"].iloc[-1] if not route_df.empty else None,
        "suggestion_pm25": pm25_health_message(pm2_5_latest),
        "suggestion_pm10": pm10_health_message(pm10_latest),
        "suggestion_co2": co2_message(co2_latest),
        "raw_user": str(data),
        "coins": 0
    }

    # append to CSV
    file_exists = os.path.isfile(SUBMISSIONS_FILE)
    with open(SUBMISSIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # save in-memory for dashboard
    submissions[submission_id] = row

    return redirect(url_for("dashboard", submission_id=submission_id))

@app.route("/dashboard/<submission_id>")
def dashboard(submission_id):
    if submission_id not in submissions:
        return "Invalid submission ID", 404

    result = submissions[submission_id]
    pm25 = float(result.get("pm2_5_latest") or 0)
    pm10 = float(result.get("pm10_latest") or 0)
    aqi = calculate_aqi(pm25, pm10)
    result["AQI"] = aqi

    today = dt.date.today()

    history = pd.read_csv(SUBMISSIONS_FILE, on_bad_lines="skip")
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    history["fullName_lower"] = history["fullName"].astype(str).str.strip().str.lower()

    user_name = str(result.get("fullName")).strip().lower()
    user_hist = history[history["fullName_lower"] == user_name]

    # --- Today / Yesterday / Day before ---
    today_rows      = user_hist[user_hist["timestamp"].dt.date == today]
    yesterday_rows  = user_hist[user_hist["timestamp"].dt.date == today - dt.timedelta(days=1)]
    daybefore_rows  = user_hist[user_hist["timestamp"].dt.date == today - dt.timedelta(days=2)]

    today_pm  = result.get("pm2_5_latest")
    today_co2 = result.get("co2_latest")
    yest_pm   = yesterday_rows["pm2_5_latest"].iloc[-1] if not yesterday_rows.empty else None
    yest_co2  = yesterday_rows["co2_latest"].iloc[-1]    if not yesterday_rows.empty else None
    daybefore_pm  = daybefore_rows["pm2_5_latest"].iloc[-1] if not daybefore_rows.empty else None
    daybefore_co2 = daybefore_rows["co2_latest"].iloc[-1]    if not daybefore_rows.empty else None

    # --- Exposure levels (only if data exists) ---
    today_exp      = get_exposure(today_pm, today_co2) if today_pm and today_co2 else "No data"
    yesterday_exp  = get_exposure(yest_pm, yest_co2)   if yest_pm and yest_co2 else "No data"
    daybefore_exp  = get_exposure(daybefore_pm, daybefore_co2) if daybefore_pm and daybefore_co2 else "No data"

    advice_to_show = get_health_advice(today_exp)

    # prepare trend (last 48 rows for city)
    city = str(result.get("city_matched") or "").strip().lower()
    if city and "city_clean" in DATA.columns:
        trend_df = DATA[DATA["city_clean"] == city].dropna(subset=["local_time"]).sort_values("local_time").tail(48)
    else:
        trend_df = pd.DataFrame()

    times   = trend_df["local_time"].dt.strftime("%d %b %H:%M").tolist() if not trend_df.empty else []
    pm25_ts = trend_df["PM2_5"].tolist() if "PM2_5" in trend_df else []
    pm10_ts = trend_df["PM10"].tolist()  if "PM10" in trend_df else []
    co2_ts  = trend_df["CO2"].tolist()   if "CO2"  in trend_df else []

    # weekly: list of dicts containing PM2_5, PM10, CO2 per day (last 7 days)
    weekly = []
    if not trend_df.empty:
        tmp = trend_df.copy()
        tmp["day"] = tmp["local_time"].dt.date
        g = tmp.groupby("day").agg({"PM2_5":"mean","PM10":"mean","CO2":"mean"}).tail(7).reset_index()
        for _, r in g.iterrows():
            weekly.append({
                "day": str(r["day"]),
                "PM2_5": round(r["PM2_5"],1) if pd.notna(r["PM2_5"]) else None,
                "PM10":  round(r["PM10"],1)  if pd.notna(r["PM10"])  else None,
                "CO2":   round(r["CO2"],1)   if pd.notna(r["CO2"])   else None
            })

    # today vs yesterday for this user (from submissions CSV history)
    try:
        history = pd.read_csv(SUBMISSIONS_FILE, on_bad_lines="skip")
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history["fullName_lower"] = history["fullName"].astype(str).str.strip().str.lower()
        user_name = str(result.get("fullName")).strip().lower()
        user_hist = history[history["fullName_lower"] == user_name]

        today = dt.date.today()
        yesterday = today - dt.timedelta(days=1)

        today_rows = user_hist[user_hist["timestamp"].dt.date == today]
        yest_rows  = user_hist[user_hist["timestamp"].dt.date == yesterday]

        today_pm = result.get("pm2_5_latest")
        today_co2 = result.get("co2_latest")
        yest_pm = yest_rows["pm2_5_latest"].mean() if not yest_rows.empty else None
        yest_co2 = yest_rows["co2_latest"].mean() if not yest_rows.empty else None

    except Exception as e:
        print("History error:", e)
        today_pm = yest_pm = today_co2 = yest_co2 = None

    # exposure breakdown (use submitted values)
    try:
        exposure = {
            "indoor": float(result.get("indoor") or 0),
            "outdoor": float(result.get("outdoor") or 0),
            "work": float(result.get("work") or 0)
        }
    except:
        exposure = {"indoor":0,"outdoor":0,"work":0}

    exp_map = {"Low": 1, "Medium": 2, "High": 3, "No data": 0}

    bar_labels = ["Day Before", "Yesterday", "Today"]
    bar_data = [
        exp_map.get(daybefore_exp, 0),
        exp_map.get(yesterday_exp, 0),
        exp_map.get(today_exp, 0)
    ]

    return render_template("dashboard.html",
                           result=result,
                           submission_id=submission_id,
                           times=times, pm25_ts=pm25_ts, pm10_ts=pm10_ts, co2_ts=co2_ts,
                           weekly=weekly,
                           exposure=exposure,
                           bar_labels=bar_labels,
                           bar_data=bar_data,
                           today=today,
                           advice_to_show=advice_to_show,
                           today_exp =today_exp,
                           yesterday_exp =yesterday_exp,
                           daybefore_exp = daybefore_exp,
                           yesterday=yesterday,
                           user_name=user_name,
                           user_hist=user_hist,
                           today_pm=today_pm, yest_pm=yest_pm, today_co2=today_co2, yest_co2=yest_co2,
                           coins=result.get("coins", 0)
                        )

@app.route("/products/<submission_id>")
def products(submission_id):
    if submission_id not in submissions:
        return "Invalid submission ID", 404
    info = submissions[submission_id]

    # Inputs
    indoor_h = float(info.get("indoor") or 0)
    outdoor_h = float(info.get("outdoor") or 0)
    pm25 = float(info.get("pm2_5_latest") or info.get("pm2_5_avg") or 0)
    pm10 = float(info.get("pm10_latest") or info.get("pm10_avg") or 0)
    co2 = float(info.get("co2_latest") or info.get("co2_avg") or 0)

    recs = []

    # 1. Indoor exposure > 8 hrs OR PM2.5 high
    if indoor_h >= 8 or pm25 > 35:
        recs.append({
            "name": "HEPA Air Purifier (CADR 250+)",
            "why": "WHO: Long indoor hours + elevated PM2.5",
            "est_price": "₹8,000–₹15,000",
            "tag": "Indoor",
            "image": "images/products/hepa.png"
        })

    # 2. Outdoor exposure > 2 hrs OR PM2.5 > 35
    if pm25 > 35 or outdoor_h >= 2:
        recs.append({
            "name": "N95 / KN95 Mask (pack of 5)",
            "why": "WHO: High PM2.5 outdoors — wear respirators",
            "est_price": "₹400–₹1,200",
            "tag": "Mask",
            "image": "images/products/N95.png"
        })
        recs.append({
            "name": "Surgical/Disposable Masks (pack)",
            "why": "WHO: Affordable daily protection",
            "est_price": "₹150–₹400",
            "tag": "Mask",
            "image": "images/products/surgical.png"
        })

    # 3. High PM10 (dust-prone areas)
    if pm10 > 100:
        recs.append({
            "name": "Workplace Respirator / P100",
            "why": "WHO: PM10 exceeds 100 µg/m³ — strong filtration needed",
            "est_price": "₹2,000–₹4,000",
            "tag": "Work",
            "image": "images/products/respirator.png"
        })

    # 4. Poor indoor ventilation (CO2 > 1000 ppm) - commented out in original
    # if co2 > 1000:
    #     recs.append({...})

    # 5. Default (low-risk days)
    if not recs:
        recs.append({
            "name": "Carbon Pre-Filter Pack",
            "why": "WHO: Maintain healthy PM2.5 levels",
            "est_price": "₹300–₹800",
            "tag": "General",
            "image": "images/products/carbon.png"
        })

    # Save computed recommendations back into in-memory submissions (so template and later calls can access)
    submissions[submission_id]["recommendations"] = recs
    # persist submissions so the recommendations and coins are guaranteed saved
    _save_all_submissions_to_csv()

    # Render the original products.html (we didn't remove or alter features - only added persistence & storage)
    return render_template(
        "products.html",
        recommendations=recs,
        info=submissions[submission_id],
        submission_id=submission_id
    )


@app.route("/routes/<submission_id>")
def routes(submission_id):
    if submission_id not in submissions:
        return "Invalid submission ID", 404
    info = submissions[submission_id]
    # We'll show mock routes here (front-end draws them)
    return render_template("routes.html", result=info, submission_id=submission_id)


@app.route("/exposure_summary/<submission_id>")
def exposure_summary(submission_id):
    info = submissions.get(submission_id)
    if not info:
        return "Invalid submission ID", 404

    # Note: exposure_level and exposure_recommendations are used here as in your original file.
    # If those functions are defined elsewhere in your project, they will be used. We are keeping your original calls.
    level = exposure_level(
        pm25=float(info.get("pm2_5_latest") or 0),
        pm10=float(info.get("pm10_latest") or 0),
        co2=float(info.get("co2_latest") or 0),
        indoor_h=float(info.get("indoor") or 0),
        outdoor_h=float(info.get("outdoor") or 0),
        work_h=float(info.get("work") or 0)
    )

    suggestions = exposure_recommendations(level)

    return render_template("exposure_popup.html",
                           exposure_level=level,
                           suggestions=suggestions)


@app.route("/add_coins/<submission_id>", methods=["POST"])
def add_coins(submission_id):
    if submission_id not in submissions:
        return jsonify({"error": "Invalid submission ID"}), 404

    submissions[submission_id]["coins"] = submissions[submission_id].get("coins", 0) + 10
    # persist coins change immediately
    _save_all_submissions_to_csv()
    return jsonify({"coins": submissions[submission_id]["coins"]})


@app.route("/buy_with_coins/<submission_id>", methods=["POST"])
def buy_with_coins(submission_id):
    if submission_id not in submissions:
        return jsonify({"error": "Invalid submission ID"}), 404

    # Accept JSON body
    data = request.get_json(silent=True) or {}
    price_raw = data.get("price", 0)

    # parse price robustly: accept int/float or strings like "₹8,000–₹15,000" -> take first number (8000)
    def parse_price_to_int(p):
        try:
            if isinstance(p, (int, float)):
                return int(p)
            s = str(p)
            # find first number (could contain commas)
            m = re.search(r'(\d[\d,\,]*)', s)
            if m:
                num = m.group(1).replace(",", "")
                return int(num)
            return 0
        except Exception:
            return 0

    price_val = parse_price_to_int(price_raw)
    if price_val <= 0:
        return jsonify({"error": "Invalid product price sent."}), 400

    # Only 80% of product price can be covered by coins
    max_cover = int(round(price_val * 0.8))

    current_coins = submissions[submission_id].get("coins", 0) or 0
    if int(current_coins) < max_cover:
        return jsonify({"error": f"Not enough coins. Need {max_cover}, you have {current_coins}."}), 400

    # Deduct coins (use integer arithmetic)
    submissions[submission_id]["coins"] = int(current_coins) - max_cover
    # persist change
    _save_all_submissions_to_csv()

    return jsonify({
        "message": f"Purchase successful! Used {max_cover} coins.",
        "remaining_coins": submissions[submission_id]["coins"]
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
