"""Leakage-aware data preparation, model evaluation, and live matchup prediction."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
STAT_COLS = ["sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds"]
BASE_FEATURES = ["pre_bouts", "pre_win_pct", "last_3_win_pct", "age", "height_inches", "reach_inches", "weight_lbs"] + [f"pre_{x}_avg" for x in STAT_COLS]
WIN_FEATURES = [f"{x}_difference" for x in BASE_FEATURES]
FIGHT_FEATURES = [f"red_{x}" for x in BASE_FEATURES] + [f"blue_{x}" for x in BASE_FEATURES] + [f"{x}_difference" for x in BASE_FEATURES]

QUESTIONS = {
    "win": {"title": "1. Which fighter is more likely to win?", "unit": "fighter-fight", "target": "Fighter win", "features": WIN_FEATURES, "description": "A fighter-perspective classification: does the selected fighter win this bout?"},
    "finish": {"title": "2. Will the bout end before a decision?", "unit": "fight", "target": "Finish", "features": FIGHT_FEATURES, "description": "A fight-level classification: KO/TKO or submission rather than a judges' decision."},
    "ko": {"title": "3. Will the bout end by KO/TKO?", "unit": "fight", "target": "KO/TKO", "features": FIGHT_FEATURES, "description": "A fight-level classification using the recorded UFC method."},
    "submission": {"title": "4. Will the bout end by submission?", "unit": "fight", "target": "Submission", "features": FIGHT_FEATURES, "description": "A fight-level classification using the recorded UFC method."},
    "long": {"title": "5. Will the bout last at least 10 minutes?", "unit": "fight", "target": "10+ minutes", "features": FIGHT_FEATURES, "description": "A fight-level classification using elapsed bout time reconstructed from round and clock."},
}


def clean_name(value):
    if pd.isna(value): return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z\s]", "", value)).strip().upper()


def number(value):
    if pd.isna(value): return np.nan
    return pd.to_numeric(str(value).replace("%", "").replace("lbs.", "").replace('"', "").replace("---", "").strip(), errors="coerce")


def of_value(value, position):
    match = re.match(r"^\s*(\d+)\s+of\s+(\d+)\s*$", str(value))
    return float(match.group(position + 1)) if match else np.nan


def clock_seconds(value):
    match = re.match(r"^\s*(\d+):(\d+)\s*$", str(value))
    return float(int(match.group(1)) * 60 + int(match.group(2))) if match else np.nan


def load_bios():
    bio = pd.read_csv(ROOT / "fighters_raw.csv")
    bio.columns = bio.columns.str.lower().str.strip().str.replace(".", "", regex=False)
    bio["fighter_key"] = bio["name"].map(clean_name)
    parts = bio["height"].astype("string").str.extract(r"(\d+)'(\d+)")
    bio["height_inches"] = pd.to_numeric(parts[0], errors="coerce") * 12 + pd.to_numeric(parts[1], errors="coerce")
    bio["reach_inches"] = bio["reach"].map(number)
    bio["weight_lbs"] = bio["weight"].map(number)
    bio["dob"] = pd.to_datetime(bio["dob"], format="%b%d,%Y", errors="coerce")
    return bio[["fighter_key", "name", "height_inches", "reach_inches", "weight_lbs", "dob", "stance"]].drop_duplicates("fighter_key")


def build_history():
    raw = pd.read_csv(ROOT / "merged_stats_n_scorecards.csv")
    raw.columns = raw.columns.str.lower().str.strip()
    raw["event_date"] = pd.to_datetime(raw["event_date"], dayfirst=True, errors="coerce")
    valid = raw.dropna(subset=["event_date", "red_fighter_name", "blue_fighter_name"]).copy()
    valid = valid.loc[valid.red_fighter_result.isin(["W", "L"]) & valid.blue_fighter_result.isin(["W", "L"])].copy()
    valid["fight_id"] = np.arange(len(valid))

    def corner(name, opponent, label):
        out = pd.DataFrame({"fight_id": valid.fight_id, "event_date": valid.event_date, "fighter": valid[f"{name}_fighter_name"], "opponent": valid[f"{opponent}_fighter_name"], "won_fight": valid[f"{name}_fighter_result"].eq("W").astype(int), "corner": label})
        out["sig_str_landed"] = valid[f"{name}_fighter_sig_str"].map(lambda x: of_value(x, 0))
        out["sig_str_attempted"] = valid[f"{name}_fighter_sig_str"].map(lambda x: of_value(x, 1))
        out["td_landed"] = valid[f"{name}_fighter_td"].map(lambda x: of_value(x, 0))
        out["td_attempted"] = valid[f"{name}_fighter_td"].map(lambda x: of_value(x, 1))
        out["sub_att"] = valid[f"{name}_fighter_sub_att"].map(number)
        out["rev"] = valid[f"{name}_fighter_rev"].map(number)
        out["ctrl_seconds"] = valid[f"{name}_fighter_ctrl"].map(clock_seconds)
        return out

    history = pd.concat([corner("red", "blue", "red"), corner("blue", "red", "blue")], ignore_index=True)
    history["fighter_key"], history["opponent_key"] = history.fighter.map(clean_name), history.opponent.map(clean_name)
    history = history.sort_values(["fighter_key", "event_date", "fight_id"]).reset_index(drop=True)
    grouped = history.groupby("fighter_key", group_keys=False)
    history["pre_bouts"] = grouped.cumcount()
    history["pre_wins"] = grouped.won_fight.transform(lambda x: x.shift().fillna(0).cumsum())
    history["pre_win_pct"] = (history.pre_wins + 1) / (history.pre_bouts + 2)
    history["last_3_win_pct"] = grouped.won_fight.transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(.5)
    for stat in STAT_COLS:
        history[f"pre_{stat}_avg"] = grouped[stat].transform(lambda x: x.shift().expanding(min_periods=1).mean())
    bios = load_bios()
    history = history.merge(bios.drop(columns="name"), on="fighter_key", how="left")
    history["age"] = (history.event_date - history.dob).dt.days / 365.25
    return raw, valid, history, bios


def build_tables(valid, history):
    red = history.loc[history.corner.eq("red"), ["fight_id", "event_date", "fighter", "opponent", "won_fight", *BASE_FEATURES]].copy()
    blue = history.loc[history.corner.eq("blue"), ["fight_id", *BASE_FEATURES]].copy()
    blue = blue.rename(columns={x: f"blue_{x}" for x in BASE_FEATURES})
    fight = red.merge(blue, on="fight_id", how="inner").rename(columns={x: f"red_{x}" for x in BASE_FEATURES})
    for feature in BASE_FEATURES:
        fight[f"{feature}_difference"] = fight[f"red_{feature}"] - fight[f"blue_{feature}"]
    methods = valid.set_index("fight_id").loc[fight.fight_id, "method"].fillna("").astype(str)
    rounds = pd.to_numeric(valid.set_index("fight_id").loc[fight.fight_id, "round"], errors="coerce")
    times = valid.set_index("fight_id").loc[fight.fight_id, "time"].map(clock_seconds)
    fight["finish"] = (~methods.str.contains("DEC", case=False)).astype(int).to_numpy()
    fight["ko"] = methods.str.contains("KO/TKO", case=False, regex=False).astype(int).to_numpy()
    fight["submission"] = methods.str.contains("SUBMISSION", case=False).astype(int).to_numpy()
    fight["elapsed_seconds"] = ((rounds - 1) * 300 + times).to_numpy()
    fight["long"] = fight.elapsed_seconds.ge(600).astype(int)
    # Both sides need history for fair model evaluation. Live prediction still supports sparse fighters.
    fight = fight.loc[fight.red_pre_bouts.gt(0) & fight.blue_pre_bouts.gt(0)].copy()

    fighter = pd.concat([
        fight[["fight_id", "event_date", "fighter", "opponent", "won_fight", *WIN_FEATURES]],
        pd.DataFrame({
            "fight_id": fight.fight_id, "event_date": fight.event_date, "fighter": fight.opponent,
            "opponent": fight.fighter, "won_fight": 1 - fight.won_fight,
            **{x: -fight[x] for x in WIN_FEATURES},
        }),
    ], ignore_index=True)
    return fighter, fight


def chronological_split(table):
    dates = np.sort(table.event_date.dropna().unique())
    cutoff = pd.Timestamp(dates[int(.8 * len(dates))])
    return table.loc[table.event_date < cutoff], table.loc[table.event_date >= cutoff], cutoff


def pipeline(features, kind):
    preprocessor = ColumnTransformer([("numbers", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), features)])
    if kind == "logistic":
        return Pipeline([("prep", preprocessor), ("model", LogisticRegression(C=.4, max_iter=3000, random_state=42))])
    return Pipeline([("prep", preprocessor), ("model", GradientBoostingClassifier(n_estimators=120, learning_rate=.035, max_depth=2, min_samples_leaf=18, subsample=.8, random_state=42))])


def score(y, probability):
    return {"Accuracy": accuracy_score(y, probability >= .5), "ROC AUC": roc_auc_score(y, probability), "Brier score": brier_score_loss(y, probability)}


@lru_cache(maxsize=1)
def load_artifacts():
    raw, valid, history, bios = build_history()
    fighter, fight = build_tables(valid, history)
    _, _, cutoff = chronological_split(fight)
    artifacts, scores = {}, []
    for key, detail in QUESTIONS.items():
        table = fighter.rename(columns={"won_fight": "target"}) if key == "win" else fight.rename(columns={key: "target"})
        train, test = table.loc[table.event_date < cutoff], table.loc[table.event_date >= cutoff]
        baseline = np.repeat(train.target.mean(), len(test))
        models = {"Logistic regression": pipeline(detail["features"], "logistic"), "Gradient boosting": pipeline(detail["features"], "tree")}
        fitted = {}
        for name, model in models.items():
            model.fit(train[detail["features"]], train.target)
            fitted[name] = model
        for name, probability in [("Baseline", baseline)] + [(name, model.predict_proba(test[detail["features"]])[:, 1]) for name, model in fitted.items()]:
            scores.append({"Question": detail["title"], "Approach": name, **score(test.target, probability), "Test rows": len(test), "Positive rate": test.target.mean()})
        # Refit after scoring so the interactive site uses every available historical bout.
        # Held-out results above remain the report's evaluation results.
        for model in fitted.values(): model.fit(table[detail["features"]], table.target)
        artifacts.setdefault("live_models", {})[key] = fitted
    names = pd.concat([bios[["fighter_key", "name"]], history[["fighter_key", "fighter"]].rename(columns={"fighter": "name"})]).dropna().drop_duplicates("fighter_key").sort_values("name")
    missing = pd.DataFrame({"Field": raw.columns, "Missing %": raw.isna().mean().mul(100).round(1)}).sort_values("Missing %", ascending=False)
    diagnostics = {
        "Raw fight rows": len(raw), "Completed W/L fights retained": len(valid), "Fighter-fight rows": len(history),
        "Unique fighters": history.fighter_key.nunique(), "Date range": f"{history.event_date.min():%b %Y} - {history.event_date.max():%b %Y}",
        "Rows excluded before modeling": len(raw) - len(valid), "Chronological cutoff": cutoff,
    }
    return {"raw": raw, "valid": valid, "history": history, "bios": bios, "fighter": fighter, "fight": fight, "scores": pd.DataFrame(scores), "live_models": artifacts["live_models"], "names": names, "missing": missing, "diagnostics": diagnostics, "cutoff": cutoff}


def profile_at(artifacts, fighter_key, when):
    history, bios = artifacts["history"], artifacts["bios"]
    earlier = history.loc[(history.fighter_key == fighter_key) & (history.event_date < when)].sort_values(["event_date", "fight_id"])
    profile = {feature: np.nan for feature in BASE_FEATURES}
    profile["pre_bouts"] = len(earlier)
    profile["pre_win_pct"] = ((earlier.won_fight.sum() + 1) / (len(earlier) + 2)) if len(earlier) else .5
    profile["last_3_win_pct"] = earlier.won_fight.tail(3).mean() if len(earlier) else .5
    for stat in STAT_COLS: profile[f"pre_{stat}_avg"] = earlier[stat].mean() if len(earlier) else np.nan
    bio = bios.loc[bios.fighter_key.eq(fighter_key)].head(1)
    if len(bio):
        for x in ["height_inches", "reach_inches", "weight_lbs"]: profile[x] = bio.iloc[0][x]
        profile["age"] = (pd.Timestamp(when) - bio.iloc[0].dob).days / 365.25 if pd.notna(bio.iloc[0].dob) else np.nan
    return profile


def predict_matchup(fighter_a, fighter_b, event_date):
    """Return all five live probabilities for the selected fighters and date."""
    artifacts = load_artifacts(); names = artifacts["names"]; when = pd.Timestamp(event_date)
    key_a = names.loc[names.name.eq(fighter_a), "fighter_key"].iloc[0]; key_b = names.loc[names.name.eq(fighter_b), "fighter_key"].iloc[0]
    a, b = profile_at(artifacts, key_a, when), profile_at(artifacts, key_b, when)
    forward = pd.DataFrame([{f"{x}_difference": a[x] - b[x] for x in BASE_FEATURES}])
    reverse = -forward
    winner_probabilities = {}
    for name, model in artifacts["live_models"]["win"].items():
        pa = model.predict_proba(forward[WIN_FEATURES])[:, 1][0]; pb = model.predict_proba(reverse[WIN_FEATURES])[:, 1][0]
        winner_probabilities[name] = float(pa / (pa + pb))
    bout_row = {}
    for feature in BASE_FEATURES:
        bout_row[f"red_{feature}"] = a[feature]
        bout_row[f"blue_{feature}"] = b[feature]
        bout_row[f"{feature}_difference"] = a[feature] - b[feature]
    bout_features = pd.DataFrame([bout_row])
    bout_probabilities = {}
    for key in ["finish", "ko", "submission", "long"]:
        bout_probabilities[key] = {
            name: float(model.predict_proba(bout_features[QUESTIONS[key]["features"]])[:, 1][0])
            for name, model in artifacts["live_models"][key].items()
        }
    return {"a": a, "b": b, "winner": winner_probabilities, "bout": bout_probabilities}


def predict_winner(fighter_a, fighter_b, event_date):
    """Backwards-compatible access to only the primary winner question."""
    output = predict_matchup(fighter_a, fighter_b, event_date)
    return {"a": output["a"], "b": output["b"], "probabilities": output["winner"]}
