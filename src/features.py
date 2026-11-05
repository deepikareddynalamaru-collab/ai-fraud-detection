"""
src/features.py
Feature Engineering Pipeline for Fraud Detection System.
Handles temporal, behavioral, sequential, and user-profile baseline features.
Compatible with both batch DataFrames and single-transaction inputs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


ALL_CATEGORIES = [
    "crypto_investment",
    "electronics",
    "entertainment",
    "food_dining",
    "grocery",
    "healthcare",
    "money_transfer",
    "shopping",
    "travel",
    "utilities",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes behavioral features for fraud detection:
    - amount_zscore, amount_ratio (vs. user average)
    - hour_of_day, is_night (1-4 AM)
    - txn_count_last_1h
    - location_changed
    - is_new_device
    - is_international
    - merchant_category (one-hot encoded)
    """

    def __init__(self):
        self.user_stats = {}
        self.global_mean_amount = 3000.0
        self.global_std_amount = 2000.0
        self.encoder = None
        self.categories_ = sorted(ALL_CATEGORIES)
        self.feature_names = []

    def fit(self, df: pd.DataFrame, y=None):
        """
        Fit user profiles and category encoders from historical transaction data.
        """
        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

        self.global_mean_amount = float(data["amount"].mean())
        self.global_std_amount = float(data["amount"].std()) if data["amount"].std() > 0 else 1000.0

        # Group by user to compute profiles
        user_grouped = data.groupby("user_id")
        
        user_means = user_grouped["amount"].mean().to_dict()
        user_stds = user_grouped["amount"].std().fillna(self.global_std_amount).to_dict()
        user_first_cities = user_grouped["city"].first().to_dict()
        user_first_devices = user_grouped["device_type"].first().to_dict()
        user_all_cities = user_grouped["city"].unique().to_dict()
        user_all_devices = user_grouped["device_type"].unique().to_dict()

        self.user_stats = {}
        for uid in user_means:
            self.user_stats[uid] = {
                "mean_amount": float(user_means[uid]),
                "std_amount": float(user_stds[uid]) if user_stds[uid] > 0 else self.global_std_amount,
                "home_city": user_first_cities.get(uid, "Mumbai"),
                "primary_device": user_first_devices.get(uid, "Android Phone"),
                "known_cities": set(user_all_cities.get(uid, [])),
                "known_devices": set(user_all_devices.get(uid, [])),
            }

        # Fit One-Hot Encoder on merchant categories
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Ensure all defined categories are accounted for
        cat_df = pd.DataFrame({"merchant_category": self.categories_})
        self.encoder.fit(cat_df)

        self.feature_names = [
            "amount",
            "amount_zscore",
            "amount_ratio",
            "hour_of_day",
            "is_night",
            "txn_count_last_1h",
            "location_changed",
            "is_new_device",
            "is_international",
        ] + [f"cat_{c}" for c in self.categories_]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform any input DataFrame into engineered features matrix.
        Preserves original row index and order.
        """
        data = df.copy()
        orig_index = data.index
        orig_len = len(data)

        # Ensure required columns exist with robust fallbacks
        if "timestamp" not in data.columns:
            data["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            data["timestamp"] = data["timestamp"].fillna(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

        if "user_id" not in data.columns:
            data["user_id"] = "USR_UNKNOWN"
        else:
            data["user_id"] = data["user_id"].fillna("USR_UNKNOWN")

        if "amount" not in data.columns:
            data["amount"] = self.global_mean_amount
        else:
            data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(self.global_mean_amount)

        if "city" not in data.columns:
            data["city"] = "Mumbai"
        else:
            data["city"] = data["city"].fillna("Mumbai")

        if "device_type" not in data.columns:
            data["device_type"] = "Android Phone"
        else:
            data["device_type"] = data["device_type"].fillna("Android Phone")

        if "is_international" not in data.columns:
            data["is_international"] = 0
        else:
            data["is_international"] = pd.to_numeric(data["is_international"], errors="coerce").fillna(0).astype(int)

        if "merchant_category" not in data.columns:
            data["merchant_category"] = "shopping"
        else:
            data["merchant_category"] = data["merchant_category"].fillna("shopping")

        # Parse timestamp safely
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # 1. Temporal features
        hour = data["timestamp"].dt.hour
        data["hour_of_day"] = hour
        # Odd hours (1-4 AM) as specified in PRD FR1.3 / FR2
        data["is_night"] = hour.apply(lambda h: 1 if 1 <= h <= 4 else 0).astype(int)

        # 2. User Amount Z-score and Ratio
        z_scores = []
        ratios = []
        for _, row in data.iterrows():
            uid = str(row.get("user_id", ""))
            amt = float(row.get("amount", 0.0))
            if uid in self.user_stats:
                u_mean = self.user_stats[uid]["mean_amount"]
                u_std = self.user_stats[uid]["std_amount"]
            else:
                u_mean = self.global_mean_amount
                u_std = self.global_std_amount

            z = (amt - u_mean) / (u_std + 1e-5)
            ratio = amt / (u_mean + 1e-5)
            z_scores.append(z)
            ratios.append(ratio)

        data["amount_zscore"] = z_scores
        data["amount_ratio"] = ratios

        # 3. Behavioral / Sequential features
        has_txn_count = "txn_count_last_1h" in data.columns and not data["txn_count_last_1h"].isna().all()
        has_loc_chg = "location_changed" in data.columns and not data["location_changed"].isna().all()
        has_new_dev = "is_new_device" in data.columns and not data["is_new_device"].isna().all()

        if orig_len > 1 and "user_id" in data.columns and (not has_txn_count or not has_loc_chg or not has_new_dev):
            data["_order"] = np.arange(orig_len)
            data = data.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

            if not has_txn_count:
                counts = []
                for _, u_group in data.groupby("user_id", sort=False):
                    temp = u_group.set_index("timestamp")
                    c = temp.rolling("1h")["amount"].count().values
                    counts.extend(c)
                data["txn_count_last_1h"] = counts

            if not has_loc_chg:
                loc_flags = []
                prev_city = None
                curr_user = None
                for _, row in data.iterrows():
                    u = row["user_id"]
                    c = row["city"]
                    if u != curr_user:
                        curr_user = u
                        prev_city = c
                        loc_flags.append(0)
                    else:
                        loc_flags.append(1 if c != prev_city else 0)
                        prev_city = c
                data["location_changed"] = loc_flags

            if not has_new_dev:
                dev_flags = []
                curr_user = None
                seen_devices = set()
                for _, row in data.iterrows():
                    u = row["user_id"]
                    d = row["device_type"]
                    if u != curr_user:
                        curr_user = u
                        seen_devices = set(self.user_stats[u]["known_devices"]) if u in self.user_stats else set()
                        if seen_devices:
                            dev_flags.append(1 if d not in seen_devices else 0)
                        else:
                            dev_flags.append(0)
                        seen_devices.add(d)
                    else:
                        dev_flags.append(1 if d not in seen_devices else 0)
                        seen_devices.add(d)
                data["is_new_device"] = dev_flags

            # Restore original order
            data = data.sort_values(by="_order").drop(columns=["_order"]).reset_index(drop=True)

        elif orig_len == 1:
            row = data.iloc[0]
            uid = str(row.get("user_id", ""))
            
            if not has_txn_count:
                data["txn_count_last_1h"] = [int(row.get("txn_count_last_1h", 1))]
                
            if not has_loc_chg:
                if uid in self.user_stats:
                    is_loc_diff = 1 if row["city"] != self.user_stats[uid]["home_city"] else 0
                else:
                    is_loc_diff = 0
                data["location_changed"] = [is_loc_diff]
                
            if not has_new_dev:
                if uid in self.user_stats:
                    is_dev_diff = 1 if row["device_type"] not in self.user_stats[uid]["known_devices"] else 0
                else:
                    is_dev_diff = 0
                data["is_new_device"] = [is_dev_diff]

        # 4. Merchant Category One-Hot Encoding
        cat_series = data["merchant_category"].fillna("other")
        encoded_cats = self.encoder.transform(cat_series.to_frame())
        cat_df = pd.DataFrame(encoded_cats, columns=[f"cat_{c}" for c in self.categories_])

        # 5. Base numerical & binary features
        base_cols = [
            "amount",
            "amount_zscore",
            "amount_ratio",
            "hour_of_day",
            "is_night",
            "txn_count_last_1h",
            "location_changed",
            "is_new_device",
            "is_international",
        ]
        
        feature_matrix = pd.concat([data[base_cols].astype(float), cat_df.astype(float)], axis=1)
        feature_matrix.index = orig_index
        
        return feature_matrix[self.feature_names]
