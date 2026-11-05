"""
data/generate_data.py
Synthetic Transaction Generator for AI-Powered Fraud Detection System.
Generates ~20,000 transactions with ~2% realistic fraud patterns.
Reproducible with fixed random seed (42).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configuration
RANDOM_SEED = 42
TOTAL_TRANSACTIONS = 20000
FRAUD_RATIO = 0.02  # ~2% fraud
NUM_USERS = 800

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", 
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat"
]

INTERNATIONAL_CITIES = [
    "Dubai", "Singapore", "London", "New York", "Bangkok"
]

CATEGORIES = [
    "grocery", "food_dining", "shopping", "electronics", 
    "entertainment", "travel", "utilities", "healthcare", 
    "money_transfer", "crypto_investment"
]

DEVICE_TYPES = [
    "Android Phone", "iPhone", "Windows Laptop", "MacBook", "iPad"
]


def generate_user_profiles(num_users=NUM_USERS):
    """
    Generate realistic user profiles with characteristic spending habits,
    home cities, and primary devices.
    """
    profiles = {}
    for i in range(1, num_users + 1):
        uid = f"USR_{i:04d}"
        home_city = random.choice(CITIES)
        primary_device = random.choice(DEVICE_TYPES)
        # Average ticket size between Rs. 400 and Rs. 6000
        avg_spend = round(random.uniform(400, 6000), 2)
        std_spend = round(avg_spend * random.uniform(0.2, 0.5), 2)
        profiles[uid] = {
            "home_city": home_city,
            "primary_device": primary_device,
            "avg_spend": avg_spend,
            "std_spend": std_spend
        }
    return profiles


def generate_transactions(num_txns=TOTAL_TRANSACTIONS, fraud_ratio=FRAUD_RATIO):
    """
    Generate synthetic transactions with injected fraud patterns.
    """
    profiles = generate_user_profiles()
    user_ids = list(profiles.keys())
    
    num_fraud = int(num_txns * fraud_ratio)
    num_normal = num_txns - num_fraud
    
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    records = []
    
    # 1. Generate Normal Transactions
    print(f"Generating {num_normal} normal transactions...")
    for i in range(num_normal):
        uid = random.choice(user_ids)
        prof = profiles[uid]
        
        # Normal amounts follow user's normal distribution
        amount = max(50.0, np.random.normal(prof["avg_spend"], prof["std_spend"]))
        # Round to 2 decimals
        amount = round(float(amount), 2)
        
        # Normal hours: heavily weighted towards 08:00 - 23:00
        # Occasional 00:00 - 01:00 or 06:00 - 08:00
        hour_prob = np.array([
            0.005, 0.002, 0.001, 0.001, 0.002, 0.005,  # 00-05 AM (rare)
            0.02, 0.04, 0.06, 0.07, 0.08, 0.08,        # 06-11 AM
            0.08, 0.07, 0.06, 0.07, 0.08, 0.09,        # 12-17 PM
            0.09, 0.08, 0.06, 0.04, 0.02, 0.01         # 18-23 PM
        ])
        hour_prob = hour_prob / hour_prob.sum()
        hour = np.random.choice(24, p=hour_prob)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        day_offset = random.randint(0, 45)
        txn_time = start_time + timedelta(days=day_offset, hours=int(hour), minutes=minute, seconds=second)
        
        # Normal device: 95% primary device, 5% secondary
        if random.random() < 0.95:
            device = prof["primary_device"]
        else:
            alt_devices = [d for d in DEVICE_TYPES if d != prof["primary_device"]]
            device = random.choice(alt_devices)
            
        # Normal location: 96% home city, 4% domestic travel
        if random.random() < 0.96:
            city = prof["home_city"]
        else:
            city = random.choice([c for c in CITIES if c != prof["home_city"]])
            
        # Normal international: 1%
        is_international = 1 if random.random() < 0.01 else 0
        if is_international:
            city = random.choice(INTERNATIONAL_CITIES)
            
        # Category
        # Normal transactions rarely use crypto_investment
        cat_weights = [0.22, 0.20, 0.18, 0.10, 0.08, 0.07, 0.08, 0.05, 0.019, 0.001]
        category = random.choices(CATEGORIES, weights=cat_weights, k=1)[0]
        
        records.append({
            "transaction_id": f"TXN_{len(records)+1:06d}",
            "user_id": uid,
            "timestamp": txn_time,
            "amount": amount,
            "merchant_category": category,
            "city": city,
            "device_type": device,
            "is_international": is_international,
            "is_fraud": 0,
            "fraud_type": "None"
        })
        
    # 2. Generate Fraud Transactions with Specific Injected Patterns
    print(f"Injecting {num_fraud} fraudulent transactions across 6 attack patterns...")
    
    fraud_patterns = [
        "high_amount",
        "odd_hours",
        "sudden_city_change",
        "burst_frequency",
        "new_device_compromise",
        "international_high_value"
    ]
    
    for i in range(num_fraud):
        uid = random.choice(user_ids)
        prof = profiles[uid]
        day_offset = random.randint(5, 40)
        pattern = random.choice(fraud_patterns)
        
        # Base defaults
        txn_hour = random.randint(10, 20)
        txn_min = random.randint(0, 59)
        city = prof["home_city"]
        device = prof["primary_device"]
        is_international = 0
        amount = round(prof["avg_spend"] * random.uniform(1.2, 2.5), 2)
        category = random.choice(["shopping", "electronics", "money_transfer", "crypto_investment"])
        
        if pattern == "high_amount":
            # 8x to 25x normal average spending
            amount = round(prof["avg_spend"] * random.uniform(8.0, 22.0) + random.uniform(20000, 60000), 2)
            category = random.choice(["electronics", "crypto_investment", "money_transfer", "shopping"])
            # Often late night or unfamiliar device
            if random.random() < 0.5:
                txn_hour = random.choice([1, 2, 3, 4])
                
        elif pattern == "odd_hours":
            # Deep night: 1 AM to 4 AM
            txn_hour = random.choice([1, 2, 3, 4])
            amount = round(prof["avg_spend"] * random.uniform(4.0, 10.0) + random.uniform(8000, 25000), 2)
            category = random.choice(["crypto_investment", "money_transfer", "electronics"])
            
        elif pattern == "sudden_city_change":
            # Sudden shift to distant city or foreign hub
            city = random.choice([c for c in CITIES if c != prof["home_city"]])
            amount = round(prof["avg_spend"] * random.uniform(3.5, 9.0) + random.uniform(10000, 30000), 2)
            alt_devices = [d for d in DEVICE_TYPES if d != prof["primary_device"]]
            device = random.choice(alt_devices)
            
        elif pattern == "burst_frequency":
            # Part of a rapid burst of transactions
            txn_hour = random.choice([2, 3, 14, 22])
            txn_min = random.randint(5, 45)
            amount = round(prof["avg_spend"] * random.uniform(3.0, 8.0) + random.uniform(7000, 20000), 2)
            category = random.choice(["money_transfer", "crypto_investment", "shopping"])
            
        elif pattern == "new_device_compromise":
            # Stolen credentials used on a new unfamiliar device
            alt_devices = [d for d in DEVICE_TYPES if d != prof["primary_device"]]
            device = random.choice(alt_devices)
            amount = round(prof["avg_spend"] * random.uniform(5.0, 12.0) + random.uniform(15000, 40000), 2)
            if random.random() < 0.6:
                txn_hour = random.choice([1, 2, 3, 4])
            category = random.choice(["electronics", "crypto_investment", "money_transfer"])
            
        elif pattern == "international_high_value":
            # International transaction with massive sum
            is_international = 1
            city = random.choice(INTERNATIONAL_CITIES)
            amount = round(prof["avg_spend"] * random.uniform(7.0, 18.0) + random.uniform(25000, 75000), 2)
            alt_devices = [d for d in DEVICE_TYPES if d != prof["primary_device"]]
            device = random.choice(alt_devices)
            category = random.choice(["shopping", "electronics", "crypto_investment", "money_transfer"])

        txn_time = start_time + timedelta(days=day_offset, hours=txn_hour, minutes=txn_min, seconds=random.randint(0, 59))
        
        records.append({
            "transaction_id": f"TXN_{len(records)+1:06d}",
            "user_id": uid,
            "timestamp": txn_time,
            "amount": amount,
            "merchant_category": category,
            "city": city,
            "device_type": device,
            "is_international": is_international,
            "is_fraud": 1,
            "fraud_type": pattern
        })
        
    df = pd.DataFrame(records)
    
    # Sort by user_id and timestamp so temporal and behavioral sequence is coherent
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    
    # Inject burst frequency sequence explicitly:
    # For burst_frequency fraud transactions, create 2-3 accompanying rapid transactions right before/after
    burst_indices = df[df["fraud_type"] == "burst_frequency"].index.tolist()
    # For a subset of burst frauds, inject rapid preceding transactions within 15-30 mins
    extra_burst_txns = []
    for idx in burst_indices[:30]:
        base_txn = df.loc[idx]
        for burst_step in range(1, 4):
            burst_time = base_txn["timestamp"] - timedelta(minutes=burst_step * 8, seconds=random.randint(5, 45))
            extra_burst_txns.append({
                "transaction_id": f"TXN_{len(df) + len(extra_burst_txns) + 1:06d}",
                "user_id": base_txn["user_id"],
                "timestamp": burst_time,
                "amount": round(float(base_txn["amount"]) * random.uniform(0.7, 1.1), 2),
                "merchant_category": base_txn["merchant_category"],
                "city": base_txn["city"],
                "device_type": base_txn["device_type"],
                "is_international": base_txn["is_international"],
                "is_fraud": 1,
                "fraud_type": "burst_frequency"
            })
            
    if extra_burst_txns:
        df = pd.concat([df, pd.DataFrame(extra_burst_txns)], ignore_index=True)
        
    # Re-sort chronologically and regenerate unique transaction IDs
    df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
    df["transaction_id"] = [f"TXN_{i+1:06d}" for i in range(len(df))]
    
    # Drop fraud_type column from public dataset or keep it as an optional reference
    # FR1.2 specifies columns: transaction_id, user_id, timestamp, amount, merchant_category, city, device_type, is_international, is_fraud
    output_columns = [
        "transaction_id", "user_id", "timestamp", "amount", 
        "merchant_category", "city", "device_type", "is_international", "is_fraud"
    ]
    final_df = df[output_columns].copy()
    
    # Format timestamp as ISO string
    final_df["timestamp"] = final_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return final_df


def main():
    os.makedirs("data", exist_ok=True)
    df = generate_transactions(num_txns=TOTAL_TRANSACTIONS, fraud_ratio=FRAUD_RATIO)
    
    output_path = os.path.join("data", "transactions.csv")
    df.to_csv(output_path, index=False)
    
    total = len(df)
    fraud_cnt = df["is_fraud"].sum()
    fraud_pct = (fraud_cnt / total) * 100
    
    print("\nDataset successfully generated!")
    print(f"Saved to: {output_path}")
    print(f"Total Transactions: {total:,}")
    print(f"Fraud Transactions: {fraud_cnt:,} ({fraud_pct:.2f}%)")
    print(f"Legitimate Transactions: {total - fraud_cnt:,} ({100 - fraud_pct:.2f}%)")
    print("\nSample records:")
    print(df.head(3))


if __name__ == "__main__":
    main()

