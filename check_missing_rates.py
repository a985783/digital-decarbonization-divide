
import pandas as pd
import os
from scripts.analysis_config import load_config

def check_missing():
    cfg = load_config("analysis_spec.yaml")
    dci_components = cfg["dci_components"]
    
    input_file = os.path.join("data", "wdi_expanded_raw.csv")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    
    print("Missing rates for DCI components:")
    for col in dci_components:
        if col in df.columns:
            missing = df[col].isna().sum()
            total = len(df)
            pct = missing / total * 100
            print(f"- {col}: {pct:.2f}% ({missing}/{total})")
        else:
            print(f"- {col}: Column not found")

if __name__ == "__main__":
    check_missing()
