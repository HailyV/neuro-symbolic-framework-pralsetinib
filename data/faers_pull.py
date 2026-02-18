import requests
import pandas as pd
import os

def get_faers_counts(drug_name):
    # API query
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:\"{drug_name}\"&count=patient.reaction.reactionmeddrapt.exact"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json().get('results', [])
            return pd.DataFrame(data).rename(columns={'term': 'ae', 'count': 'target_faers_score'})
        else:
            print(f"Warning: status {response.status_code} for {drug_name}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error for {drug_name}: {e}")
        return pd.DataFrame()

drug_list = ["SELPERCATINIB", "VANDETANIB", "CABOZANTINIB"]

# Collect data
all_data = []
for drug in drug_list:
    print(f"Fetching data for: {drug}...")
    df_drug = get_faers_counts(drug)
    if not df_drug.empty:
        df_drug['drug_id'] = f"drug:{drug.capitalize()}"
        all_data.append(df_drug)

# Labeling
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create the binary label
    combined_df['label'] = (combined_df['target_faers_score'] > 10).astype(int)

    # Save to CSV
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "comparative_drug_ae_data.csv")
    combined_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Success, data saved to: {output_path}")
    print(f"Total rows collected: {len(combined_df)}")
    print("Label distribution:")
    print(combined_df['label'].value_counts())
else:
    print("No data was collected")