import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import numpy as np
import random

train = pd.read_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/filtered_train.csv')
test = pd.read_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/filtered_test.csv')

def calculate_tanimoto(fp1, fp2):
    return TanimotoSimilarity(fp1, fp2)

def filter_invalid_smiles(df):
    valid_indices = []
    for i in range(len(df)):
        mol1 = Chem.MolFromSmiles(df.iloc[i]['Drug1'])
        mol2 = Chem.MolFromSmiles(df.iloc[i]['Drug2'])
        if mol1 and mol2:
            valid_indices.append(i)
        else:
            print(f"Invalid SMILES at index {i}: Drug1={df.iloc[i]['Drug1']}, Drug2={df.iloc[i]['Drug2']}")
    return df.iloc[valid_indices].reset_index(drop=True)

filtered_train = filter_invalid_smiles(train)
filtered_train.to_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/filtered_train.csv', index=False)
filtered_test = filter_invalid_smiles(test)
filtered_test.to_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/filtered_test.csv', index=False)

def create_unique_drug_dataset(df):
    unique_drugs = pd.concat([
        df[['Drug1_ID', 'Drug1']].rename(columns={'Drug1_ID': 'Drug_ID', 'Drug1': 'Drug'}),
        df[['Drug2_ID', 'Drug2']].rename(columns={'Drug2_ID': 'Drug_ID', 'Drug2': 'Drug'})
    ])
    unique_drugs = unique_drugs.drop_duplicates(subset=['Drug_ID']).reset_index(drop=True)
    return unique_drugs

unique_train_drugs = create_unique_drug_dataset(filtered_train)
unique_test_drugs = create_unique_drug_dataset(filtered_test)

def generate_fingerprints_with_mapping(molecule_list, drug_ids):
    fingerprints = []
    for mol in molecule_list:
        if mol is None:
            print(f"Invalid molecule detected, skipping...")
            continue
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    
    drug_to_fingerprint = {}
    for i in range(len(fingerprints)):
        if fingerprints[i] is None:
            print(f"Fingerprint missing for drug ID {drug_ids[i]}")
        else:
            drug_to_fingerprint[drug_ids[i]] = fingerprints[i]
    return drug_to_fingerprint

train_mols = [Chem.MolFromSmiles(smi) for smi in unique_train_drugs['Drug'].tolist()]
test_mols = [Chem.MolFromSmiles(smi) for smi in unique_test_drugs['Drug'].tolist()]
train_drug_ids = unique_train_drugs['Drug_ID'].tolist()
test_drug_ids = unique_test_drugs['Drug_ID'].tolist()

train_fingerprints = generate_fingerprints_with_mapping(train_mols, train_drug_ids)
test_fingerprints = generate_fingerprints_with_mapping(test_mols, test_drug_ids)

def generate_negative_samples(data, positive_data, drug_to_fingerprint, num_neg_samples):
    neg_samples = []
    unique_neg_pairs = set()
    positive_pairs = set((row['Drug1_ID'], row['Drug2_ID']) for _, row in positive_data.iterrows())
    
    drug_ids = list(drug_to_fingerprint.keys())

    while len(neg_samples) < num_neg_samples:
        drug1_id = random.choice(drug_ids)
        drug1_fp = drug_to_fingerprint.get(drug1_id)
        
        if drug1_fp is None:
            print(f"Skipping {drug1_id} due to missing fingerprint.")
            continue
        
        min_sim_indices = np.argsort([calculate_tanimoto(drug1_fp, drug_to_fingerprint[drug_id]) if drug_to_fingerprint.get(drug_id) is not None else 1 for drug_id in drug_ids])
        min_sim_drug_ids = [drug_ids[i] for i in min_sim_indices]

        for drug2_id in min_sim_drug_ids:
            if drug1_id == drug2_id:
                continue
            
            drug2_fp = drug_to_fingerprint.get(drug2_id)
            if drug2_fp is None:
                print(f"Skipping {drug2_id} due to missing fingerprint.")
                continue
            
            tanimoto_sim = calculate_tanimoto(drug1_fp, drug2_fp)
            if tanimoto_sim >= 0.2: 
                break
            
            try:
                drug1 = data[data['Drug_ID'] == drug1_id]['Drug'].values[0]
                drug2 = data[data['Drug_ID'] == drug2_id]['Drug'].values[0]
            except IndexError:
                print(f"Drug pair not found for Drug1_ID={drug1_id} or Drug2_ID={drug2_id}. Skipping pair.")
                continue
            
            pair = tuple(sorted((drug1_id, drug2_id)))
            if pair not in positive_pairs and pair not in unique_neg_pairs:
                neg_samples.append({
                    'Drug1_ID': drug1_id,
                    'Drug1': drug1,
                    'Drug2_ID': drug2_id,
                    'Drug2': drug2,
                    'Y': 0
                })
                unique_neg_pairs.add(pair)
                print(f"Generated negative sample: Drug1_ID={drug1_id}, Drug2_ID={drug2_id}, Tanimoto similarity={tanimoto_sim:.4f}")
                break
    return pd.DataFrame(neg_samples)


neg_trainsamples = generate_negative_samples(unique_train_drugs,filtered_train, train_fingerprints, len(filtered_train))
combined_trainsamples = pd.concat([filtered_train, neg_trainsamples], ignore_index=True)
combined_trainsamples['Y'] = combined_trainsamples['Y'].apply(lambda x: 1 if x != 0 else 0)
combined_trainsamples.to_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/combinedtrain_stage1.csv', index=False)

neg_testsamples = generate_negative_samples(unique_test_drugs,filtered_test, test_fingerprints, len(filtered_test))
combined_testsamples = pd.concat([filtered_test, neg_testsamples], ignore_index=True)
combined_testsamples['Y'] = combined_testsamples['Y'].apply(lambda x: 1 if x != 0 else 0)
combined_testsamples.to_csv('/share/home/grp-huangxd/huangyixian/df_ddi/data/combinedtest_stage1.csv', index=False)

for dataset, name in zip([combined_trainsamples, combined_testsamples], ['Train', 'Test']):
    sample_counts = dataset['Y'].value_counts()
    print(f"{name} Positive samples (Y=1):", sample_counts.get(1, 0))
    print(f"{name} Negative samples (Y=0):", sample_counts.get(0, 0))
    print(f"{name} Dataset Balance Ratio:", sample_counts.get(1, 0) / max(sample_counts.get(0, 1), 1))
