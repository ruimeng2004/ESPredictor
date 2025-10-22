import numpy as np
import torch
import pandas as pd
import os
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
from Suplement.FE.FE_BERT.FE_Bert import gnnsmile
from Suplement.FE.FE_3Ds.preprocess import smiles_to_data, prepare_batch
from Suplement.FE.FE_3Ds.pretrain_MolMVC import EmbeddingModel

def process_dataset(train_data, feature_type, device, save_dir="extracted_features"):
    os.makedirs(save_dir, exist_ok=True)
    
    drug1_ids = np.array(train_data["Drug1_ID"])
    drug2_ids = np.array(train_data["Drug2_ID"])
    all_drugs = np.unique(np.append(drug1_ids, drug2_ids))
    ftrain = list(all_drugs)
    
    alldrugtrain = {}
    removelist = []
    
    print(f"Starting to process {len(ftrain)} drugs...")
    
    for drug_id in ftrain:
        try:
            print(f"\nProcessing drug: {drug_id}")
            
            try:
                index = list(drug1_ids).index(drug_id)
                drug_col = "Drug1_ID"
                smiles_col = train_data.columns[1]
                print(f"Found drug in Drug1_ID column, index: {index}")
            except ValueError:
                try:
                    index = list(drug2_ids).index(drug_id)
                    drug_col = "Drug2_ID"
                    smiles_col = train_data.columns[3]
                    print(f"Found drug in Drug2_ID column, index: {index}")
                except ValueError:
                    removelist.append(drug_id)
                    print(f"Error: Drug {drug_id} not found in the dataset")
                    continue
            
            drug_name = train_data.iloc[index][drug_col]
            smiles = train_data.iloc[index][smiles_col]
            print(f"Drug name: {drug_name}, SMILES: {smiles}")
            
            if feature_type == 'bert':
                print("Using BERT feature extraction...")
                drug_fingerprints = gnnsmile(smiles)
                drug_fingerprints = list(drug_fingerprints[0])
                print(f"BERT feature dimension: {len(drug_fingerprints)}")
                
            elif feature_type in ['1D', '2D', '3D', 'multi']:
                print(f"Using MolMVC model for {feature_type} feature extraction...")
                
                print("Step 1: Converting SMILES to data object...")
                try:
                    data = smiles_to_data(smiles, xyz_path=None)
                    print("Data object created successfully")
                except Exception as e:
                    print(f"Error: Unable to convert SMILES to data object: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                print("Step 2: Preparing batch data...")
                try:
                    batch = prepare_batch(data)
                    print("Batch data prepared successfully")
                except Exception as e:
                    print(f"Error: Unable to prepare batch data: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                print("Step 3: Loading EmbeddingModel...")
                try:
                    model = EmbeddingModel().to(device)
                    model.eval()
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error: Unable to load model: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                print("Step 4: Extracting embeddings...")
                try:
                    with torch.no_grad():
                        emb_1d_low, emb_1d_high, emb_2d, emb_3d = model(batch)
                    print("Embeddings extracted successfully")
                except Exception as e:
                    print(f"Error: Failed to extract embeddings: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                if feature_type == "1D":
                    print("Processing 1D features...")
                    emb_1d = (emb_1d_low + emb_1d_high) / 2
                    drug_fingerprints = emb_1d.cpu().numpy().flatten().tolist()
                    print(f"1D feature dimension: {len(drug_fingerprints)}")
                    
                elif feature_type == "2D":
                    print("Processing 2D features...")
                    drug_fingerprints = emb_2d.cpu().numpy().flatten().tolist()
                    print(f"2D feature dimension: {len(drug_fingerprints)}")
                    
                elif feature_type == "3D":
                    print("Processing 3D features...")
                    drug_fingerprints = emb_3d.cpu().numpy().flatten().tolist()
                    print(f"3D feature dimension: {len(drug_fingerprints)}")
                    
                elif feature_type == "multi":
                    print("Processing multimodal features...")
                    emb_1d = (emb_1d_low + emb_1d_high) / 2
                    
                    print("Extracting BERT features...")
                    try:
                        bert_features = np.array(gnnsmile(smiles)[0])
                        print(f"BERT feature dimension: {len(bert_features)}")
                    except Exception as e:
                        print(f"Error: Failed to extract BERT features: {str(e)}")
                        removelist.append(drug_id)
                        continue
                    
                    print("Combining all features...")
                    try:
                        drug_fingerprints = np.concatenate([
                            emb_1d.cpu().numpy().flatten(),
                            emb_2d.cpu().numpy().flatten(),
                            emb_3d.cpu().numpy().flatten(),
                            bert_features
                        ]).tolist()
                        print(f"Multimodal feature total dimension: {len(drug_fingerprints)}")
                    except Exception as e:
                        print(f"Error: Failed to combine features: {str(e)}")
                        removelist.append(drug_id)
                        continue
                
            elif feature_type == "fingerprint":
                print("Using fingerprint feature extraction...")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Error: Cannot create molecule from SMILES: {smiles}")
                        removelist.append(drug_id)
                        continue
                    
                    fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    drug_fingerprints = [float(bit) for bit in fps.ToBitString()]
                    print(f"Fingerprint feature dimension: {len(drug_fingerprints)}")
                except Exception as e:
                    print(f"Error: Failed to extract fingerprint features: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
                
            alldrugtrain[drug_id] = drug_fingerprints
            print(f"Successfully extracted features for drug {drug_id}")
            
        except Exception as e:
            print(f"An uncaught exception occurred while processing drug {drug_id}: {str(e)}")
            traceback.print_exc()
            removelist.append(drug_id)
    
    for drug_id in removelist:
        if drug_id in ftrain:
            ftrain.remove(drug_id)
    
    if alldrugtrain:
        feature_matrix = np.array(list(alldrugtrain.values()))
        
        np.save(f"{save_dir}/{feature_type}_features.npy", feature_matrix)
        np.save(f"{save_dir}/{feature_type}_drug_ids.npy", np.array(ftrain))
        
        print(f"\n=== {feature_type} features stored successfully ===")
        print(f"Number of drugs: {len(alldrugtrain)}")
        print(f"Feature dimension: {feature_matrix.shape[1]}")
        print(f"Total samples: {feature_matrix.shape[0]}")
        print(f"Storage path: {save_dir}/{feature_type}_features.npy")
        
        if np.isnan(feature_matrix).any():
            print("Warning: Feature data contains NaN values!")
        if np.isinf(feature_matrix).any():
            print("Warning: Feature data contains Inf values!")
        
    else:
        print("Warning: No drug features were successfully extracted!")
    
    print(f"\nProcessing statistics:")
    print(f"Total number of drugs: {len(ftrain) + len(removelist)}")
    print(f"Successfully processed: {len(alldrugtrain)}")
    print(f"Failed processing: {len(removelist)}")
    
    if removelist:
        pd.DataFrame({'failed_drugs': removelist}).to_csv(f"{save_dir}/failed_drugs.csv", index=False)
        print(f"Saved failed drugs list to {save_dir}/failed_drugs.csv")
    
    return alldrugtrain, ftrain
