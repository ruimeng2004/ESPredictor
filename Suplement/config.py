import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DDI-ESPredictor Configuration')
    
    parser.add_argument('--model',
                      type=str,
                      default=None,
                      help='Pretrained model path')
    
    parser.add_argument('--input_drug1', 
                      type=str,
                      help='SMILES or ID of drug1')
    
    parser.add_argument('--input_drug2',
                      type=str,
                      help='SMILES or ID of drug2')
    
    parser.add_argument('--outputfile',
                      type=str,
                      default='results.csv',
                      help='Output file path')
    
    parser.add_argument('--feature',
                      type=str,
                      default='multi',
                      choices=['bert', 'fingerprint', '1D', '2D', '3D', 'multi','1D+2D','1D+3D','1D+bert','2D+3D','2D+bert','3D+bert','1D+2D+3D','1D+2D+bert','1D+3D+bert','2D+3D+bert'],
                      help='Feature extraction method')
    
    parser.add_argument('--fusion',
                      type=str,
                      default='yes',
                      choices=['yes', 'no'],
                      help='Enable cross-feature fusion')
    
    parser.add_argument('--lr',
                      type=float,
                      default=1e-3,
                      help='Learning rate')
    
    parser.add_argument('--batch_size',
                      type=int,
                      default=32,
                      help='Training batch size')
    
    parser.add_argument('--device',
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to use')
    
    return parser.parse_args()
