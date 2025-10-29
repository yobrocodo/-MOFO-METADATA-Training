import torch
import os

def convert_weights_to_cpu(weights_path):
    print(f"Converting {weights_path} to CPU format...")
    try:
        # Load the weights file with CPU mapping
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # Create backup of original file
        backup_path = weights_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(weights_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # Save the weights back to the original filename
        torch.save(weights, weights_path)
        print(f"Successfully converted {weights_path} to CPU format!")
        return True
    except Exception as e:
        print(f"Error converting {weights_path}: {str(e)}")
        return False

# Convert the main model weights
convert_weights_to_cpu('mofo_weights/mofo_weight_sel.pth')

# Look for any other .pth files that might need conversion
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pth') and 'backup' not in file:
            weights_path = os.path.join(root, file)
            if weights_path != 'mofo_weights/mofo_weight_sel.pth':  # Skip the one we already did
                convert_weights_to_cpu(weights_path)