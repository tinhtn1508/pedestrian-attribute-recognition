import pickle
import numpy as np

# m = {}
# with open('glove.6B.300d.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.split()
#         data = [float(i) for i in line[1:-1]]
#         m[line[0]] = data

# with open('glove.6B.300d.pkl', 'wb') as f:
#     pickle.dump(m, f)

with open('glove.6B.300d.pkl', 'rb') as f:
    m = pickle.load(f)

attributes_map = {
                'Female': ['female'],
                'AgeOver60': ['odler'],
                'Age18-60': ['younger'],
                'AgeLess18': ['teenager'],
                'Front': ['front'],
                'Side': ['side'],
                'Back': ['back'],
                'Hat': ['hat'],
                'Glasses': ['glasses'],
                'HandBag': ['handbag'],
                'ShoulderBag': ['shoulder', 'bag'],
                'Backpack': ['backpack'],
                'HoldObjectsInFront': ['hold', 'objects', 'in', 'front'],
                'ShortSleeve': ['short', 'sleeve'],
                'LongSleeve': ['short', 'sleeve'],
                'UpperStride': ['upper', 'stride'],
                'UpperLogo': ['upper', 'logo'],
                'UpperPlaid': ['upper', 'plaid'],
                'UpperSplice': ['upper', 'splice'],
                'LowerStripe': ['lower', 'stripe'],
                'LowerPattern': ['lower', 'pattern'],
                'LongCoat': ['long', 'coat'],
                'Trousers': ['trousers'],
                'Shorts': ['shorts'],
                'Skirt&Dress': ['skirt', 'dress'],
                'boots': ['boots']
                }

