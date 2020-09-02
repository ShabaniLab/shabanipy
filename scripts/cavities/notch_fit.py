import os

from shabanipy.utils.labber_io import LabberData

SAMPLE = 'JS314'
BASEPATH = '/Users/joe_yuan/Desktop/Desktop/Shabani Lab/Projects/ResonatorPaper'

DATA_DIR = os.path.join(BASEPATH, 'data')
CSV_DIR = os.path.join(BASEPATH, 'fits', SAMPLE)
IMG_DIR = os.path.join(BASEPATH, 'images', SAMPLE)

filenames = [
    'JS314_CD1_att20_004',
    'JS314_CD1_att40_006',
    'JS314_CD1_att60_007'
]

# Strip off '.hdf5' in case file extensions are left on
filenames = [i.rstrip('.hdf5') for i in filenames]

attenuation_on_vna = [-20, -40, -60]

for att, fname in zip(attenuation_on_vna, filenames):
    
    filepath = os.path.join(DATA_DIR, fname + '.hdf5')

    with LabberData(filepath) as data:
        print(data)
        print(data.list_channels())
