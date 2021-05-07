# -*- coding: utf-8 -*-
"""SurvivalPredictionDocker.ipynb

"""



import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
import cv2
import csv
import pickle
import six
import pandas as pd
import glob
import nibabel as nib
import numpy as np
from glob import glob
from keras.utils import multi_gpu_model
from keras import optimizers
import matplotlib.pyplot as plt
import collections as clt
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, getTestCase
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, getTestCase, gldm, ngtdm
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
#from sklearn.cross_validation import train_test_split 
from sklearn.svm import SVR  
#from sklearn.metrics import mean_squared_error  
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl import load_workbook
import math
from collections import Counter
from openpyxl.styles import Font, colors, Alignment
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
import os
import numpy as np
import SimpleITK as sitk
import time
import pandas as pd
import nibabel as nib
from tqdm import tqdm

##### Pyradiomics parameters setting
# First define the settings
settings = {}
settings['binWidth'] = 25
settings['sigma'] = [0.5, 1.5, 2]
settings['interpolator']: sitk.sitkBSpline
settings['resampledPixelSpacing']: None
settings['normalization']: True
settings['label']:[1]

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  # ** 'unpacks' the dictionary in the function call

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)  # Still the default parameters
print('Enabled features:\n\t', extractor.enabledFeatures)  # Still the default parameters
# Enable a filter (in addition to the 'Original' filter already enabled)
extractor.enableImageTypeByName('LoG')
print('')
print('Enabled filters:\n\t', extractor.enabledImagetypes)
# Disable all feature classes, save firstorder
#extractor.disableAllFeatures()
#extractor.enableFeatureClassByName('firstorder')
print('')
print('Enabled features:\n\t', extractor.enabledFeatures)

# Specify some additional features in the firstorder feature class
extractor.enableFeaturesByName(firstorder=['Mean', 'Entropy', 'StandardDeviation'])
print('')
print('Enabled features:\n\t', extractor.enabledFeatures)

# Function to extract all the imaging features given folder_path and folder_id of a person
def extract_features(folder_path, folder_id,folder_id_seg):


        # Load in preprocessed mri volumes
    write = []
    scans=os.listdir(folder_path)
    print(scans)
    name = scans[0]
    seg=scans[1]
    orig_i=scans[2]
    image = nib.load(os.path.join(folder_path, name)).get_data()
    region = nib.load(os.path.join(folder_path, seg)).get_data()
    orig_image = nib.load(os.path.join(folder_path, orig_i)).get_data()
    print(region.shape)
    data = (np.array([1,2,4]) == np.array(region)[...,None]).astype(int)
    whole = image.copy()
    whole[whole>0] = 1
    whole_brain = orig_image.copy()
    whole_brain[whole_brain>0] = 1
    nec_volume = np.sum(data[...,0])
    ede_volume = np.sum(data[...,1])
    enh_volume = np.sum(data[...,2])

    Whole_Brain = np.sum(whole)/np.sum(whole_brain)
    nec_brain = nec_volume/np.sum(whole_brain)
    ede_brain = ede_volume/np.sum(whole_brain)
    enh_brain = enh_volume/np.sum(whole_brain)

    nec_enh_volume = nec_volume/enh_volume
    ede_enh_volume = ede_volume/enh_volume
    nec_ede_volume = nec_volume/ede_volume

    necrotic = data[..., 0]
    edema = data[..., 1]
    enhance = data[..., 2]
        
    grad_image = np.zeros(region.shape)
    for layer in range(grad_image.shape[-1]):
      sobelx = cv2.Sobel(np.float32(np.squeeze(necrotic[:,:,layer])),cv2.CV_64F, 1, 0, ksize=3)
      sobely = cv2.Sobel(np.float32(np.squeeze(necrotic[:,:,layer])),cv2.CV_64F, 0, 1,  ksize=3)
      grad_image[:,:,layer] = np.sqrt(sobelx**2+sobely**2)

      sobelx = cv2.Sobel(np.float32(np.squeeze(edema[:,:,layer])),cv2.CV_64F, 1, 0, ksize=3)
      sobely = cv2.Sobel(np.float32(np.squeeze(edema[:,:,layer])),cv2.CV_64F, 0, 1,  ksize=3)
      grad_image[:,:,layer] += np.sqrt(sobelx**2+sobely**2)

      sobelx = cv2.Sobel(np.float32(np.squeeze(enhance[:,:,layer])),cv2.CV_64F, 1, 0, ksize=3)
      sobely = cv2.Sobel(np.float32(np.squeeze(enhance[:,:,layer])),cv2.CV_64F, 0, 1,  ksize=3)
      grad_image[:,:,layer] += np.sqrt(sobelx**2+sobely**2)    
    grad_image[grad_image>0] = 1
    grad_image = grad_image * image
    grad_image = (np.array([1,2,4]) == np.array(grad_image)[...,None]).astype(int)
    whole_grad_image = grad_image.copy()
    whole_grad_image[whole_grad_image>0] = 1
    nec_grad_volume = np.sum(grad_image[...,0])
    ede_grad_volume = np.sum(grad_image[...,1])
    enh_grad_volume = np.sum(grad_image[...,2])
        
    nec_SVRatio = nec_grad_volume/nec_volume
    ede_SVRatio = ede_grad_volume/ede_volume
    enh_SVRatio = enh_grad_volume/enh_volume

    whole_volume = np.sum(whole)
    whole_coord = np.mean(np.where(whole == 1), 1)
    whole_brain_coord = np.mean(np.where(whole_brain == 1), 1)

    whole_grad_volume = np.sum(whole_grad_image)
    whole_SVRatio = whole_grad_volume/whole_volume

    X = [math.floor(whole_coord[0]), math.ceil(whole_coord[0])]
    Y = [math.floor(whole_coord[1]), math.ceil(whole_coord[1])]
    Z = [math.floor(whole_coord[2]), math.ceil(whole_coord[2])]

    Region_value = [region[i][j][k] for i in X for j in Y for k in Z]
    ID_value = Counter(Region_value)
    whole_value = ID_value.most_common(1)[0][0]
        

    if enh_volume == 0:
      true_coord = whole_coord
    else:
      true_coord = np.mean(np.where(enhance == 1), 1)

    True_X = [math.floor(true_coord[0]), math.ceil(true_coord[0])]
    True_Y = [math.floor(true_coord[1]), math.ceil(true_coord[1])]
    True_Z = [math.floor(true_coord[2]), math.ceil(true_coord[2])]

    Region_value = [region[i][j][k] for i in True_X for j in True_Y for k in True_Z]
    ID_value = Counter(Region_value)
    true_value = ID_value.most_common(1)[0][0]

    print(whole_value,true_value)

    write.append(name)
    write.append(whole_volume)
    write.append(nec_volume)
    write.append(ede_volume)
    write.append(enh_volume)

    write.append(Whole_Brain)
    write.append(nec_brain)
    write.append(ede_brain)
    write.append(enh_brain)
        
    write.append(nec_enh_volume)
    write.append(ede_enh_volume)
    write.append(nec_ede_volume)

    write.append(whole_grad_volume)
    write.append(nec_grad_volume)
    write.append(ede_grad_volume)
    write.append(enh_grad_volume)

    write.append(whole_SVRatio)
    write.append(nec_SVRatio)
    write.append(ede_SVRatio)
    write.append(enh_SVRatio)

    write.append(whole_value)
    write.append(true_value)

    write.append(round(whole_coord[0], 1))
    write.append(round(whole_coord[1], 1))
    write.append(round(whole_coord[2], 1))

    write.append(round(true_coord[0], 1))
    write.append(round(true_coord[1], 1))
    write.append(round(true_coord[2], 1))

    write.append(round(whole_coord[0], 1) - round(whole_brain_coord[0], 1))
    write.append(round(whole_coord[1], 1) - round(whole_brain_coord[1], 1))
    write.append(round(whole_coord[2], 1) - round(whole_brain_coord[2], 1))

    write.append(round(true_coord[0], 1) - round(whole_brain_coord[0], 1))
    write.append(round(true_coord[1], 1) - round(whole_brain_coord[1], 1))
    write.append(round(true_coord[2], 1) - round(whole_brain_coord[2], 1))
    print(write)

    # Load in preprocessed mri volumes
    scans=os.listdir(folder_path)
    #scans = np.load(r"{}/{}.nii.gz".format(folder_path, folder_id))

    # Get t1ce and flair image from which to extract features
    img=nib.load(os.path.join(folder_path,scans[1])).get_data()
    t1ce_img = sitk.GetImageFromArray(img)
    img=nib.load(os.path.join(folder_path,scans[3])).get_data()
    flair_img = sitk.GetImageFromArray(img)
    img=nib.load(os.path.join(mri_seg_path,folder_id_seg+'.nii.gz')).get_data()
    print("shape of mask")
    print(img.shape)
    mask= img

    nr_classes = len(np.unique(mask))
    print(nr_classes)
    enhancing = (mask == 3).astype('long')
    edema = (mask == 2).astype('long')
    ncr_nenhancing = (mask == 1).astype('long')
    whole_tumor = (mask > 0).astype('long')

    regions = {'edema': {'mask': edema, 'modality': flair_img}, 'enhancing': {'mask': enhancing, 'modality': t1ce_img},
               'ncr_nenhancing': {'mask':ncr_nenhancing, 'modality': t1ce_img}, 'whole_tumor': {'mask':whole_tumor, 'modality':t1ce_img}}

    # Convert the region arrays into SITK image objects so they can be inputted to the PyRadiomics featureextractor functions.
    all_features = {}
    printed = 0
    if nr_classes == 4:
        for (region_name, images) in regions.items():
            lbl_img = sitk.GetImageFromArray(mask)
            # Get First order features
            try:
              result = extractor.execute(images['modality'], lbl_img)
              previous_features=result
            except:
              print("Unexpected error:")
              result=previous_features
              raise
            for (key, val) in result.items():
                all_features[region_name + '_' + key] = val
    else:
        print(folder_id)
    return all_features, nr_classes

def predict_test_features(mri_data_path,mri_seg_path,survival_data_path):
  survival_data=pd.read_csv(survival_data_path)
  survival_data_gtr=survival_data[survival_data['ResectionStatus']=='GTR']
  patients_gtr=survival_data_gtr['BraTS20ID'].values
  folder_ids=patients_gtr.tolist()
  folder_ids_seg=patients_gtr.tolist()
  print(len(folder_ids_seg))
  print(len(folder_ids))
  print(len(patients_gtr))
  # Get paths and names (IDS) of folders that store the preprocessed data for each example
  folder_paths = []
  for subdir in patients_gtr:
    folder_paths.append(os.path.join(mri_data_path, subdir))
    if (mri_data_path in patients_gtr):
      #print(mri_data_path)
      print(len(mri_data_path))
  print("junaid")
  print(len(folder_paths))
  features = {}
  start = time.time()
  not_seg = 0
  for idx in tqdm(range(0, len(folder_ids_seg))):  # Loop over every person,
  #for idx in tqdm(range(0, 5)):  # Loop over every person,
    print(folder_ids_seg)
    try:
      feat, nr_cl = extract_features(folder_paths[idx], folder_ids[idx],folder_ids_seg[idx])
      previous_features=feat
      print("JJ")
    except Exception as e:
      print(":")
      feat=previous_features
    finally: #but this work
      print(":")
      feat=previous_features
    if nr_cl == 4:
        features[folder_ids[idx]] = feat
    else:
        features[folder_ids[idx]] = previous_features
        not_seg += 1
    print("Extracted features from person {}/{}".format(idx + 1, len(folder_paths)))
  print("{} not segmented".format(not_seg))
  elapsed = time.time() - start
  hours, rem = divmod(elapsed, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Extracting Features took {} min {} s".format(minutes, seconds))

  features = pd.DataFrame.from_dict(features, orient='index')
  surv_data = pd.read_csv(survival_data_path, index_col=0)

  ages = surv_data['Age'].astype('float')  # Only get ages of people who to keep in training data
  #surv = surv_data['Survival'][to_keep].astype('float')

  features['Age'] = ages[features.index]
  #features['Survival'] = surv[features.index]
  features=features.drop(['edema_diagnostics_Versions_PyRadiomics','edema_diagnostics_Versions_Numpy','edema_diagnostics_Versions_SimpleITK','edema_diagnostics_Versions_PyWavelet','edema_diagnostics_Versions_Python','edema_diagnostics_Configuration_Settings',
               'edema_diagnostics_Configuration_EnabledImageTypes','edema_diagnostics_Image-original_Hash','edema_diagnostics_Image-original_Dimensionality','edema_diagnostics_Image-original_Spacing',
               'edema_diagnostics_Image-original_Size','edema_diagnostics_Mask-original_Hash','edema_diagnostics_Mask-original_Spacing','edema_diagnostics_Mask-original_Size','edema_diagnostics_Mask-original_BoundingBox',
               'edema_diagnostics_Mask-original_CenterOfMassIndex','edema_diagnostics_Mask-original_CenterOfMass','enhancing_diagnostics_Versions_PyRadiomics','enhancing_diagnostics_Versions_Numpy','enhancing_diagnostics_Versions_SimpleITK',
               'enhancing_diagnostics_Versions_PyWavelet','enhancing_diagnostics_Versions_Python','enhancing_diagnostics_Configuration_Settings','enhancing_diagnostics_Configuration_EnabledImageTypes',
               'enhancing_diagnostics_Image-original_Hash','enhancing_diagnostics_Image-original_Dimensionality','enhancing_diagnostics_Image-original_Spacing','enhancing_diagnostics_Image-original_Size',
               'enhancing_diagnostics_Mask-original_Hash','enhancing_diagnostics_Mask-original_Spacing','enhancing_diagnostics_Mask-original_Size','enhancing_diagnostics_Mask-original_BoundingBox',
               'enhancing_diagnostics_Mask-original_CenterOfMassIndex','enhancing_diagnostics_Mask-original_CenterOfMass','ncr_nenhancing_diagnostics_Versions_PyRadiomics','ncr_nenhancing_diagnostics_Versions_Numpy',
               'ncr_nenhancing_diagnostics_Versions_SimpleITK','ncr_nenhancing_diagnostics_Versions_PyWavelet','ncr_nenhancing_diagnostics_Versions_Python','ncr_nenhancing_diagnostics_Configuration_Settings','ncr_nenhancing_diagnostics_Configuration_EnabledImageTypes',
               'ncr_nenhancing_diagnostics_Image-original_Hash','ncr_nenhancing_diagnostics_Image-original_Dimensionality','ncr_nenhancing_diagnostics_Image-original_Spacing','ncr_nenhancing_diagnostics_Image-original_Size',
               'ncr_nenhancing_diagnostics_Mask-original_Hash','ncr_nenhancing_diagnostics_Mask-original_Spacing','ncr_nenhancing_diagnostics_Mask-original_Size','ncr_nenhancing_diagnostics_Mask-original_BoundingBox',
               'ncr_nenhancing_diagnostics_Mask-original_CenterOfMassIndex','ncr_nenhancing_diagnostics_Mask-original_CenterOfMass','whole_tumor_diagnostics_Versions_PyRadiomics','whole_tumor_diagnostics_Versions_Numpy',
               'whole_tumor_diagnostics_Versions_SimpleITK','whole_tumor_diagnostics_Versions_PyWavelet','whole_tumor_diagnostics_Versions_Python','whole_tumor_diagnostics_Configuration_Settings',
               'whole_tumor_diagnostics_Configuration_EnabledImageTypes','whole_tumor_diagnostics_Image-original_Hash','whole_tumor_diagnostics_Image-original_Dimensionality','whole_tumor_diagnostics_Image-original_Spacing',
               'whole_tumor_diagnostics_Image-original_Size','whole_tumor_diagnostics_Mask-original_Hash','whole_tumor_diagnostics_Mask-original_Spacing','whole_tumor_diagnostics_Mask-original_Size','whole_tumor_diagnostics_Mask-original_BoundingBox',
               'whole_tumor_diagnostics_Mask-original_CenterOfMassIndex','whole_tumor_diagnostics_Mask-original_CenterOfMass'], axis=1)
  test_data=features
  print(test_data.shape)
  test_data=test_data.iloc[:,[i for i in range(35,1322)]]
  # load the model from disk
  loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
  predictions=loaded_model.predict(test_data)
  survival_data_gtr=survival_data[survival_data['ResectionStatus']=='GTR']
  patients_gtr=survival_data_gtr['BraTS20ID'].values
  patients_gtr.tolist()
  predictions.tolist()
  df = pd.DataFrame(list(zip(patients_gtr.tolist(), predictions.tolist())))
  df.to_csv('results/predictions.csv',sep=',', index = False)

###################################################################
# Code for extracting all imaging features from preprocessed training set and saving them in a csv file, along with the
# age and survival outcome inserted at the last two columns.

# To specify - path where the preprocessed mri scans are stored, path where to load model from for obtaining segmentations
# and path of survival data csv file
mri_data_path = "data/MICCAI_BraTS2020_TestingData"
mri_seg_path = "data/Segmented_Test"
survival_data_path = 'data/survival_evaluation_test.csv'

####################################################################

predict_test_features(mri_data_path,mri_seg_path,survival_data_path)

