################################################################################
# Part 1: Preprocessing Dose Information
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re

from google.colab import drive
drive.mount('/content/drive')

# The paths are based on our Drive, be careful and change them if you need
import os
folder = "drive/MyDrive"
path_dataset= "dataset_students.xlsx"
name_files_doses_patients_downloaded= 'interpolated_doses_all_patients (1).npz'
name_file_doses_patient_in_your_folder='interpolated_doses_all_patients_test.npz'

### Pre Work on the data

## MEGA DOWNLOADING

# To download the file here
!apt-get install -y megatools
!megadl 'https://mega.nz/file/Z5JHXYAb#caoSZaHQ8iP1emyKkOftT_PkRMG9KGpNXB5P7aZJodU'

# To copy the file into colab
import shutil

src = name_files_doses_patients_downloaded
dst = os.path.join(folder,name_file_doses_patient_in_your_folder)

shutil.copy(src, dst)

## Importing the data

# Dataset
df=pd.read_excel(os.path.join(folder,path_dataset) ,sheet_name=1)

# We remove the id that are not in the patients from MIM
id_to_remove_in_df_sure=["LM1027","LM1028","LM1007"]
id_to_remove_in_df_not_sure=["LM1009","LM-02024","LM-02040","LM-02061-1","LM-05008","LM-050011","LM-050015","LM6037"]
id_to_remove_in_df=id_to_remove_in_df_sure +id_to_remove_in_df_not_sure

mask = ~df["ID"].isin(id_to_remove_in_df)

df = df[mask].reset_index()

# Patient id
id_to_remove_in_mim_sure= ["LM-02035_t"] # 87
id_to_remove_in_mim_not_sure=["LM-02061-2_t"] # 112
id_to_dedouble_in_mim=["LM-050001_t","3049_t","LM6008_t"] # (44,45) , (131,132), (250,251)
id_to_remove_in_mim=id_to_remove_in_mim_sure + id_to_remove_in_mim_sure
index_to_remove_mim= [44,87,112,131,250]

# Import the data from MIM
data = np.load(dst)

doses=data[data.files[0]]
patients_id=data[data.files[1]]
del data

# Filter the patients that we do not keep because they are not in the dataset
mask = np.ones(len(patients_id), dtype=bool)
mask[index_to_remove_mim] = False
patients_id = patients_id[mask]

np.save(os.path.join(folder,"initial_array_doses"), doses)

## Aligning the ID from dataset and MIM patients

# Put in the same format that were in MIM the ID of our dataset
def align_id(id):
  if id=="LM3042":
    return id+ "_t"
  if id[2]=="3":
    id= id[2:]
  return id+ "_t"
df["ID_cleaned"]=df.ID.apply(align_id)

# Map the index of patient from MIM to the index of our current dataset
id_list=df.ID_cleaned.to_list()
new_index=[]
for patient_id in patients_id:
  new_index.append(id_list.index(patient_id))


# Reindex the dataset with the index of patient received by MIM
df_aligned=df.reindex(new_index).reset_index()

# Save them
df_aligned.to_csv(os.path.join('data_reindexed.csv'), index=False)

## Creating the final array
doses_memmap = np.load(os.path.join(folder, "initial_array_doses.npy"), mmap_mode='r')
d = df_aligned["Number of Fractions"].values
ratio = 2
keep_indices = [i for i in range(doses_memmap.shape[0]) if i not in index_to_remove_mim]
shape = (len(keep_indices),) + doses_memmap.shape[1:]

# Output file
output_path = os.path.join(folder, "doses_scaled_ratio_2.npy")
doses_scaled = np.lib.format.open_memmap(output_path, mode='w+', dtype=doses_memmap.dtype, shape=shape)

# Creating the final array
output_index = 0
for i in keep_indices:
    scaled = doses_memmap[i] * (doses_memmap[i] / d[output_index] + ratio) / (2 + ratio) # We compute the formula for each voxel based on the number of fractions
    doses_scaled[output_index] = np.nan_to_num(scaled, nan=0.0)  # We replace the NA with 0
    output_index += 1
doses_scaled.flush()


# ratio = 5

# # Output file
# output_path = os.path.join(folder, "doses_scaled_ratio_5.npy")
# doses_scaled = np.lib.format.open_memmap(output_path, mode='w+', dtype=doses_memmap.dtype, shape=shape)

# # Creating the final array
# output_index = 0
# for i in keep_indices:
#     scaled = doses_memmap[i] * (doses_memmap[i] / d[output_index]+ ratio) / (2 + ratio) # We compute the formula for each voxel based on the number of fractions
#     doses_scaled[output_index] = np.nan_to_num(scaled, nan=0.0)  # We replace the NA with 0
#     output_index += 1

# ratio = 10

# # Output file
# output_path = os.path.join(folder, "doses_scaled_ratio_10.npy")
# doses_scaled = np.lib.format.open_memmap(output_path, mode='w+', dtype=doses_memmap.dtype, shape=shape)

# # Creating the final array
# output_index = 0
# for i in keep_indices:
#     scaled = doses_memmap[i] * (doses_memmap[i] / d[output_index] + ratio) / (2 + ratio) # We compute the formula for each voxel based on the number of fractions
#     doses_scaled[output_index] = np.nan_to_num(scaled, nan=0.0)  # We replace the NA with 0
#     output_index += 1

ratio=2
x_1=np.linspace(0,100,10000)
n=10
voxels_values_1=[x* (x/n+ratio)/(2+ratio) for x in np.linspace(0,100,10000)]
ratio=5
x_1=np.linspace(0,100,10000)
n=10
voxels_values_1=[x* (x/n+ratio)/(2+ratio) for x in np.linspace(0,100,10000)]
ratio=10
x_1=np.linspace(0,100,10000)
n=10
voxels_values_1=[x* (x/n+ratio)/(2+ratio) for x in np.linspace(0,100,10000)]

plt.plot(voxels_values,x)

################################################################################
# Part 2: Analyzing Dose Ratios (Version ratio=2)
################################################################################


index_to_remove_mim= [44,87,112,131,250]
doses_memmap = np.load("drive/MyDrive/initial_array_doses.npy", mmap_mode='r')
d = df_aligned["Number of Fractions"].values

keep_indices = [i for i in range(doses_memmap.shape[0]) if i not in index_to_remove_mim]
shape = (len(keep_indices),) + doses_memmap.shape[1:]

ratio = 2

# Output file
output_path = "drive/MyDrive/doses_scaled_ratio_2.npy"
doses_scaled = np.lib.format.open_memmap(output_path, mode='w+', dtype=doses_memmap.dtype, shape=shape)

# Creating the final array
output_index = 0
for i in keep_indices:
    scaled = doses_memmap[i] * (doses_memmap[i] / d[output_index] + ratio) / (2 + ratio) # We compute the formula for each voxel based on the number of fractions
    doses_scaled[output_index] = np.nan_to_num(scaled, nan=0.0)  # We replace the NA with 0
    output_index += 1


## Data Vizualisation Survival and No Survival

# Import the doses scaled and compute the doses reduced focusing only on this square part
doses_scaled_ratio_2=np.load("drive/MyDrive/doses_scaled_ratio_2.npy", mmap_mode='r')
doses_reduced_ratio_2=doses_scaled_ratio_2[:,20:146,40:140,40:180]

# Compute the indices to have the two groups
overall_survival=df_aligned["Overall Survival at 2 years"]
indices_yes = np.where(overall_survival == 'YES')[0]
indices_no = np.where(overall_survival == 'NO')[0]

# Compute the mean for the axis 0 (patient axis) for the two groups
mean_yes = np.mean(doses_scaled_ratio_2[indices_yes], axis=0)
mean_no = np.mean(doses_scaled_ratio_2[indices_no], axis=0)

np.save("drive/MyDrive/mean_yes_ratio_2",mean_yes)
np.save("drive/MyDrive/mean_no_ratio_2",mean_no)

## Vizualisation slices survival vs non survival

# Plot at a slice level the mean doses for survival, no-survival and their differences
import ipywidgets as widgets
from IPython.display import display

def plot_slice(i):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # --- VMIN/VMAX for the common scal
    vmin_common = min(mean_yes[i].min(), mean_no[i].min())
    vmax_common = max(mean_yes[i].max(), mean_no[i].max())

    # 1. YES
    im1 = axes[0].imshow(mean_yes[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[0].set_title(f'Survival YES - Slice Z={i}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], shrink=0.7)

    # 2. NO
    im2 = axes[1].imshow(mean_no[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(f'Survival NO - Slice Z={i}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], shrink=0.7)

    # 3. DIFF (YES - NO)
    diff = mean_yes[i] - mean_no[i]
    max_abs = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap='bwr', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title(f'DIFF (YES - NO) - Slice Z={i}')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    plt.show()

widgets.interact(plot_slice, i=(0, mean_yes.shape[0] - 1));


# Same plot with the mask to show what we will not consider in the further code
import matplotlib.patches as patches

# mask of indices to not consider
noise_mask = np.ones((147, 168, 201), dtype=bool)
noise_mask[20:146,40:140,40:180] = False

diff = mean_yes - mean_no
threshold = 0
diff[np.abs(diff) < threshold] = 0

def plot_slice(i):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Scaling of the barcolor
    vmin_common = min(mean_yes[i].min(), mean_no[i].min())
    vmax_common = max(mean_yes[i].max(), mean_no[i].max())

    # YES
    im1 = axes[0].imshow(mean_yes[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[0].set_title(f'Survival YES - Slice Z={i}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], shrink=0.7)

    y_coords, x_coords = np.where(noise_mask[i])
    axes[0].scatter(x_coords, y_coords, color='red', s=5)

    # NO
    im2 = axes[1].imshow(mean_no[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(f'NO Survival - Slice Z={i}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], shrink=0.7)
    axes[1].scatter(x_coords, y_coords, color='red', s=5)

    # DIFF
    max_abs = np.max(np.abs(diff[i]))
    im3 = axes[2].imshow(diff[i], cmap='bwr', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title(f'DIFF (YES - NO) - Slice Z={i}')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], shrink=0.7)

    # Black Square corresponds to the red square limit
    y_coords, x_coords = np.where(~noise_mask[i])
    if np.any(~noise_mask[i]):
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        axes[2].add_patch(rect)

    plt.tight_layout()
    plt.show()

widgets.interact(plot_slice, i=(0, mean_yes.shape[0] - 1));


## Statistics on 3D images of Mean of two groups (mean_yes/no)

number_of_0=[]
mean_image_3D=[]
for i in range(doses_scaled_ratio_2.shape[0]):
  mask=doses_scaled_ratio_2[i]==0
  number_of_0.append(doses_scaled_ratio_2[i][mask].size /  (147* 168* 201) ) #conta il numero totale di voxel a zero e 
  #normalizza per il numero teorico di voxels
  mean_image_3D.append(doses_scaled_ratio_2[i][~mask].mean()) #prende i voxel con dose diversa da zero e fa la media #della dose 


number_of_0 = np.array(number_of_0)
mean_image_3D = np.array(mean_image_3D)

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# --- Boxplots
axes[0, 0].boxplot([number_of_0[indices_yes], number_of_0[indices_no]],
                   labels=['YES', 'NO'], patch_artist=True)
axes[0, 0].set_title('% of voxels = 0')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].grid(True, linestyle='--', alpha=0.3)

axes[0, 1].boxplot([mean_image_3D[indices_yes], mean_image_3D[indices_no]],
                   labels=['YES', 'NO'], patch_artist=True)
axes[0, 1].set_title('Mean voxels ≠ 0')
axes[0, 1].set_ylabel('Mean Dose')
axes[0, 1].grid(True, linestyle='--', alpha=0.3)

# --- Histograms for % of 0
bins_0 = np.linspace(np.min(number_of_0), np.max(number_of_0), 30)
hist_yes_0 = sns.histplot(number_of_0[indices_yes], bins=bins_0, ax=axes[1, 0],
                          color='blue', kde=False, stat='probability')
hist_no_0 = sns.histplot(number_of_0[indices_no], bins=bins_0, ax=axes[1, 1],
                         color='orange', kde=False, stat='probability')

# Same scale of Y
max_y_0 = max(hist_yes_0.get_ylim()[1], hist_no_0.get_ylim()[1])
axes[1, 0].set_ylim(0, max_y_0)
axes[1, 1].set_ylim(0, max_y_0)

axes[1, 0].set_title('Histogram % voxels = 0 (YES)')
axes[1, 0].set_xlabel('Proportion of 0')
axes[1, 0].set_ylabel('Porcentage')
axes[1, 0].grid(True, linestyle='--', alpha=0.3)

axes[1, 1].set_title('Histogram % voxels = 0 (NO)')
axes[1, 1].set_xlabel('Proportion of 0')
axes[1, 1].set_ylabel('Porcentage')
axes[1, 1].grid(True, linestyle='--', alpha=0.3)

# --- Histograms for means ≠ 0
bins_mean = np.linspace(np.min(mean_image_3D), np.max(mean_image_3D), 30)
hist_yes_mean = sns.histplot(mean_image_3D[indices_yes], bins=bins_mean, ax=axes[2, 0],
                             color='blue', kde=False, stat='probability')
hist_no_mean = sns.histplot(mean_image_3D[indices_no], bins=bins_mean, ax=axes[2, 1],
                            color='orange', kde=False, stat='probability')

# Same scale of Y
max_y_mean = max(hist_yes_mean.get_ylim()[1], hist_no_mean.get_ylim()[1])
axes[2, 0].set_ylim(0, max_y_mean)
axes[2, 1].set_ylim(0, max_y_mean)

axes[2, 0].set_title('Histogram mean voxels ≠ 0 (YES)')
axes[2, 0].set_xlabel('Mean Dose')
axes[2, 0].set_ylabel('Porcentage')
axes[2, 0].grid(True, linestyle='--', alpha=0.3)

axes[2, 1].set_title('Histogram mean voxels ≠ 0 (NO)')
axes[2, 1].set_xlabel('Mean dose')
axes[2, 1].set_ylabel('Porcentage')
axes[2, 1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


print(f"Percentage of values= 0 in mean_yes : {round(np.mean(mean_yes == 0)*100,2)} %")
print(f"Percentage of values= 0 in mean_no : {round(np.mean(mean_no == 0)*100,2)} %")

print("mean_values for survivor",np.mean(mean_yes))
print("mean_values for non-survivor",np.mean(mean_no))
print("mean_values for values different than 0 for survivor",np.mean(mean_yes[mean_yes>0]))
print("mean_values for values different than 0 for non-survivor",np.mean(mean_no[mean_no>0]))

# # Supervoxel

# This function will return the full doses shape matrix with the p_value associated with the supervoxel algorithm
# It drops the supervoxels where the absolute difference between the supervoxel of the survivor and the non-survivor is less than the threshold
def supervoxel_pvalues(n_clusters,compact,mean,min_size=0.5,max_size=3,mean_non_survival=mean_no,mean_survival=mean_yes,doses_of_patients_4D=doses_reduced_ratio_2,index_yes=indices_yes,index_no=indices_no,threshold=0):

  from skimage.segmentation import slic
  from skimage.util import img_as_float
  from scipy import ndimage
  from scipy.stats import t
  from scipy.stats import false_discovery_control
  from IPython.display import HTML, display

  print(f"The algorithm is running for n_clusters={n_clusters}, compactness={compact}, min_size={min_size}, max_size={max_size} with {doses_of_patients_4D.shape[0]} patients")

  # Computing the labels of the superclusters
  labels = slic(mean,max_num_iter=50, n_segments=n_clusters, compactness=compact,min_size_factor=min_size,max_size_factor=max_size,  start_label=1,channel_axis=None,enforce_connectivity=True)
  print("the Number of cluster found is", np.unique(labels).size)

  # Computing the supervoxels

  label_values = np.unique(labels)
  n_patients = doses_of_patients_4D.shape[0]
  n_labels = len(label_values)
  supervoxels = np.zeros((n_patients, n_labels))
  for j in range(n_patients):
    supervoxels[j] = ndimage.mean(doses_of_patients_4D[j], labels=labels, index=label_values)

  # Computing all the individuals t test

  mean_supervoxel_yes=np.mean(supervoxels[index_yes],axis=0)
  mean_supervoxel_no=np.mean(supervoxels[index_no],axis=0)
  supervoxel_sample_var_yes = np.var(supervoxels[index_yes], axis=0, ddof=1)
  supervoxel_sample_var_no = np.var(supervoxels[index_no], axis=0, ddof=1)

  n_yes = len(index_yes)
  n_no = len(index_no)
  t_stat = (mean_supervoxel_yes - mean_supervoxel_no) / np.sqrt(supervoxel_sample_var_no / n_no + supervoxel_sample_var_yes / n_yes)

  numerator = (supervoxel_sample_var_yes / n_yes + supervoxel_sample_var_no / n_no) ** 2
  denominator = ( (supervoxel_sample_var_yes ** 2) / (n_yes ** 2 * (n_yes - 1)) ) + ( (supervoxel_sample_var_no ** 2) / (n_no ** 2 * (n_no - 1)) )
  df = numerator / denominator

  p_values_supervoxel = 2 * t.sf(np.abs(t_stat), df)

  # Computing the FDR correction

  mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold )
  p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
  # print("The number of clusters dropped is ",np.sum(np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold ))
  print("the Number of FDR-pvalue <0.05 is", np.sum(p_values_fdr_supervoxel<0.05))
  min_val = round(p_values_fdr_supervoxel.min(),3)
  display(HTML(f'<span style="color:red">Min value of FDR-pvalue is {min_val}</span>'))

  # Compute the p-value associated to the voxel knowing its cluster
  p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
  p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
  labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels - 1)

  # return mean_supervoxel_yes,mean_supervoxel_no
  return labels_pvalue,labels

# The means that will be used for the algorithm of the supervoxel
mean_survival=np.mean(doses_reduced_ratio_2[indices_yes], axis=0)
mean_non_survival= np.mean(doses_reduced_ratio_2[indices_no], axis=0)
diff_mean=np.abs( mean_non_survival - mean_survival)

np.random.seed(42)
for n in [50,100,150,200,500,1000]:
  for c in [0.001,0.01,0.1]:
    tmp=supervoxel_pvalues(n_clusters=n, compact=c,mean=diff_mean,min_size=0.5,max_size=3,
                    mean_non_survival=mean_non_survival,
                    mean_survival=mean_survival,doses_of_patients_4D=doses_reduced_ratio_2,threshold=0)


log_text = """The algorithm is running for n_clusters=50, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 20
the Number of FDR-pvalue <0.05 is 2
Min value of FDR-pvalue is 0.013
The algorithm is running for n_clusters=50, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 42
the Number of FDR-pvalue <0.05 is 4
Min value of FDR-pvalue is 0.019
The algorithm is running for n_clusters=50, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 46
the Number of FDR-pvalue <0.05 is 4
Min value of FDR-pvalue is 0.032
The algorithm is running for n_clusters=100, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 26
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.052
The algorithm is running for n_clusters=100, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 82
the Number of FDR-pvalue <0.05 is 4
Min value of FDR-pvalue is 0.041
The algorithm is running for n_clusters=100, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 97
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.159
The algorithm is running for n_clusters=150, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 37
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.146
The algorithm is running for n_clusters=150, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 100
the Number of FDR-pvalue <0.05 is 8
Min value of FDR-pvalue is 0.028
The algorithm is running for n_clusters=150, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 116
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.088
The algorithm is running for n_clusters=200, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 66
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.234
The algorithm is running for n_clusters=200, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 174
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.063
The algorithm is running for n_clusters=200, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 207
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.143
The algorithm is running for n_clusters=500, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 124
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.386
The algorithm is running for n_clusters=500, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 409
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.186
The algorithm is running for n_clusters=500, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 498
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.241
The algorithm is running for n_clusters=1000, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 251
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.846
The algorithm is running for n_clusters=1000, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 794
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.203
The algorithm is running for n_clusters=1000, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 948
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.352"""

# Regular expression to extract values
pattern = re.compile(
    r'n_clusters=(\d+), compactness=([\d.]+), min_size=([\d.]+), max_size=([\d.]+).*?'
    r'Number of cluster found is (\d+).*?'
    r'Number of FDR-pvalue <0.05 is (\d+).*?'
    r'Min value of FDR-pvalue is ([\d.]+)',
    re.DOTALL
)

# Extract matches
matches = pattern.findall(log_text)

# Convert to DataFrame
df = pd.DataFrame(matches, columns=[
    'n_clusters', 'compactness', 'min_size', 'max_size',
    'clusters_found', 'significant_pvals', 'min_fdr_pval'
])

# Convert data types
df = df.astype({
    'n_clusters': int,
    'compactness': float,
    'min_size': float,
    'max_size': float,
    'clusters_found': int,
    'significant_pvals': int,
    'min_fdr_pval': float
})

# Display the DataFrame
print(df)


# We compute the pvalues for those supervoxel
np.random.seed(42)

p_values_super,labels_supervox=supervoxel_pvalues(n_clusters=100, compact=0.01,mean=diff_mean,min_size=0.5,max_size=3,
                   mean_non_survival=mean_non_survival,
                   mean_survival=mean_survival,doses_of_patients_4D=doses_reduced_ratio_2,threshold=0)

print("The number of voxels belonging to the regions that are different is:",np.sum( p_values_super< 0.05))

# Since we worked on a masked version of the doses, we have to reconstruct the full 3D grid of pvalues to have the same dimension as our original dose
full_extended_super_pv=np.full(doses_scaled_ratio_2.shape[1:],fill_value=np.nan)
extended_labels_supervox=np.full(doses_scaled_ratio_2.shape[1:],fill_value=0)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super[z-21,y-41,x-40])< 0.05:
          full_extended_super_pv[z,y,x]=p_values_super[z-21,y-41,x-40]
          extended_labels_supervox[z,y,x]=labels_supervox[z-21,y-41,x-40]

label_39 = extended_labels_supervox == 39
label_42 = extended_labels_supervox == 42
label_46 = extended_labels_supervox == 46
label_47 = extended_labels_supervox == 47

import plotly.graph_objects as go


max_points = 100000
dims = full_extended_super_pv.shape
combined_mask = label_39 | label_42 | label_46 | label_47

# Keeping only the voxels in the labels
valid_idx = np.argwhere(combined_mask & ~np.isnan(full_extended_super_pv))  # [z, y, x]
valid_labels = extended_labels_supervox[combined_mask & ~np.isnan(full_extended_super_pv)]

# If too much points, we can sample them
if len(valid_idx) > max_points:
    idx_sample = np.random.choice(len(valid_idx), max_points, replace=False)
    valid_idx = valid_idx[idx_sample]
    valid_labels = valid_labels[idx_sample]

# [z, y, x] → [x, y, z]
valid_idx = valid_idx[:, [2, 1, 0]]

# Color for each label
label_to_color = {
    39: 'blue',
    42: 'green',
    46: 'orange',
    47: 'red'
}

traces = []
for label_value, color in label_to_color.items():
    label_mask = valid_labels == label_value
    if np.sum(label_mask) == 0:
        continue
    coords = valid_idx[label_mask]
    traces.append(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            name=f'Label {label_value}',
            marker=dict(size=2, color=color, opacity=0.8)
        )
    )

fig = go.Figure(data=traces)

# Config layout
fig.update_layout(
    title="3D plot of significant regions colored by label",
    scene=dict(
        xaxis=dict(title='X', range=[0, dims[2]]),
        yaxis=dict(title='Y', range=[0, dims[1]]),
        zaxis=dict(title='Z', range=[0, dims[0]])
    ),
    width=800,
    height=800,
    legend=dict(title='Labels', itemsizing='constant')
)

fig.show()


# # Supervoxel into the 4 clusters

# This function will return the full doses shape matrix with the p_value associated with the supervoxel algorithm
# It drops the supervoxels where the absolute difference between the supervoxel of the survivor and the non-survivor is less than the threshold
def supervoxel_pvalues_masked(n_clusters,mask,compact,mean,min_size=0.5,max_size=3,doses_of_patients_4D=doses_reduced_ratio_2,index_yes=indices_yes,index_no=indices_no,threshold=0,to_max=False):

  from skimage.segmentation import slic
  from skimage.util import img_as_float
  from scipy import ndimage
  from scipy.stats import t
  from scipy.stats import false_discovery_control
  from IPython.display import HTML, display

  print(f"The algorithm is running for n_clusters={n_clusters}, compactness={compact}, min_size={min_size}, max_size={max_size} with {doses_of_patients_4D.shape[0]} patients")

  # Computing the labels of the superclusters
  labels = slic(mean,max_num_iter=50, n_segments=n_clusters, compactness=compact,min_size_factor=min_size,max_size_factor=max_size,  start_label=1,channel_axis=None,enforce_connectivity=True,mask=mask)
  print("the Number of cluster found is", np.unique(labels).size-1)

  # Computing the supervoxels

  label_values = np.unique(labels)
  n_patients = doses_of_patients_4D.shape[0]
  n_labels = len(label_values)
  supervoxels = np.zeros((n_patients, n_labels))
  for j in range(n_patients):
    supervoxels[j] = ndimage.mean(doses_of_patients_4D[j], labels=labels, index=label_values)
  # Computing all the individuals t test

  mean_supervoxel_yes=np.mean(supervoxels[index_yes],axis=0)
  mean_supervoxel_no=np.mean(supervoxels[index_no],axis=0)
  supervoxel_sample_var_yes = np.var(supervoxels[index_yes], axis=0, ddof=1)
  supervoxel_sample_var_no = np.var(supervoxels[index_no], axis=0, ddof=1)

  n_yes = len(index_yes)
  n_no = len(index_no)
  t_stat = (mean_supervoxel_yes - mean_supervoxel_no) / np.sqrt(supervoxel_sample_var_no / n_no + supervoxel_sample_var_yes / n_yes)

  numerator = (supervoxel_sample_var_yes / n_yes + supervoxel_sample_var_no / n_no) ** 2
  denominator = ( (supervoxel_sample_var_yes ** 2) / (n_yes ** 2 * (n_yes - 1)) ) + ( (supervoxel_sample_var_no ** 2) / (n_no ** 2 * (n_no - 1)) )
  df = numerator / denominator



  p_values_supervoxel = 2 * t.sf(np.abs(t_stat), df)

  # Computing the FDR correction

  d= {} # to store all the values for each treshold
  best_threshold=0
  if to_max:
    for threshold in np.linspace(0, np.max(diff),1000): # we divide the interval [0, max value of the difference of the mean] in 1000 numbers

      mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold )
      mask[0]=True

      p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
      p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
      p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
      labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels - 1)
      d[threshold]=np.sum(labels_pvalue<0.05)
      if d[threshold]> d[best_threshold]:
        best_threshold=threshold

  # Compute the p-value associated to the voxel knowing its cluster

  mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<best_threshold )
  mask[0]=True
  p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
  print("the Number of FDR-pvalue <0.05 is", np.sum(p_values_fdr_supervoxel<0.05))
  min_val = round(p_values_fdr_supervoxel.min(),3)
  display(HTML(f'<span style="color:red">Min value of FDR-pvalue is {min_val}</span>'))



  # Compute the p-value associated to the voxel knowing its cluster
  p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
  p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
  labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels )
  return d,labels_pvalue,labels

for n_clust in [8380]:
  for c in [0.01]:
    min_size=4*n_clust/np.sum(labels_supervox==47)
    max_size=4*n_clust/np.sum(labels_supervox==47)
    d,p_values_super_2,l=supervoxel_pvalues_masked(n_clust,labels_supervox==47,compact=c,mean=diff_mean,min_size=min_size,max_size=max_size)

import plotly.graph_objects as go

# --- Parameter : number of max points not significatifs to show ---
max_points_nonsig = 20000

# --- Masks ---
mask_significant = (p_values_super_2 < 0.05) & (~np.isnan(p_values_super_2))
mask_nonsignificant = (p_values_super < 0.05) & (~np.isnan(p_values_super_2)) & (p_values_super_2 >= 0.05)
coords_sig = np.argwhere(mask_significant)        # [z, y, x]
coords_nonsig = np.argwhere(mask_nonsignificant)  # [z, y, x]

# --- Random sampling of points to show ---
if len(coords_nonsig) > max_points_nonsig:
    idx_sample = np.random.choice(len(coords_nonsig), size=max_points_nonsig, replace=False)
    coords_nonsig = coords_nonsig[idx_sample]

# --- [z, y, x] → [x, y, z] ---
coords_sig_xyz = coords_sig[:, [2, 1, 0]]
coords_nonsig_xyz = coords_nonsig[:, [2, 1, 0]]

# --- Plotly ---
traces = [
    go.Scatter3d(
        x=coords_sig_xyz[:, 0],
        y=coords_sig_xyz[:, 1],
        z=coords_sig_xyz[:, 2],
        mode='markers',
        name='p < 0.05',
        marker=dict(size=2, color='red', opacity=0.8)
    ),
    go.Scatter3d(
        x=coords_nonsig_xyz[:, 0],
        y=coords_nonsig_xyz[:, 1],
        z=coords_nonsig_xyz[:, 2],
        mode='markers',
        name='p ≥ 0.05',
        marker=dict(size=2, color='blue', opacity=0.4)
    )
]

# --- Layout ---
layout = go.Layout(
    title='Visualisation 3D : p < 0.05 en rouge, sinon en bleu (échantillonné)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Affichage ---
fig = go.Figure(data=traces, layout=layout)
fig.show()


extended_p_values_super_2=np.full(doses_scaled_ratio_2.shape[1:],fill_value=np.nan)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super_2[z-21,y-41,x-40])< 0.05:
          extended_p_values_super_2[z,y,x]=p_values_super_2[z-21,y-41,x-40]

np.save("drive/MyDrive/extended_p_values_super_ratio_2",extended_p_values_super_2)

import os
os.remove("drive/MyDrive/doses_scaled_ratio_2.npy")

################################################################################
# Part 3: Analyzing Dose Ratios (Version ratio=5)
################################################################################


index_to_remove_mim= [44,87,112,131,250]
doses_memmap = np.load("drive/MyDrive/initial_array_doses.npy", mmap_mode='r')
d = df_aligned["Number of Fractions"].values

keep_indices = [i for i in range(doses_memmap.shape[0]) if i not in index_to_remove_mim]
shape = (len(keep_indices),) + doses_memmap.shape[1:]

ratio = 5

# Output file
output_path = "drive/MyDrive/doses_scaled_ratio_5.npy"
doses_scaled = np.lib.format.open_memmap(output_path, mode='w+', dtype=doses_memmap.dtype, shape=shape)

# Creating the final array
output_index = 0
for i in keep_indices:
    scaled = doses_memmap[i] * (doses_memmap[i] / d[output_index] + ratio) / (2 + ratio) # We compute the formula for each voxel based on the number of fractions
    doses_scaled[output_index] = np.nan_to_num(scaled, nan=0.0)  # We replace the NA with 0
    output_index += 1


## Data Vizualisation Survival and No Survival

# Import the doses scaled and compute the doses reduced focusing only on this square part
doses_scaled_ratio_5=np.load("drive/MyDrive/doses_scaled_ratio_5.npy", mmap_mode='r')
doses_reduced_ratio_5=doses_scaled_ratio_5[:,20:146,40:140,40:180]

# Compute the indices to have the two groups
overall_survival=df_aligned["Overall Survival at 2 years"]
indices_yes = np.where(overall_survival == 'YES')[0]
indices_no = np.where(overall_survival == 'NO')[0]

# Compute the mean for the axis 0 (patient axis) for the two groups
mean_yes = np.mean(doses_scaled_ratio_5[indices_yes], axis=0)
mean_no = np.mean(doses_scaled_ratio_5[indices_no], axis=0)

## Vizualisation slices survival vs non survival

# Plot at a slice level the mean doses for survival, no-survival and their differences
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

def plot_slice(i):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # --- VMIN/VMAX for the common scal
    vmin_common = min(mean_yes[i].min(), mean_no[i].min())
    vmax_common = max(mean_yes[i].max(), mean_no[i].max())

    # 1. YES
    im1 = axes[0].imshow(mean_yes[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[0].set_title(f'Survival YES - Slice Z={i}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], shrink=0.7)

    # 2. NO
    im2 = axes[1].imshow(mean_no[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(f'Survival NO - Slice Z={i}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], shrink=0.7)

    # 3. DIFF (YES - NO)
    diff = mean_yes[i] - mean_no[i]
    max_abs = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap='bwr', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title(f'DIFF (YES - NO) - Slice Z={i}')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    plt.show()

widgets.interact(plot_slice, i=(0, mean_yes.shape[0] - 1));


# Same plot with the mask to show what we will not consider in the further code
import matplotlib.patches as patches

# mask of indices to not consider
noise_mask = np.ones((147, 168, 201), dtype=bool)
noise_mask[20:146,40:140,40:180] = False

diff = mean_yes - mean_no
threshold = 0
diff[np.abs(diff) < threshold] = 0

def plot_slice(i):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Scaling of the barcolor
    vmin_common = min(mean_yes[i].min(), mean_no[i].min())
    vmax_common = max(mean_yes[i].max(), mean_no[i].max())

    # YES
    im1 = axes[0].imshow(mean_yes[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[0].set_title(f'Survival YES - Slice Z={i}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], shrink=0.7)

    y_coords, x_coords = np.where(noise_mask[i])
    axes[0].scatter(x_coords, y_coords, color='red', s=5)

    # NO
    im2 = axes[1].imshow(mean_no[i], cmap='viridis', vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(f'NO Survival - Slice Z={i}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], shrink=0.7)
    axes[1].scatter(x_coords, y_coords, color='red', s=5)

    # DIFF
    max_abs = np.max(np.abs(diff[i]))
    im3 = axes[2].imshow(diff[i], cmap='bwr', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title(f'DIFF (YES - NO) - Slice Z={i}')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], shrink=0.7)

    # Black Square corresponds to the red square limit
    y_coords, x_coords = np.where(~noise_mask[i])
    if np.any(~noise_mask[i]):
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        axes[2].add_patch(rect)

    plt.tight_layout()
    plt.show()

widgets.interact(plot_slice, i=(0, mean_yes.shape[0] - 1));


# ## Statistics on 3D images of Mean of two groups (mean_yes/no)

number_of_0=[]
mean_image_3D=[]
for i in range(doses_scaled_ratio_5.shape[0]):
  mask=doses_scaled_ratio_5[i]==0
  number_of_0.append(doses_scaled_ratio_5[i][mask].size /  (147* 168* 201) )
  mean_image_3D.append(doses_scaled_ratio_5[i][~mask].mean())

number_of_0 = np.array(number_of_0)
mean_image_3D = np.array(mean_image_3D)

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# --- Boxplots
axes[0, 0].boxplot([number_of_0[indices_yes], number_of_0[indices_no]],
                   labels=['YES', 'NO'], patch_artist=True)
axes[0, 0].set_title('% of voxels = 0')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].grid(True, linestyle='--', alpha=0.3)

axes[0, 1].boxplot([mean_image_3D[indices_yes], mean_image_3D[indices_no]],
                   labels=['YES', 'NO'], patch_artist=True)
axes[0, 1].set_title('Mean voxels ≠ 0')
axes[0, 1].set_ylabel('Mean Dose')
axes[0, 1].grid(True, linestyle='--', alpha=0.3)

# --- Histograms for % of 0
bins_0 = np.linspace(np.min(number_of_0), np.max(number_of_0), 30)
hist_yes_0 = sns.histplot(number_of_0[indices_yes], bins=bins_0, ax=axes[1, 0],
                          color='blue', kde=False, stat='probability')
hist_no_0 = sns.histplot(number_of_0[indices_no], bins=bins_0, ax=axes[1, 1],
                         color='orange', kde=False, stat='probability')

# Same scale of Y
max_y_0 = max(hist_yes_0.get_ylim()[1], hist_no_0.get_ylim()[1])
axes[1, 0].set_ylim(0, max_y_0)
axes[1, 1].set_ylim(0, max_y_0)

axes[1, 0].set_title('Histogram % voxels = 0 (YES)')
axes[1, 0].set_xlabel('Proportion of 0')
axes[1, 0].set_ylabel('Porcentage')
axes[1, 0].grid(True, linestyle='--', alpha=0.3)

axes[1, 1].set_title('Histogram % voxels = 0 (NO)')
axes[1, 1].set_xlabel('Proportion of 0')
axes[1, 1].set_ylabel('Porcentage')
axes[1, 1].grid(True, linestyle='--', alpha=0.3)

# --- Histograms for means ≠ 0
bins_mean = np.linspace(np.min(mean_image_3D), np.max(mean_image_3D), 30)
hist_yes_mean = sns.histplot(mean_image_3D[indices_yes], bins=bins_mean, ax=axes[2, 0],
                             color='blue', kde=False, stat='probability')
hist_no_mean = sns.histplot(mean_image_3D[indices_no], bins=bins_mean, ax=axes[2, 1],
                            color='orange', kde=False, stat='probability')

# Same scale of Y
max_y_mean = max(hist_yes_mean.get_ylim()[1], hist_no_mean.get_ylim()[1])
axes[2, 0].set_ylim(0, max_y_mean)
axes[2, 1].set_ylim(0, max_y_mean)

axes[2, 0].set_title('Histogram mean voxels ≠ 0 (YES)')
axes[2, 0].set_xlabel('Mean Dose')
axes[2, 0].set_ylabel('Porcentage')
axes[2, 0].grid(True, linestyle='--', alpha=0.3)

axes[2, 1].set_title('Histogram mean voxels ≠ 0 (NO)')
axes[2, 1].set_xlabel('Mean dose')
axes[2, 1].set_ylabel('Porcentage')
axes[2, 1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


print(f"Percentage of values= 0 in mean_yes : {round(np.mean(mean_yes == 0)*100,2)} %")
print(f"Percentage of values= 0 in mean_no : {round(np.mean(mean_no == 0)*100,2)} %")

print("mean_values for survivor",np.mean(mean_yes))
print("mean_values for non-survivor",np.mean(mean_no))
print("mean_values for values different than 0 for survivor",np.mean(mean_yes[mean_yes>0]))
print("mean_values for values different than 0 for non-survivor",np.mean(mean_no[mean_no>0]))

# # Optimization based on ratio 2 hyperparameters

# ## Supervoxel

# This function will return the full doses shape matrix with the p_value associated with the supervoxel algorithm
# It drops the supervoxels where the absolute difference between the supervoxel of the survivor and the non-survivor is less than the threshold
def supervoxel_pvalues(n_clusters,compact,mean,min_size=0.5,max_size=3,mean_non_survival=mean_no,mean_survival=mean_yes,doses_of_patients_4D=doses_reduced_ratio_5,index_yes=indices_yes,index_no=indices_no,threshold=0):

  from skimage.segmentation import slic
  from skimage.util import img_as_float
  from scipy import ndimage
  from scipy.stats import t
  from scipy.stats import false_discovery_control
  from IPython.display import HTML, display

  print(f"The algorithm is running for n_clusters={n_clusters}, compactness={compact}, min_size={min_size}, max_size={max_size} with {doses_of_patients_4D.shape[0]} patients")

  # Computing the labels of the superclusters
  labels = slic(mean,max_num_iter=50, n_segments=n_clusters, compactness=compact,min_size_factor=min_size,max_size_factor=max_size,  start_label=1,channel_axis=None,enforce_connectivity=True)
  print("the Number of cluster found is", np.unique(labels).size)

  # Computing the supervoxels

  label_values = np.unique(labels)
  n_patients = doses_of_patients_4D.shape[0]
  n_labels = len(label_values)
  supervoxels = np.zeros((n_patients, n_labels))
  for j in range(n_patients):
    supervoxels[j] = ndimage.mean(doses_of_patients_4D[j], labels=labels, index=label_values)

  # Computing all the individuals t test

  mean_supervoxel_yes=np.mean(supervoxels[index_yes],axis=0)
  mean_supervoxel_no=np.mean(supervoxels[index_no],axis=0)
  supervoxel_sample_var_yes = np.var(supervoxels[index_yes], axis=0, ddof=1)
  supervoxel_sample_var_no = np.var(supervoxels[index_no], axis=0, ddof=1)

  n_yes = len(index_yes)
  n_no = len(index_no)
  t_stat = (mean_supervoxel_yes - mean_supervoxel_no) / np.sqrt(supervoxel_sample_var_no / n_no + supervoxel_sample_var_yes / n_yes)

  numerator = (supervoxel_sample_var_yes / n_yes + supervoxel_sample_var_no / n_no) ** 2
  denominator = ( (supervoxel_sample_var_yes ** 2) / (n_yes ** 2 * (n_yes - 1)) ) + ( (supervoxel_sample_var_no ** 2) / (n_no ** 2 * (n_no - 1)) )
  df = numerator / denominator

  p_values_supervoxel = 2 * t.sf(np.abs(t_stat), df)

  # Computing the FDR correction

  mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold )
  p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
  # print("The number of clusters dropped is ",np.sum(np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold ))
  print("the Number of FDR-pvalue <0.05 is", np.sum(p_values_fdr_supervoxel<0.05))
  min_val = round(p_values_fdr_supervoxel.min(),3)
  display(HTML(f'<span style="color:red">Min value of FDR-pvalue is {min_val}</span>'))

  # Compute the p-value associated to the voxel knowing its cluster
  p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
  p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
  labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels - 1)

  # return mean_supervoxel_yes,mean_supervoxel_no
  return labels_pvalue,labels

# The means that will be used for the algorithm of the supervoxel
mean_survival=np.mean(doses_reduced_ratio_5[indices_yes], axis=0)
mean_non_survival= np.mean(doses_reduced_ratio_5[indices_no], axis=0)
diff_mean=np.abs( mean_non_survival - mean_survival)

# We compute the pvalues for those supervoxel
np.random.seed(42)

p_values_super,labels_supervox=supervoxel_pvalues(n_clusters=100, compact=0.01,mean=diff_mean,min_size=0.5,max_size=3,
                   mean_non_survival=mean_non_survival,
                   mean_survival=mean_survival,doses_of_patients_4D=doses_reduced_ratio_5,threshold=0)

print("The number of voxels belonging to the regions that are different is:",np.sum( p_values_super< 0.05))

# Since we worked on a masked version of the doses, we have to reconstruct the full 3D grid of pvalues to have the same dimension as our original dose
full_extended_super_pv=np.full(doses_scaled_ratio_5.shape[1:],fill_value=np.nan)
extended_labels_supervox=np.full(doses_scaled_ratio_5.shape[1:],fill_value=0)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super[z-21,y-41,x-40])< 0.05:
          full_extended_super_pv[z,y,x]=p_values_super[z-21,y-41,x-40]
          extended_labels_supervox[z,y,x]=labels_supervox[z-21,y-41,x-40]

np.unique(extended_labels_supervox)

label_28 = extended_labels_supervox == 28
label_37 = extended_labels_supervox == 37
label_39 = extended_labels_supervox == 39
label_43 = extended_labels_supervox == 43
label_44 = extended_labels_supervox == 44
label_46 = extended_labels_supervox == 46
label_47 = extended_labels_supervox == 47
label_48 = extended_labels_supervox == 48

import plotly.graph_objects as go

max_points = 100000
dims = full_extended_super_pv.shape

# Nouveau combined_mask basé sur les labels que tu as définis
combined_mask = (
    label_28 | label_37 | label_39 | label_43 | label_44 | label_46 | label_47 | label_48
)

# On ne conserve que les voxels valides et non NaN
valid_idx = np.argwhere(combined_mask & ~np.isnan(full_extended_super_pv))  # [z, y, x]
valid_labels = extended_labels_supervox[combined_mask & ~np.isnan(full_extended_super_pv)]

# Échantillonnage si trop de points
if len(valid_idx) > max_points:
    idx_sample = np.random.choice(len(valid_idx), max_points, replace=False)
    valid_idx = valid_idx[idx_sample]
    valid_labels = valid_labels[idx_sample]

# Passage [z, y, x] → [x, y, z]
valid_idx = valid_idx[:, [2, 1, 0]]

# Couleurs pour chaque label, à adapter si tu veux
label_to_color = {
    28: 'purple',
    37: 'cyan',
    39: 'blue',
    43: 'lime',
    44: 'yellow',
    46: 'orange',
    47: 'red',
    48: 'magenta'
}

traces = []
for label_value, color in label_to_color.items():
    label_mask = valid_labels == label_value
    if np.sum(label_mask) == 0:
        continue
    coords = valid_idx[label_mask]
    traces.append(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            name=f'Label {label_value}',
            marker=dict(size=2, color=color, opacity=0.8)
        )
    )

fig = go.Figure(data=traces)

fig.update_layout(
    title="3D plot of significant regions colored by label",
    scene=dict(
        xaxis=dict(title='X', range=[0, dims[2]]),
        yaxis=dict(title='Y', range=[0, dims[1]]),
        zaxis=dict(title='Z', range=[0, dims[0]])
    ),
    width=800,
    height=800,
    legend=dict(title='Labels', itemsizing='constant')
)

fig.show()


# ## Supervoxel into the 4 clusters

# This function will return the full doses shape matrix with the p_value associated with the supervoxel algorithm
# It drops the supervoxels where the absolute difference between the supervoxel of the survivor and the non-survivor is less than the threshold
def supervoxel_pvalues_masked(n_clusters,mask,compact,mean,min_size=0.5,max_size=3,doses_of_patients_4D=doses_reduced_ratio_5,index_yes=indices_yes,index_no=indices_no,threshold=0,to_max=False):

  import numpy as np
  from skimage.segmentation import slic
  from skimage.util import img_as_float
  from scipy import ndimage
  from scipy.stats import t
  from scipy.stats import false_discovery_control
  from IPython.display import HTML, display

  print(f"The algorithm is running for n_clusters={n_clusters}, compactness={compact}, min_size={min_size}, max_size={max_size} with {doses_of_patients_4D.shape[0]} patients")

  # Computing the labels of the superclusters
  labels = slic(mean,max_num_iter=50, n_segments=n_clusters, compactness=compact,min_size_factor=min_size,max_size_factor=max_size,  start_label=1,channel_axis=None,enforce_connectivity=True,mask=mask)
  print("the Number of cluster found is", np.unique(labels).size-1)

  # Computing the supervoxels

  label_values = np.unique(labels)
  n_patients = doses_of_patients_4D.shape[0]
  n_labels = len(label_values)
  supervoxels = np.zeros((n_patients, n_labels))
  for j in range(n_patients):
    supervoxels[j] = ndimage.mean(doses_of_patients_4D[j], labels=labels, index=label_values)
  # Computing all the individuals t test

  mean_supervoxel_yes=np.mean(supervoxels[index_yes],axis=0)
  mean_supervoxel_no=np.mean(supervoxels[index_no],axis=0)
  supervoxel_sample_var_yes = np.var(supervoxels[index_yes], axis=0, ddof=1)
  supervoxel_sample_var_no = np.var(supervoxels[index_no], axis=0, ddof=1)

  n_yes = len(index_yes)
  n_no = len(index_no)
  t_stat = (mean_supervoxel_yes - mean_supervoxel_no) / np.sqrt(supervoxel_sample_var_no / n_no + supervoxel_sample_var_yes / n_yes)

  numerator = (supervoxel_sample_var_yes / n_yes + supervoxel_sample_var_no / n_no) ** 2
  denominator = ( (supervoxel_sample_var_yes ** 2) / (n_yes ** 2 * (n_yes - 1)) ) + ( (supervoxel_sample_var_no ** 2) / (n_no ** 2 * (n_no - 1)) )
  df = numerator / denominator



  p_values_supervoxel = 2 * t.sf(np.abs(t_stat), df)

  # Computing the FDR correction

  d= {} # to store all the values for each treshold
  best_threshold=0
  if to_max:
    for threshold in np.linspace(0, np.max(diff),1000): # we divide the interval [0, max value of the difference of the mean] in 1000 numbers

      mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<threshold )
      mask[0]=True

      p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
      p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
      p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
      labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels - 1)
      d[threshold]=np.sum(labels_pvalue<0.05)
      if d[threshold]> d[best_threshold]:
        best_threshold=threshold

  # Compute the p-value associated to the voxel knowing its cluster

  mask=np.isnan(p_values_supervoxel) |  (np.abs(mean_supervoxel_yes - mean_supervoxel_no)<best_threshold )
  mask[0]=True
  p_values_fdr_supervoxel=false_discovery_control(p_values_supervoxel[~mask], method='bh')
  print("the Number of FDR-pvalue <0.05 is", np.sum(p_values_fdr_supervoxel<0.05))
  min_val = round(p_values_fdr_supervoxel.min(),3)
  display(HTML(f'<span style="color:red">Min value of FDR-pvalue is {min_val}</span>'))



  # Compute the p-value associated to the voxel knowing its cluster
  p_values_fdr_supervoxel_with_NA= np.ones_like(p_values_supervoxel)
  p_values_fdr_supervoxel_with_NA[~mask]=p_values_fdr_supervoxel
  labels_pvalue = np.take(p_values_fdr_supervoxel_with_NA, labels )
  return d,labels_pvalue,labels

for n_clust in [8380]:
  for c in [0.01]:
    min_size=4*n_clust/np.sum(labels_supervox==48)
    max_size=4*n_clust/np.sum(labels_supervox==48)
    d,p_values_super_2,l=supervoxel_pvalues_masked(n_clust,labels_supervox==47,compact=c,mean=diff_mean,min_size=min_size,max_size=max_size)

import plotly.graph_objects as go

# --- Parameter : number of max points not significatifs to show ---
max_points_nonsig = 50000

# --- Masks ---
mask_significant = (p_values_super_2 < 0.05) & (~np.isnan(p_values_super_2))
mask_nonsignificant = (p_values_super < 0.05) & (~np.isnan(p_values_super_2)) & (p_values_super_2 >= 0.05)
coords_sig = np.argwhere(mask_significant)        # [z, y, x]
coords_nonsig = np.argwhere(mask_nonsignificant)  # [z, y, x]

# --- Random sampling of points to show ---
if len(coords_nonsig) > max_points_nonsig:
    idx_sample = np.random.choice(len(coords_nonsig), size=max_points_nonsig, replace=False)
    coords_nonsig = coords_nonsig[idx_sample]

# --- [z, y, x] → [x, y, z] ---
coords_sig_xyz = coords_sig[:, [2, 1, 0]]
coords_nonsig_xyz = coords_nonsig[:, [2, 1, 0]]

# --- Plotly ---
traces = [
    go.Scatter3d(
        x=coords_sig_xyz[:, 0],
        y=coords_sig_xyz[:, 1],
        z=coords_sig_xyz[:, 2],
        mode='markers',
        name='p < 0.05',
        marker=dict(size=2, color='red', opacity=0.8)
    ),
    go.Scatter3d(
        x=coords_nonsig_xyz[:, 0],
        y=coords_nonsig_xyz[:, 1],
        z=coords_nonsig_xyz[:, 2],
        mode='markers',
        name='p ≥ 0.05',
        marker=dict(size=2, color='blue', opacity=0.4)
    )
]

# --- Layout ---
layout = go.Layout(
    title='Visualisation 3D : p < 0.05 en rouge, sinon en bleu (échantillonné)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Affichage ---
fig = go.Figure(data=traces, layout=layout)
fig.show()


extended_p_values_super_2=np.full(doses_scaled_ratio_5.shape[1:],fill_value=np.nan)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super_2[z-21,y-41,x-40])< 0.05:
          extended_p_values_super_2[z,y,x]=p_values_super_2[z-21,y-41,x-40]

np.save("drive/MyDrive/extended_p_values_super_ratio_5",extended_p_values_super_2)

# # Optimization based on ratio 2 hyperparameter

# ## Supervoxel

# np.random.seed(42)
# for n in [50,100,150,200,500]:
#   for c in [0.001,0.01,0.1]:
#     tmp=supervoxel_pvalues(n_clusters=n, compact=c,mean=diff_mean,min_size=0.5,max_size=3,
#                     mean_non_survival=mean_non_survival,
#                     mean_survival=mean_survival,doses_of_patients_4D=doses_reduced_ratio_5,threshold=0)


texte = """
The algorithm is running for n_clusters=50, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 17
the Number of FDR-pvalue <0.05 is 2
Min value of FDR-pvalue is 0.013
The algorithm is running for n_clusters=50, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 35
the Number of FDR-pvalue <0.05 is 5
Min value of FDR-pvalue is 0.02
The algorithm is running for n_clusters=50, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 46
the Number of FDR-pvalue <0.05 is 3
Min value of FDR-pvalue is 0.039
The algorithm is running for n_clusters=100, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 27
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.089
The algorithm is running for n_clusters=100, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 79
the Number of FDR-pvalue <0.05 is 8
Min value of FDR-pvalue is 0.031
The algorithm is running for n_clusters=100, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 96
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.111
The algorithm is running for n_clusters=150, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 33
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.153
The algorithm is running for n_clusters=150, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 91
the Number of FDR-pvalue <0.05 is 6
Min value of FDR-pvalue is 0.035
The algorithm is running for n_clusters=150, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 116
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.089
The algorithm is running for n_clusters=200, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 60
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.205
The algorithm is running for n_clusters=200, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 172
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.082
The algorithm is running for n_clusters=200, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 206
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.177
The algorithm is running for n_clusters=500, compactness=0.001, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 123
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.839
The algorithm is running for n_clusters=500, compactness=0.01, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 390
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.226
The algorithm is running for n_clusters=500, compactness=0.1, min_size=0.5, max_size=3 with 310 patients
the Number of cluster found is 501
the Number of FDR-pvalue <0.05 is 0
Min value of FDR-pvalue is 0.239
"""

# On split le texte en blocs de 4 lignes
blocs = texte.strip().split("\n\n")

data = []
lines = texte.strip().split('\n')
for i in range(0, len(lines), 4):
    # Extraire les paramètres depuis la première ligne
    line1 = lines[i]
    n_clusters = int(re.search(r"n_clusters=(\d+)", line1).group(1))
    compactness = float(re.search(r"compactness=([\d\.]+)", line1).group(1))
    min_size = float(re.search(r"min_size=([\d\.]+)", line1).group(1))
    max_size = float(re.search(r"max_size=([\d\.]+)", line1).group(1))
    patients = int(re.search(r"with (\d+) patients", line1).group(1))

    # Extraire les valeurs numériques des autres lignes
    clusters_found = int(re.search(r"Number of cluster found is (\d+)", lines[i+1]).group(1))
    fdr_less_005 = int(re.search(r"Number of FDR-pvalue <0.05 is (\d+)", lines[i+2]).group(1))
    min_fdr = float(re.search(r"Min value of FDR-pvalue is ([\d\.]+)", lines[i+3]).group(1))

    data.append({
        "n_clusters": n_clusters,
        "compactness": compactness,
        "min_size": min_size,
        "max_size": max_size,
        "patients": patients,
        "clusters_found": clusters_found,
        "fdr_less_005": fdr_less_005,
        "min_fdr": min_fdr
    })

df = pd.DataFrame(data)
print(df)


# We compute the pvalues for those supervoxel
np.random.seed(42)
p_values_super_ratio_5,labels_supervox_ratio_5=supervoxel_pvalues(n_clusters=150, compact=0.01,mean=diff_mean,min_size=0.5,max_size=3,
                   mean_non_survival=mean_non_survival,
                   mean_survival=mean_survival,doses_of_patients_4D=doses_reduced_ratio_5,threshold=0)

print("The number of voxels belonging to the regions that are different is:",np.sum( p_values_super< 0.05))

# Since we worked on a masked version of the doses, we have to reconstruct the full 3D grid of pvalues to have the same dimension as our original dose
full_extended_super_pv_ratio_5=np.full(doses_scaled_ratio_5.shape[1:],fill_value=np.nan)
extended_labels_supervox_ratio_5=np.full(doses_scaled_ratio_5.shape[1:],fill_value=0)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super[z-21,y-41,x-40])< 0.05:
          full_extended_super_pv_ratio_5[z,y,x]=p_values_super[z-21,y-41,x-40]
          extended_labels_supervox_ratio_5[z,y,x]=labels_supervox[z-21,y-41,x-40]

np.unique(extended_labels_supervox_ratio_5)

label_43_ratio_5 = extended_labels_supervox_ratio_5 == 43
label_46_ratio_5 = extended_labels_supervox_ratio_5 == 46
label_47_ratio_5 = extended_labels_supervox_ratio_5 == 47
label_49_ratio_5 = extended_labels_supervox_ratio_5 == 49
label_51_ratio_5 = extended_labels_supervox_ratio_5 == 51
label_53_ratio_5 = extended_labels_supervox_ratio_5 == 53

import plotly.graph_objects as go

max_points = 100000
dims = full_extended_super_pv_ratio_5.shape

combined_mask = label_43_ratio_5 | label_46_ratio_5 | label_47_ratio_5 | label_49_ratio_5 | label_51_ratio_5 | label_53_ratio_5

valid_idx = np.argwhere(combined_mask & ~np.isnan(full_extended_super_pv_ratio_5))  # [z, y, x]
valid_labels = extended_labels_supervox_ratio_5[combined_mask & ~np.isnan(full_extended_super_pv_ratio_5)]

# Échantillonnage si trop de points
if len(valid_idx) > max_points:
    idx_sample = np.random.choice(len(valid_idx), max_points, replace=False)
    valid_idx = valid_idx[idx_sample]
    valid_labels = valid_labels[idx_sample]

valid_idx = valid_idx[:, [2, 1, 0]]

label_to_color = {
    43: 'lime',
    46: 'orange',
    47: 'red',
    49: 'blue',
    51: 'purple',
    53: 'magenta'
}

traces = []
for label_value, color in label_to_color.items():
    label_mask = valid_labels == label_value
    if np.sum(label_mask) == 0:
        continue
    coords = valid_idx[label_mask]
    traces.append(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            name=f'Label {label_value}',
            marker=dict(size=2, color=color, opacity=0.8)
        )
    )

fig = go.Figure(data=traces)

fig.update_layout(
    title="3D plot of significant regions colored by label",
    scene=dict(
        xaxis=dict(title='X', range=[0, dims[2]]),
        yaxis=dict(title='Y', range=[0, dims[1]]),
        zaxis=dict(title='Z', range=[0, dims[0]])
    ),
    width=800,
    height=800,
    legend=dict(title='Labels', itemsizing='constant')
)

fig.show()


# ## Supervoxel into the 4 clusters

for n_clust in [5000]:
  for c in [0.01]:
    min_size=4*n_clust/np.sum(labels_supervox_ratio_5==53)
    max_size=4*n_clust/np.sum(labels_supervox_ratio_5==53)
    d_ratio_5,p_values_super_2_ratio_5,l_ratio_5=supervoxel_pvalues_masked(n_clust,labels_supervox_ratio_5==53,compact=c,mean=diff_mean,min_size=min_size,max_size=max_size)

np.sum(p_values_super_2_ratio_5<0.05)

import numpy as np
import plotly.graph_objects as go

# --- Parameter : number of max points not significatifs to show ---
max_points_nonsig = 50000

# --- Masks ---
mask_significant = (p_values_super_2_ratio_5 < 0.05) & (~np.isnan(p_values_super_2_ratio_5))
mask_nonsignificant = (p_values_super_ratio_5 < 0.05) & (~np.isnan(p_values_super_2_ratio_5)) & (p_values_super_2_ratio_5 >= 0.05)
coords_sig = np.argwhere(mask_significant)        # [z, y, x]
coords_nonsig = np.argwhere(mask_nonsignificant)  # [z, y, x]

# --- Random sampling of points to show ---
if len(coords_nonsig) > max_points_nonsig:
    idx_sample = np.random.choice(len(coords_nonsig), size=max_points_nonsig, replace=False)
    coords_nonsig = coords_nonsig[idx_sample]

# --- [z, y, x] → [x, y, z] ---
coords_sig_xyz = coords_sig[:, [2, 1, 0]]
coords_nonsig_xyz = coords_nonsig[:, [2, 1, 0]]

# --- Plotly ---
traces = [
    go.Scatter3d(
        x=coords_sig_xyz[:, 0],
        y=coords_sig_xyz[:, 1],
        z=coords_sig_xyz[:, 2],
        mode='markers',
        name='p < 0.05',
        marker=dict(size=2, color='red', opacity=0.8)
    ),
    go.Scatter3d(
        x=coords_nonsig_xyz[:, 0],
        y=coords_nonsig_xyz[:, 1],
        z=coords_nonsig_xyz[:, 2],
        mode='markers',
        name='p ≥ 0.05',
        marker=dict(size=2, color='blue', opacity=0.4)
    )
]

# --- Layout ---
layout = go.Layout(
    title='Visualisation 3D : p < 0.05 en rouge, sinon en bleu (échantillonné)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Affichage ---
fig = go.Figure(data=traces, layout=layout)
fig.show()


extended_p_values_super_2_ratio_5=np.full(doses_scaled_ratio_5.shape[1:],fill_value=np.nan)
for z in range(147):
  for y in range(168):
    for x in range(201):
      if not ((z <20) or (z>=146) or (y<40) or (y>=140) or (x<40) or (x>=180)):
        if (p_values_super_2_ratio_5[z-21,y-41,x-40])< 0.05:
          extended_p_values_super_2_ratio_5[z,y,x]=p_values_super_2_ratio_5[z-21,y-41,x-40]

np.save("drive/MyDrive/extended_p_values_super_ratio_5_optimized",extended_p_values_super_2_ratio_5)

import os
os.remove("drive/MyDrive/doses_scaled_ratio_5.npy")

################################################################################
# Notebook 4: Handling Voxel Outliers
################################################################################

# # Notebook 4: Handling Voxel Outliers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# # Importing the data

df = pd.read_csv("drive/MyDrive/AS_Project/data_reindexed.csv")

doses_scaled = np.load("drive/MyDrive/AS_Project/doses_scaled.npy", mmap_mode='r')

print(type(doses_scaled))


doses_scaled.shape

# ## Outliers

print(f"Min dose : {doses_scaled.min():.2f}")
print(f"Max dose : {doses_scaled.max():.2f}")
print(f"Mean dose (1st patient) : {doses_scaled[0].mean():.2f}")


max_dose_per_patient = np.array([doses_scaled[i].max() for i in range(doses_scaled.shape[0])])
plt.hist(max_dose_per_patient,bins=30)
plt.show()

filtered_max_doses = max_dose_per_patient[(max_dose_per_patient > 475) & (max_dose_per_patient < 565)]
plt.hist(filtered_max_doses,bins=30)
plt.show()

print(f"Number of patients kept after filtering by maximum dose of each patient : {len(filtered_max_doses)}")

index_max_outliers = np.where((max_dose_per_patient <= 475) | (max_dose_per_patient >= 565))[0]
print("List of patient's index outliers based on maximum dose of each patient :\n")
index_max_outliers

for i in index_max_outliers:
  print(f"Patient {i} : {max_dose_per_patient[i]} Gy")

mean_dose_per_patient = np.array([doses_scaled[i].mean() for i in range(doses_scaled.shape[0])])
plt.hist(mean_dose_per_patient,bins=30)
plt.show()

filtered_mean_doses = mean_dose_per_patient[mean_dose_per_patient <= 42]
plt.hist(filtered_mean_doses,bins=20)
plt.show()

print(f"Number of patients kept after filtering by mean dose of each patient : {len(filtered_mean_doses)}")

index_mean_outliers = np.where(mean_dose_per_patient > 42)[0]
print("List of patient's index outliers based on mean dose of each patient :\n")
index_mean_outliers

for i in index_mean_outliers:
  print(f"Patient {i} : {mean_dose_per_patient[i]} Gy")

index_common_outliers = np.intersect1d(index_max_outliers, index_mean_outliers)
print("List of patient's index outliers based on maximum dose and mean dose of each patient :\n")
index_common_outliers

import matplotlib.pyplot as plt

plt.hist(doses_scaled[0].flatten(), bins=100)
plt.title("Doses distribution (patient 0)")
plt.xlabel("Dose")
plt.ylabel("Numbers of voxels")
plt.yscale("log")
plt.grid(True)
plt.show()


def compute_dose_descriptors(doses_scaled, voxel_volume_cc=0.001):
    """
    doses_scaled: ndarray (n_patients, x, y, z) with dose values in Gy
    voxel_volume_cc: volume of each voxel in cubic centimeters (default 1mm^3 = 0.001cc)
    Returns: pd.DataFrame with one row per patient and dose metrics
    """
    n_patients = doses_scaled.shape[0]
    descriptors = []

    # Define dose's thresholds for Vx
    dose_thresholds = [5, 10, 20, 30, 40]

    for i in range(n_patients):
        dose_3d = doses_scaled[i]
        dose_flat = dose_3d.flatten()

        # Sort the doses decreasingly according to Dcc
        sorted_dose = np.sort(dose_flat)[::-1]
        total_voxels = dose_flat.size

        # Maximum doses
        Dmax = sorted_dose[0]
        Dmean = dose_flat.mean()

        # Doses in the more exposed volumes
        n_voxels_2cc = int(2 / voxel_volume_cc)
        n_voxels_5cc = int(5 / voxel_volume_cc)

        D2cc = sorted_dose[:n_voxels_2cc].mean() if n_voxels_2cc < total_voxels else np.nan
        D5cc = sorted_dose[:n_voxels_5cc].mean() if n_voxels_5cc < total_voxels else np.nan

        # Volumes receiving > x threshold dose
        Vx = {}
        for x in dose_thresholds:
            Vx[f'V{x}'] = 100 * np.sum(dose_flat >= x) * voxel_volume_cc  # in cc
            Vx[f'V{x}_rel'] = 100 * np.mean(dose_flat >= x)  # in %

        # Stack the results
        patient_data = {
            'Dmax': Dmax,
            'Dmean': Dmean,
            'D2cc': D2cc,
            'D5cc': D5cc,
            **Vx
        }
        descriptors.append(patient_data)

    return pd.DataFrame(descriptors)


descriptors = compute_dose_descriptors(doses_scaled)

descriptors.head()

for i in [5,10,20,30,40] :
  descriptors[f"V{i}"] = descriptors[f"V{i}"]/1000

descriptors.head()

plt.figure(figsize=(12, 6))
sns.boxplot(data=descriptors)
plt.xticks(rotation=45)
plt.title("Boxplots of dosimetric descriptors")
plt.grid(True)
plt.show()


# ### Finding outliers with PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize dosimetric descriptors
X_scaled = StandardScaler().fit_transform(descriptors)

# Apply PCA
pca = PCA(n_components=min(10, X_scaled.shape[1]))
pca.fit(X_scaled)

# Cumulated explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel("Number of principal components")
plt.ylabel("Cumulated explained variance")
plt.title("Curve of explained variance by PCA")
plt.grid(True)
plt.axhline(0.9, color='red', linestyle='--', label='90% of variance explained')
plt.legend()
plt.tight_layout()
plt.show()


X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA on patient's doses")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axhline(y=-6.3, color='r', linestyle='--')
plt.axvline(x=-6.5, color='r', linestyle='--')
plt.grid(True)
plt.show()


pca_outliers = X_pca[(X_pca[:, 0] < -6.5) & (X_pca[:, 1] < -6.3)]
pca_outliers

mask = (X_pca[:, 0] < -6.5) & (X_pca[:, 1] < -6.3)
outlier_indices = np.where(mask)[0]
print(f"Outliers from PCA : {outlier_indices}")

#Stack the results
df["outliers_pca"] = df.index.isin(outlier_indices)

# Weights matrice : shape (n_components, n_features)
loadings = pca.components_
feature_names = descriptors.columns


def top_features_per_PC(loadings, feature_names, top_n=5):
    for i, component in enumerate(loadings):
        sorted_idx = np.argsort(np.abs(component))[::-1]
        top_features = [(feature_names[j], component[j]) for j in sorted_idx[:top_n]]
        print(f"\nPC{i+1} (explained variance: {pca.explained_variance_ratio_[i]:.2%}):")
        for feat, weight in top_features:
            print(f"  {feat}: {weight:.3f}")


top_features_per_PC(loadings, feature_names, top_n=15)


# **PC1 (77.84% of explained variance)**
# 
# Top contributing variables: V20, V20_rel, V30, V30_rel, V40_rel
# 
# *Interpretation* :
# This component primarily reflects the overall dose burden across a wide range of volume thresholds (5–40 Gy), especially in terms of volume-based dose metrics.
# 
# --> PC1 captures how much of the heart volume receives low to moderately high doses.
# It strongly emphasizes relative and absolute Vx metrics, indicating that patients with larger irradiated cardiac volumes across these thresholds will score higher on this component.
# 
# *Additional Notes* :
# The small positive contributions of Dmax, D5cc, and D2cc suggest that peak/focal doses also correlate slightly with the overall volume-based dose pattern, but much less strongly.

# **PC2 (18.97% of explained variance)**
# 
# Top contributing variables: D2cc, D5cc, Dmax
# 
# Interpretation:
# This component captures the local or focal maximum dose.
# 
# --> It distinguishes patients with small volumes exposed to very high radiation doses.

# ### Finding outliers with DBSCAN

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
labels = clustering.labels_

# -1 = outliers
df['outliers_dbscan'] = (labels == -1)


clustering.labels_
df

# ### Finding outliers with distances

from sklearn.metrics import pairwise_distances

# Mean of distances from each patient to the others
distances = pairwise_distances(X_scaled)
mean_dist = distances.mean(axis=1)

# Some naive method to extract the 5 farest patients
df['mean_dist'] = mean_dist
outliers = df.sort_values('mean_dist', ascending=False).head(5)
df["outliers_mean_dist"] = df["mean_dist"].isin(outliers["mean_dist"])


# ### Finding overall outliers

outliers

df["outliers_score"] = (df["outliers_pca"].astype(int) + df["outliers_dbscan"].astype(int) + df["outliers_mean_dist"].astype(int))

df.sort_values("outliers_score", ascending=False).head(10)

df_final = df[df["outliers_score"]>=2]
df_final["Overall Survival at 2 years"]

# df 61 : index dist is 247 ; No survival
# 
# df 164 : index dist is 164 ; Yes survival
# 
# df 197 : index dist is 37 ; Yes survival
# 
# df 199 : index dist is 46 ; Yes survival
# 
# df 258 : index dist is 103 ; No survival

final_outliers = df[df["outliers_score"] >= 2]
outliers_index = final_outliers.index
print(f"Final outliers : {outliers_index}")


data_scaled = (df[["Age", "Weight", "Height"]] - df[["Age", "Weight", "Height"]].mean()) / df[["Age", "Weight", "Height"]].std()
data_scaled = pd.concat([data_scaled, df[["level_0",	"index",	"ID", "outliers_score"]]], axis=1)
data_scaled.head()

data_scaled["dist"] = np.abs(data_scaled["Age"] - data_scaled["Age"].median()) + np.abs(data_scaled["Weight"] - data_scaled["Weight"].median()) + np.abs(data_scaled["Height"] - data_scaled["Height"].median())
# data_scaled["dist_index"] = data_scaled["dist"].rank(method="first").astype(int)
data_scaled.info()
