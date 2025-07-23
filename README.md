# 🫀 Voxel-Based Analysis and Clustering of Cardiac Regions in Lung Cancer Patients

This project explores the relationship between cardiac radiation dose and 2-year overall survival (OS) in stage 3 LA-NSCLC patients. Using **voxel-wise statistical modeling**, **unsupervised clustering**, and **advanced visualization**, we investigate spatial dose patterns in CT scans to identify regions most associated with survival outcomes.

---

## 🧠 Methodology Overview

<img src="visuals/pipeline.gif" width="100%" alt="Pipeline Overview"/>

This project follows a multi-stage analytical pipeline:
1. **Preprocessing** of DICOM/CT data
2. **Voxel-Based Analysis (VBA)** for statistical modeling
3. **Unsupervised Clustering** to identify spatial dose patterns
4. **Statistical Significance Testing** (e.g., Welch’s t-test, FDR correction)
5. **Visualization** of results for interpretation

---

## 📊 Key Visualizations

| Concept | Visualization |
|--------|----------------|
| **Dose Distribution Comparison** | <img src="clideo_editor_4eb173ba0818435a8c0e81e1763eefac.gif" width="300"/> |
| **Voxel-Wise t-Map (OS 2y)** |  |
| **Outlier Detection (PCA/DBSCAN)** | |
| **Cluster Overlay on Template** | |

---

## 📂 Repository Structure
├── preprocessing/ # CT scan alignment, resampling, normalization
├── vba/ # Voxel-wise statistical mapping (e.g., t-tests, p-values)
├── clustering/ # PCA, DBSCAN, and spatial clustering scripts
├── visualization/ # Animated plots, overlays, and 3D maps
├── notebooks/ # Jupyter notebooks for exploration and reporting
├── visuals/ # GIFs and images for publication/presentation


---

## 🧬 Dataset

- **Source:** IRCCS Istituto Nazionale dei Tumori di Milano  
- **Patients:** 321 cases across 5 centers  
- **Modalities:** DICOM CT scans + RT dose maps  
- **Features:** Clinical, anatomical, therapy, comorbidity, outcome (2-year OS)

---

## 🧪 Statistical Tools Used

- Welch’s t-test (voxel-wise analysis)
- Benjamini–Hochberg FDR correction
- PCA & DBSCAN for outlier detection
- Mean dose voxel mapping

---

## 💡 Findings

- Survivors had more **focused** and **precise** dose distributions in heart regions
- Non-survivors showed **broader, less localized** exposure
- No single heart substructure showed consistent significant association across all patients
- Template alignment was crucial for meaningful voxel-wise comparisons

---

## 🚀 Goals

- Improve understanding of **cardiac toxicity** from radiotherapy
- Identify **spatial biomarkers** predictive of survival
- Support **personalized radiotherapy planning**

---

## 🏛️ Supervision

> This research was conducted under the mentorship of  
> **Fondazione IRCCS Istituto Nazionale dei Tumori di Milano**, Italy.

---

## 📜 Citation & Literature

- McWilliam et al., "Novel Methodology to Investigate the Effect of Radiation Dose to Heart Substructures on Overall Survival", *International Journal of Radiation Oncology*
- T. Rancati et al., “SLiC Algorithm for Spatial Dose Analysis”, *Journal of the European Society for Radiotherapy and Oncology*

---

## 🧠 Analysis Workflow

<img src="photo_2025-07-23_21-43-43.jpg" alt="Voxel Analysis Pipeline" width="100%"/>

This diagram illustrates the full pipeline:
- **Input**: Patient CT scans and dose maps
- **Deformation Step**: All patient data is mapped to a common anatomical template
- **Statistical Analysis**: Voxel-wise comparisons between survivor groups
- **Output**: p-value maps indicating regions associated with 2-year survival

---

## 📊 Dose Distribution Statistics

### 🔹 Maximum Dose per Patient
<img src="photo_2025-07-23_21-43-51.jpg" alt="Histogram of Max Dose" width="400"/>

- Most patients receive a **maximum dose around 65–70 units**
- Small tails toward underdosed and overdosed individuals

### 🔹 Mean Dose per Patient
<img src="photo_2025-07-23_21-43-55.jpg" alt="Histogram of Mean Dose" width="400"/>

- The **mean dose** is more spread out
- Peak between **1.5–2.0**, with a long tail for higher means

---

## 🌐 Spatial Clustering Result

<img src="photo_2025-07-23_21-44-00.jpg" alt="3D Cluster Labels" width="450"/>

- **3D voxel cluster plot** of significant regions (labeled by color)
- Generated using spatial clustering after thresholding voxel-wise p-values
- Clusters **39, 42, 46, 47** show distinct spatial behavior within heart region

---

