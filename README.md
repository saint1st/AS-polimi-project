# Voxel-Based Analysis and Clustering of Cardiac Regions in Lung Cancer Patients
This project focuses on voxel-based statistical analysis and unsupervised clustering of CT scans to identify cardiac regions associated with 2-year survival outcomes in lung cancer patients undergoing treatment.

Using advanced voxel-wise statistical modeling and machine learning clustering techniques, we analyze imaging data to uncover spatial patterns in the heart region that may be predictive of survival. This work contributes to a better understanding of treatment-related cardiac toxicity and its potential impact on long-term outcomes.

🧠 Methodology
Voxel-Based Analysis (VBA): We performed voxel-wise survival modeling across cardiac regions using imaging-derived features from CT scans.

Clustering: Unsupervised clustering was applied to identify coherent spatial patterns in the voxel-wise statistical maps.

Statistical Evaluation: We evaluated the significance of identified clusters in relation to patient survival, adjusting for potential confounders.

🧬 Data
The dataset includes thoracic CT scans from lung cancer patients treated at Fondazione IRCCS Istituto Nazionale dei Tumori di Milano, with a minimum of 2 years of follow-up data.

🧑‍🏫 Supervision
This research project was conducted under the mentorship of the
Fondazione IRCCS Istituto Nazionale dei Tumori di Milano.

📂 Contents
preprocessing/ – CT scan preprocessing pipeline

vba/ – Voxel-based analysis scripts

clustering/ – Clustering and spatial pattern analysis

visualization/ – Tools for visualizing voxel maps and clusters

notebooks/ – Interactive notebooks demonstrating methodology and results

📌 Goal
To identify and visualize heart regions that are statistically associated with survival in lung cancer patients, potentially guiding future radiotherapy planning and risk stratification.
