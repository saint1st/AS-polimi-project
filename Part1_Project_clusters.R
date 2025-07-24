
#------- 1. Libraries and Data Loading ------


library(readxl)
library(dplyr)
library(stringr)
library(cluster)

# Load the dataset from the second sheet
data <- read_excel("dataset_students.xlsx", sheet = 2)


#------- 2. Renaming Ambiguous Column Names -------

renames <- function(data) {
  data <- data %>%
    rename(
      Chemo_Drugs = `Drug...29`,
      Chemo_Cycles = Cycles,
      Chemo_Disc = Discontinued...31,
      Reason_Chemo_Disc = `Reason...32`,
      Alt_Chemo_Drug = `Alternative Drug...33`,
      Immuno_Drug = `Drug...35`,
      Immuno_Cycles = Cycle,
      Immuno_Disc = `Discontinued...37`,
      Reason_Immuno_Disc = `Reason...38`,
      Alt_Immuno_Drug = `Alternative Drug...39`,
      RT_Disc = `Discontinued...47`,
      Reason_RT_Disc = `Reason...48`
    )
  return(data)
}

data <- renames(data)


#------- 3. Preprocessing and Value Cleaning -------

# Convert string with "+" to numeric sum
eval_sum_string <- function(x) {
  if (str_detect(x, "\\+")) {
    return(sum(as.numeric(unlist(str_split(x, "\\+")))))
  } else {
    return(as.numeric(x))
  }
}

uniformizing_categorical_data <- function(data) {
  # Standardize values in categorical variables
  data$Hypertension[data$Hypertension == "SI"] <- "YES"
  data$Immuno_Disc[data$Immuno_Disc == "SI"] <- "YES"
  data$`Cardiac Events Post RT`[data$`Cardiac Events Post RT` == 1] <- "YES"
  data$`Cardiac Events Post RT`[data$`Cardiac Events Post RT` == 0] <- "NO"
  data$`Diabetes drugs`[data$`Diabetes drugs` == "SI"] <- "YES"
  data$N <- toupper(data$N)
  data$`Target therapy`[data$`Target therapy` != "NO"] <- "YES"
  
  # Unit corrections
  data$Height[169:209] <- data$Height[169:209] * 100
  data$Weight[205] <- 71
  
  # Mutation flag
  data$mutations[is.na(data$mutations)] <- "NO"
  data$mutations[!(data$mutations %in% c("No data", "NO"))] <- "YES"
  
  # Fix SAT and BMI values
  data$SAT <- as.numeric(data$SAT)
  data$SAT <- ifelse(data$SAT < 1, 100 * data$SAT, data$SAT)
  data$BMI[c(1, 3, 4, 6, 205)] <- data$Weight[c(1, 3, 4, 6, 205)] / (data$Height[c(1, 3, 4, 6, 205)]/100)^2
  
  # Clean 'Others' column
  data$Others[data$Others %in% c("no", "-", "no patologie degne di nota", "f")] <- "NO"
  
  # Evaluate chemo cycles expressed as sums
  data$Chemo_Cycles[c(117, 178)] <- sapply(data$Chemo_Cycles[c(117, 178)], eval_sum_string)
  
  # Replace "III" in Stage with NA
  data$Stage_at_RT[data$Stage_at_RT == "III"] <- NA
  
  return(data)
}

data <- uniformizing_categorical_data(data)


#------- 4. Handling Missing Values -------

cleaning_na <- function(data) {
  # Replace NA with "NO" for selected categorical columns
  binary_vars <- c("Antilipidic drugs", "antithrombotic", "Beta-Blockers", "Diuretics", "Smoking status", "Others",
                   "mutations", "Target therapy", "N", "Mediastino", "Chemo_Disc", "Immunotherapy", "Immuno_Disc",
                   "RT_Disc", "Cardiac Events Post RT", "pulmonary events")
  data[binary_vars] <- lapply(data[binary_vars], function(x) ifelse(is.na(x), "NO", x))
  
  # Replace NA in Schedule with 9
  data$Schedule[is.na(data$Schedule)] <- 9
  
  # Replace NA in cycle columns with 0
  data$Chemo_Cycles[is.na(data$Chemo_Cycles)] <- 0
  data$Immuno_Cycles[is.na(data$Immuno_Cycles)] <- 0
  
  # BMI/Weight/Height Imputation
  data$BMI <- ifelse(is.na(data$BMI),
                     ave(data$BMI, data$Sex, FUN = function(x) median(x, na.rm = TRUE))[is.na(data$BMI)],
                     data$BMI)
  to_fill <- is.na(data$Height) & !is.na(data$Weight) & !is.na(data$BMI)
  data$Height[to_fill] <- sqrt(data$Weight[to_fill] / data$BMI[to_fill]) * 100
  data$Height <- ifelse(is.na(data$Height),
                        ave(data$Height, data$Sex, FUN = function(x) median(x, na.rm = TRUE))[is.na(data$Height)],
                        data$Height)
  to_fill <- is.na(data$Weight) & !is.na(data$Height) & !is.na(data$BMI)
  data$Weight[to_fill] <- (data$Height[to_fill] / 100)^2 * data$BMI[to_fill]
  
  # Impute Volume, KPS, PDL1, machine type
  data$VOLUME_T[is.na(data$VOLUME_T)] <- data$VOLUME_TOT[is.na(data$VOLUME_T)]
  data$KPS[is.na(data$KPS)] <- 90
  mask <- is.na(data$`PDL-1 level`)
  data[mask, "PDL-1 level"] <- ifelse(data$KPS[mask] == 100, "1-49", "0-1")
  data$`type of machine`[is.na(data$`type of machine`)] <- 2
  
  return(data)
}

data <- cleaning_na(data)


#------- 5. Classifying Free Text (Others) -------

others_classification <- function(data)
{
  ### this function creates a new column which translates doctor notes into a numerical variable.
  ### setting different values according to different types of pathologies:
  ### - 0 for NO pathology;
  ### - 1 for CARDIAC pathologies;
  ### - 2 for VASCULAR pathologies (not directly related to the heart);
  ### - 3 for RESPIRATORY pathologies;
  ### - 4 for previous TUMORAL pathologies;
  ### - 5 for OTHERS.
  ### then returns data frame
  
  data <- data %>%
    mutate(Others_Code = case_when(
      Others == "NO" ~ 0,
      Others %in% c("\"si (bypass aorto-coronarico IMA nel 1992)\"", "\"si (Dilatazione aorta ascendente ( 46-47 mm ) \"", 
                    "cardiopatia ischemica rivascolarizzata", "fibrillazione atriale", "ICD", "IMA pregresso - fac", 
                    "pregresso IMA", "pregresso IMA e PTCA nel 2019", "si (cardiopatia ipocinetica lieve)", 
                    "si (Cardiopatia ipocinetico-dilatativa con significativo peggioramento della disfunzione sistolica EF 25%; esiti di plastica mitralica con IM moderata residua. Segni indiretti di ipertensione polmonare 45 mHg)", 
                    "si (cardiopatia ischemica PTCA e stent nel 2009)", "si (IMA 2000 con FE attuale del 50%)", 
                    "SI (scompenso cardiaco in corso di FA ad elevata risposta ventricolare)", 
                    "si (scompenso cardiaco)") ~ 1,
      Others %in% c("aneurisma AA", "ectasia aorta sottorenale", "Ictus ischemico recidivante", 
                    "Ictus-aneurisma carotide") ~ 2,
      Others %in% c("BPCO", "bpco", "BPCO in terapia con trimbow", 
                    "BPCO in triplice terapia inalatoria; prolasso valvolare", "ferita arma fuoco al polmone sx", 
                    "HIV, HCV, schizofrenia,Bpco", "TEP dx") ~ 3,
      Others %in% c("ha fatto che neoadj+ intervento + RT adiuv", 
                    "ha fatto intervento e cht adiuvante e RT su residuo e mediastino", "k laringe", 
                    "k orofaringe CHT-RT trattap (2010)", "LINFOMA", "melanoma coscia sn", 
                    "melanoma parete addominale e linfectomia ascellare", "Meningioma G1", 
                    "no, pregressa k mammella", "replanning (18fr piano1+12fr piano2)") ~ 4,
      TRUE ~ 5
    )) %>%
    relocate(Others_Code, .after = Others) 
  
  return(data)
}

data <- others_classification(data)


#------- 6. Other Feature Cleaning -------

Site_of_primary_cleaning <- function(data) {
  data$Tumor_side <- ifelse(substr(data$`Site of Primary`, 1, 1) == "L", 1, 0)
  data <- data %>% select(-`Site of Primary`)
  return(data)
}

data <- Site_of_primary_cleaning(data)


#------- 7. Template Patient Selection -------

# Normalize features
data_scaled <- scale(data[c("Weight", "Height", "Age")])
data_scaled <- data.frame(data_scaled, ID = data$ID)

# Find patient closest to median
med_weight <- median(data_scaled$Weight, na.rm = TRUE)
med_height <- median(data_scaled$Height, na.rm = TRUE)
med_age <- median(data_scaled$Age, na.rm = TRUE)

data_all_closest_scaled <- data[which.min(abs(data_scaled$Weight - med_weight) +
                                            abs(data_scaled$Height - med_height) +
                                            abs(data_scaled$Age - med_age)), ]

cat("Patient ID of the template used:", data_all_closest_scaled$ID, "\n")


#------- 8. Preparing for Clustering -------

data<- subset(data,select=-c(Others, Chemo_Drugs, Alt_Chemo_Drug, Immuno_Drug, Alt_Immuno_Drug, Reason_Chemo_Disc, Reason_Immuno_Disc, Immuno_Drug,...51,...54,Reason_RT_Disc,`Date of birth`,Stage))
data_cluster <- data

# Drop columns not used in clustering
data_cluster <- data.frame(subset(data_cluster,select=-c(`type of machine`,Tumor_side)))
data_cluster$Schedule <- as.character(data_cluster$Schedule)
colnames(data_cluster) <- gsub(" +", ".", colnames(data_cluster))

# Replace 99 with "-1" and 9 with NA
data_cluster$Schedule[data_cluster$Schedule == "99"] <- "-1"
data_cluster$Schedule[data_cluster$Schedule == "9"] <- NA

# Data transformation
data_test_3 <- data.frame(data_cluster)

cols <- c("Diabetes.drugs", "Antilipidic.drugs", "Beta.Blockers", "Diuretics", "antithrombotic", "Hypertension", "Target.therapy", "Immunotherapy", "Mediastino", "N")
cols_ordinal <- c("Sex", "Stage_at_RT")
data_test_3[cols] <- lapply(data_test_3[cols], factor)
data_test_3[cols_ordinal] <- lapply(data_test_3[cols_ordinal], function(x) factor(x, ordered = TRUE))

data_test_3$KPS<-as.integer(apply(data_test_3["KPS"],1,function(x) if_else(as.integer(x)>80,x,80)))
data_test_3["Medications"]    <- as.integer(data_test_3$Diabetes.drugs)+as.integer(data_test_3$Antilipidic.drugs)+as.integer(data_test_3$Beta.Blockers)+as.integer(data_test_3$Diuretics)+as.integer(data_test_3$antithrombotic)+as.integer(data_test_3$Hypertension)-6
data_test_3$Medications<- apply(data_test_3["Medications"],1,function(x) if_else(x>=3,3,x))
data_test_3["Therapy"] <- as.integer(data_test_3$Target.therapy)+as.integer(data_test_3$Immunotherapy) -2
data_test_3["Schedule"] <- as.integer(data_test_3$Schedule)+1
data_test_3["Others_pb"]<-as.integer(data_test_3$Mediastino)+as.integer(data_test_3$N)-2
data_test_3$Smoking.status<- as.factor(apply(data_test_3["Smoking.status"],1,function(x) if_else(x=="NO",x,"YES")))

# Final subset for clustering
data_test_3<- subset(data_test_3,select=-c(ID, Weight, Height, SAT, mutations, Dimension, Chemo_Cycles, Immuno_Cycles, 
                                           start.of..RT, Prescription.Dose, Number.of.Fractions, EQD2_PrescriptionDose, 
                                           Overall.Survival.at.2.years, OS_months, Cardiac.Events.Post.RT, 
                                           pulmonary.events, Immuno_Disc,CHEMO,Chemo_Disc,RT_Disc,VOLUME_TOT,
                                           Diabetes.drugs,Antilipidic.drugs,Beta.Blockers,Diuretics,
                                           antithrombotic,Target.therapy,Immunotherapy,
                                           Others_Code,Mediastino,N,Hypertension,PDL.1.level))
#Columns to keep (in that order so the weights match) : Age,Sex, BMI, Smoking_status,KPS, Stage_at_RT, Schedule, Medications, Volume_T, Therapy, Others_pb


#------- 9. Clustering -------

# Apply Gower distance and Ward’s clustering
optimal_weight_3<- c(1,0.3, 1,0.3 ,1,1 ,1,1 ,1,1 ,1 )
d_3<- daisy(data_test_3,metric="gower",weights = optimal_weight_3)
ward.D <- hclust(d_3, method='ward.D')
cluster_ward.3c <- cutree(ward.D, k = 3)

# Visualize dendrogram
plot(ward.D)

# Silhouette score
sil_score <- function(labels, dist) {
  # Compute the average of the silhouette widths
  sil <- silhouette(labels, dist)
  sil_widths <- sil[,"sil_width"]
  mean(sil_widths)
}

sil_score(cluster_ward.3c,d_3)
cat("Silhouette Score: ", sil_score(cluster_ward.3c, d_3), "\n")


#------- 10. Export -------

data_with_clusters<- cbind(data_test_3,cluster_ward.3c)
write.csv(data_with_clusters, "data_with_clusters.csv", row.names = FALSE)
