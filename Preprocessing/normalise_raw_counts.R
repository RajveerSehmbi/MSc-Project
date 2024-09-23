# Load necessary library
library(DESeq2)
library(dplyr)
library(caret)


# List of cancer types
cancer_types <- c("ACC", "CHOL", "BLCA", "BRCA", "CESC", "COAD", "UCEC", 
                  "ESCA", "GBM", "HNSC", "KICH", "KIRC", "KIRP", "DLBC", "LAML", "LIHC", 
                  "LGG", "LUAD", "LUSC", "SKCM", "MESO", "UVM", "OV", "PAAD", 
                  "PCPG", "PRAD", "READ", "SARC", "STAD", "TGCT", "THYM", 
                  "THCA", "UCS")

# List of normal labels
normal_labels <- c("Buccal Cell Normal", "Solid Tissue Normal", "Bone Marrow Normal", "Null")

# List of cancer labels
cancer_labels <- c("Primary Tumor", "Primary Blood Derived Cancer - Peripheral Blood", "Metastatic",
                "Recurrent Tumor", "Additional - New Primary", "Additional Metastatic", "FFPE Scrolls")

normal_labels <- as.list(normal_labels)
cancer_labels <- as.list(cancer_labels)

rows_to_remove <- c("__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique")
# Read text file containing additional rows to remove
additional_rows_to_remove <- readLines("../Data/Read_Counts/CHRy.txt")
rows_to_remove <- c(rows_to_remove, additional_rows_to_remove)



# Initialize empty list to store count data
all_counts <- list()
all_sample_info <- data.frame()

print("Empty list created")

# Loop through cancer types and read in data
for (cancer in cancer_types) {

  print(paste("Processing cancer type:", cancer))
  # Construct file name
  file_name <- paste0("../Data/Read_Counts/TCGA-", cancer, ".htseq_counts.tsv")
  
  # Read the log2(count + 1) data
  log2_count_data <- read.table(file_name, header=TRUE, row.names=1, sep="\t")

  # Remove specified rows
  log2_count_data <- log2_count_data[!(rownames(log2_count_data) %in% rows_to_remove), ]


  # Append to list
  all_counts[[cancer]] <- log2_count_data
  
  # Create sample info for this cancer type
  sample_info <- data.frame(row.names=colnames(log2_count_data), 
                            cancer_code=rep(cancer, ncol(log2_count_data)),
                            sample_type=character(ncol(log2_count_data)),
                            stringsAsFactors=FALSE)

  # Read the sample types
  sample_type_file <- paste0("../Data/Phenotype/", cancer, ".tsv")
  sample_types <- read.table(sample_type_file, header=TRUE, row.names=1, sep="\t")
  sample_types <- sample_types %>% select(-samples)
  sample_types <- sample_types %>% rename(sample_type = sample_type.samples)
  rownames(sample_types) <- gsub("\\-", ".", rownames(sample_types))

  # If a row in sample_info is in sample_types, add the sample_type to sample_info
  for (i in 1:nrow(sample_info)) {
    if (rownames(sample_info)[i] %in% rownames(sample_types)) {
      sample_info$sample_type[i] <- sample_types[rownames(sample_info)[i], "sample_type"]
    }
  }

  print(paste("Sample info created for cancer type", cancer))
  
  # Append to sample info data frame
  all_sample_info <- rbind(all_sample_info, sample_info)
}

print("All data read in")

# Combine all count data into one data frame
combined_counts <- do.call(cbind, all_counts)

print("All data combined")

# Undo log2 transformation
combined_counts <- round((2^combined_counts) - 1)

print("Log2 transformation undone")

# Remove everything up to and including the first full stop from column names
colnames_combined <- colnames(combined_counts)
colnames_combined <- sub("^[^\\.]*\\.", "", colnames_combined)
colnames(combined_counts) <- colnames_combined

# Debugging step: Print column names and row names to identify the issue
cat("Checking for mismatches between column names of combined_counts and row names of all_sample_info:\n")
rownames_sample_info <- rownames(all_sample_info)

# Loop to print mismatches
mismatches_found <- FALSE
for (i in seq_along(colnames_combined)) {
  if (colnames_combined[i] != rownames_sample_info[i]) {
    cat("Mismatch at position", i, ":\n")
    cat("Column name:", colnames_combined[i], "\n")
    cat("Row name:", rownames_sample_info[i], "\n")
    mismatches_found <- TRUE
  }
}

# Ensure the column names of combined_counts are consistent with row names of all_sample_info
if (mismatches_found || !all(colnames_combined == rownames_sample_info)) {
  stop("The column names of the count data do not match the sample information.")
}

print("No mismatches found")

# Remove all rows with a mean count of less than 0.25
combined_counts <- combined_counts[rowMeans(combined_counts) > 0.25, ]

# Debugging: Check the dimensions of the combined counts
cat("Dimensions of combined counts:", dim(combined_counts), "\n")

# Debugging: Check the dimensions of the sample info
cat("Dimensions of sample info:", dim(all_sample_info), "\n")
# Columns of sample info
cat("Columns of sample info:", colnames(all_sample_info), "\n")

print("Rows with zero counts removed")


# ------ COUNTING TUMOUR AND NORMAL SAMPLES ------ 
# Initialize an empty data frame for the summary
cancer_summary <- data.frame(
  cancer_type = character(),
  tumour_count = numeric(),
  normal_count = numeric(),
  stringsAsFactors = FALSE
)

# Loop through each cancer type
for (cancer in cancer_types) {

  sample_info_cancer <- all_sample_info[all_sample_info$cancer_code == cancer, , drop=FALSE]
  
  if ("sample_type" %in% colnames(sample_info_cancer)) {
    normal_indices <- rownames(sample_info_cancer)[sample_info_cancer$sample_type %in% normal_labels]
  } else {
    stop("Column 'sample_type' does not exist in sample_info_cancer")
  }
  
  tumour_indices <- rownames(sample_info_cancer)[sample_info_cancer$sample_type %in% cancer_labels]
  normal_indices <- rownames(sample_info_cancer)[sample_info_cancer$sample_type %in% normal_labels]
  
  # Sum the counts of the indices
  tumour_count <- length(tumour_indices)
  normal_count <- length(normal_indices)

  # Append to the summary data frame
  cancer_summary <- rbind(
    cancer_summary,
    data.frame(
      cancer_type = cancer,
      tumour_count = tumour_count,
      normal_count = normal_count,
      stringsAsFactors = FALSE
    )
  )
  print(paste("Summary created for cancer type", cancer))
}

# Write the summary to a CSV file
write.csv(cancer_summary, file = "normal_tumour_counts.csv", row.names = FALSE)
print("Summary file created")


# Changing cancer type of normal samples to "Normal" and remove sample type column
if (!"Normal" %in% levels(all_sample_info$cancer_code)) {
  levels(all_sample_info$cancer_code) <- c(levels(all_sample_info$cancer_code), "Normal")
}
all_sample_info$cancer_code[all_sample_info$sample_type %in% normal_labels] <- "Normal"
all_sample_info <- all_sample_info %>% select(-sample_type)

# Debugging: Check the unique cancer codes
cat("Unique cancer codes:", unique(all_sample_info$cancer_code), "\n")
print(paste(unique(all_sample_info$cancer_code), "cancer codes found"))


# Reorder so that the column names of the counts are in the same order as the row names of the sample info
all_sample_info <- all_sample_info[colnames(combined_counts), , drop = FALSE]
cat(identical(colnames(combined_counts), rownames(all_sample_info)))


cat(class(all_sample_info$cancer_code))
all_sample_info <- all_sample_info %>% mutate(cancer_code = factor(cancer_code))
cat(class(all_sample_info$cancer_code))

# # Split the combined counts into training and testing sets
# # Set seed for reproducibility
# set.seed(123)
# # Split the data into 80% training and 20% testing, stratified by cancer type
# train_indices <- createDataPartition(all_sample_info$cancer_code, p = 0.8, list = FALSE, times = 1)
# train_counts <- combined_counts[, train_indices, drop = FALSE]
# test_counts <- combined_counts[, -train_indices, drop = FALSE]
# train_sample_info <- all_sample_info[train_indices, , drop = FALSE]
# test_sample_info <- all_sample_info[-train_indices, , drop = FALSE]

# # Debugging: Check the dimensions of the training and testing sets
# cat("Dimensions of training counts:", dim(train_counts), "\n")
# cat("Dimensions of testing counts:", dim(test_counts), "\n")
# cat("Dimensions of training sample info:", dim(train_sample_info), "\n")
# cat("Dimensions of testing sample info:", dim(test_sample_info), "\n")

# # Save the training and testing sets to files
# write.csv(train_counts, file = "train_counts.csv", row.names = TRUE)
# write.csv(test_counts, file = "test_counts.csv", row.names = TRUE)
# write.csv(train_sample_info, file = "train_sample_info.csv", row.names = TRUE)
# write.csv(test_sample_info, file = "test_sample_info.csv", row.names = TRUE)
# print("Training and testing sets saved")


# # Normalize the counts using DESeq2 on the training set
# dds_train <- DESeqDataSetFromMatrix(countData = combined_counts, colData = all_sample_info, design = ~ cancer_code)
# dds_train <- estimateSizeFactors(dds_train)
# vsd_train <- vst(dds_train, blind=FALSE)
# normalized_train_counts <- assay(vsd_train)
# print("Combined set normalized and vst transformed")


# train_counts <- train_counts + 1
# test_counts <- test_counts + 1

# # Normalize the counts using geometric means from the training set
# geoMeans_train <- exp(rowMeans(log(train_counts)))

# dds_test <- DESeqDataSetFromMatrix(countData = test_counts, colData = test_sample_info, design = ~ 1)
# dds_test <- estimateSizeFactors(dds_test, geoMeans = geoMeans_train)
# vsd_test <- vst(dds_test, blind=FALSE)
# normalized_test_counts <- assay(vsd_test)
# print("Testing set normalized and vst transformed")






# Create DESeq2 dataset for training set
dds <- DESeqDataSetFromMatrix(countData = combined_counts, colData = all_sample_info, design = ~ cancer_code)
# Normalize the counts fitted on train
dds <- estimateSizeFactors(dds)
vsd <- vst(dds, blind=FALSE)
# Extract the normalized counts
normalized_counts <- assay(vsd)
print("Normalized counts extracted")


# print ("Saving data for train features")
# colnames(normalized_train_counts) <- gsub("\\.", "-", colnames(normalized_train_counts))
# # Debugging: Check the data
# cat("Saving data with dimensions:", dim(normalized_train_counts), "\n")
# file_name <- paste0("../Data/Normalised_Read_Counts/TRAINvst_features.csv")
# write.csv(normalized_train_counts, file=file_name, row.names=TRUE, quote=FALSE)
# print("Saved normalized counts for train set")

# print("Saving labels for train labels")
# colnames(train_sample_info) <- gsub("\\.", "-", colnames(train_sample_info))
# # Debugging: Check the data
# cat("Saving data with dimensions:", dim(train_sample_info), "\n")
# file_name <- paste0("../Data/Normalised_Read_Counts/TRAINvst_labels.csv")
# write.csv(train_sample_info, file=file_name, row.names=TRUE, quote=FALSE)
# print("Saved labels for train set")

# print ("Saving data for test features")
# colnames(normalized_test_counts) <- gsub("\\.", "-", colnames(normalized_test_counts))
# # Debugging: Check the data
# cat("Saving data with dimensions:", dim(normalized_test_counts), "\n")
# file_name <- paste0("../Data/Normalised_Read_Counts/TESTvst_features.csv")
# write.csv(normalized_test_counts, file=file_name, row.names=TRUE, quote=FALSE)
# print("Saved normalized counts for test set")

# print("Saving labels for test labels")
# colnames(test_sample_info) <- gsub("\\.", "-", colnames(test_sample_info))
# # Debugging: Check the data
# cat("Saving data with dimensions:", dim(test_sample_info), "\n")
# file_name <- paste0("../Data/Normalised_Read_Counts/TESTvst_labels.csv")
# write.csv(test_sample_info, file=file_name, row.names=TRUE, quote=FALSE)
# print("Saved labels for test set")

print("Saving data for all features")
colnames(normalized_counts) <- gsub("\\.", "-", colnames(normalized_counts))
# Debugging: Check the data
cat("Saving data with dimensions:", dim(normalized_counts), "\n")
file_name <- paste0("../Data/Normalised_Read_Counts/ALLvst_features.csv")
write.csv(normalized_counts, file=file_name, row.names=TRUE, quote=FALSE)
print("Saved normalized counts for all samples")

print("Saving labels for all labels")
colnames(all_sample_info) <- gsub("\\.", "-", colnames(all_sample_info))
# Debugging: Check the data
cat("Saving data with dimensions:", dim(all_sample_info), "\n")
file_name <- paste0("../Data/Normalised_Read_Counts/ALLvst_labels.csv")
write.csv(all_sample_info, file=file_name, row.names=TRUE, quote=FALSE)
print("Saved labels for all samples")




# # Split and save the normalized counts by cancer type
# for (cancer in cancer_types) {

#   # Get the samples for this cancer type
#   samples <- rownames(all_sample_info[all_sample_info$cancer_code == cancer, , drop=FALSE])
  
#   # Subset the log2 normalized counts for this cancer type
#   cancer_vst_counts <- normalized_counts[, samples, drop=FALSE]

#   # Replace dots with dashes in column names
#   colnames(cancer_vst_counts) <- gsub("\\.", "-", colnames(cancer_vst_counts))

#   # Debugging: Check the subset data
#   cat("Saving data for", cancer, "with dimensions:", dim(cancer_vst_counts), "\n")
  
#   # Save to a file
#   file_name <- paste0("../Data/Normalised_Read_Counts/TCGA-", cancer, "_normalized_vst_counts.csv")
#   write.csv(cancer_vst_counts, file=file_name, row.names=TRUE, quote=FALSE)
#   print(paste("Saved normalized counts for cancer type", cancer))
# }

# cat("Normalization and saving completed for all cancer types.\n")