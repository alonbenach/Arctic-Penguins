import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

# Loading and examining the dataset
penguins_df = pd.read_csv("data/penguins.csv")
# Check the data types of all columns
print(penguins_df.dtypes)

# Examine the data
print(penguins_df.head())
print(penguins_df.shape)
# Count missing values
missing_values1 = penguins_df.isna().sum()
print(missing_values1)

# Drop observation with only NaN values
penguins_clean = penguins_df.dropna()

# Count missing values after dropping
missing_values2 = penguins_clean.isna().sum()
print(missing_values2)

# Identify columns that require preprocessing of dummy variables
columns_requiring_dummies = penguins_clean.select_dtypes(include=["object"]).columns
columns_requiring_dummies
# Creating a new dummy feature based on the first observation in the columns_requiring_dummies
first_observation = penguins_clean[columns_requiring_dummies[0]].iloc[0]
penguins_clean[columns_requiring_dummies[0]] = (
    penguins_clean[columns_requiring_dummies[0]] == first_observation
).astype(int)
# Rename the column to the value of its former first_observation
penguins_clean.rename(
    columns={columns_requiring_dummies[0]: first_observation}, inplace=True
)
print(penguins_clean.head())

# Selecting only the continuous features (float type)
continuous_features = penguins_clean.select_dtypes(include=["float64"])

# Creating boxplots for each continuous feature
plt.figure(figsize=(12, 8))
continuous_features.boxplot()
plt.title("Boxplots of Continuous Features")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

# Calculating IQR for 'flipper_length_mm'
Q1 = continuous_features["flipper_length_mm"].quantile(0.25)
Q3 = continuous_features["flipper_length_mm"].quantile(0.75)
IQR = Q3 - Q1

# Setting lower and upper threshold
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

# Finding values outside the threshold
outliers = continuous_features[
    (continuous_features["flipper_length_mm"] < lower_threshold)
    | (continuous_features["flipper_length_mm"] > upper_threshold)
]
print(outliers)

# Removing outlier observations based on 'flipper_length_mm'
penguins_clean = penguins_clean[
    ~(
        (penguins_clean["flipper_length_mm"] < lower_threshold)
        | (penguins_clean["flipper_length_mm"] > upper_threshold)
    )
]

# Selecting only the continuous features (float type)
continuous_features2 = penguins_clean.select_dtypes(include=["float64"])

# Creating boxplots for each continuous feature
plt.figure(figsize=(12, 8))
continuous_features2.boxplot()
plt.title("Boxplots of Continuous Features")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

# Perform scaling and PCA
scaler = StandardScaler()
pca = PCA()
penguins_scaled = scaler.fit_transform(penguins_clean)
# Fit penguins_scaled
dfx_pca = pca.fit(penguins_scaled)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel("PCA feature")
plt.ylabel("variance")
plt.xticks(features)
plt.show()

# prepare the n_component for the k-means algorithm
n_components = sum(dfx_pca.explained_variance_ratio_ > 0.1)

# Fit again with n_components
pca = PCA(n_components=n_components)
penguins_pca = pca.fit_transform(penguins_scaled)

# Detect the number of clusters with inertia
ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to penguins_pca
    model.fit(penguins_pca)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, "-o")
plt.xlabel("number of clusters, k")
plt.ylabel("inertia")
plt.xticks(ks)
plt.show()

# Initialize and fit TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_transformed = tsne.fit_transform(penguins_pca)

# Plot the TSNE penguins_pca data
plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1])
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.title("TSNE Transformed Data")
plt.show()

# Run the k-means clustering algorithm
# Create scaler
scaler = StandardScaler()
# Create KMeans instance
kmeans = KMeans(n_clusters=4)
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)
# Fit the pipeline to samples
pipeline.fit(penguins_pca)
# Calculate the cluster labels: labels
labels = pipeline.predict(penguins_pca)

# Create a final statistical DataFrame for each cluster.
penguins = penguins_clean.drop("MALE", axis=1)
penguins["labels"] = kmeans.labels_
numeric_columns = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "labels"]
stat_penguins = penguins.groupby("labels")[numeric_columns].mean()
stat_penguins
