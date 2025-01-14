import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import scanpy as sc
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from matplotlib.font_manager import FontProperties
import anndata as ad
from scipy.stats import ranksums
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import umap
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
import sys

def logistic_regression_feature_prediction_simple(splice_adata, feature, K=50, test_size=0.2, random_state=42):
    """
    Train and evaluate a multinomial logistic regression model using latent factors
    from splice_adata to predict the feature of interest. Returns prediction accuracy,
    trained model, label encoder, coefficients, and p-values.
    
    Parameters:
    - splice_adata: AnnData object containing the data
    - feature: str, name of the feature to predict (e.g., 'age', 'cell_type_grouped')
    - K: int, number of latent factors to use for regression
    - test_size: float, proportion of the data to use for testing
    - random_state: int, seed for reproducibility
    
    Returns:
    - accuracy: float, prediction accuracy on the test set
    - model: trained LogisticRegression model
    - label_encoder: LabelEncoder instance used to encode the target variable
    - coefficients_df: Pandas DataFrame with logistic regression coefficients and p-values
    """

    # Step 1: Extract latent factors and target feature
    data_combined = splice_adata.obs
    latent_factors = [f'factor_{i}' for i in range(1, K + 1)]
    X = data_combined[latent_factors]

    # Step 2: Encode the target variable (e.g., 'sex', 'age', 'cell_type_grouped')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data_combined[feature])

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

    # Step 4: Train a multinomial logistic regression model
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logreg.fit(X_train, y_train)

    # Step 5: Evaluate the model on the test set
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{feature.capitalize()} Prediction Accuracy: {accuracy:.4f}")

    # Step 6: Train the model on the full dataset for coefficient analysis
    logreg.fit(X, y_encoded)

    # Step 7: Get the logistic regression coefficients
    coefficients = logreg.coef_

    # Step 9: Create a DataFrame with coefficients and p-values
    coefficients_df = pd.DataFrame(coefficients, columns=latent_factors, index=label_encoder.classes_)
    return accuracy, coefficients_df


def get_unique_top_factors_by_group(coefficients_df, top_n=5):
    """
    Get top N unique factors across all groups (rows) in the coefficients DataFrame.
    
    Parameters:
    - coefficients_df: DataFrame with logistic regression coefficients.
    - top_n: Number of top factors to select for each group (row).
    
    Returns:
    - unique_factors: List of unique top factors across all groups.
    """
    top_factors_per_group = []
    
    # Step 1: Identify top N factors for each group (row)
    for group in coefficients_df.index:
        top_factors = coefficients_df.loc[group].abs().nlargest(top_n).index.tolist()
        top_factors_per_group.extend(top_factors)
    
    # Step 2: Get unique factors across all groups
    unique_factors = list(set(top_factors_per_group))
    return unique_factors


def plot_top_factor_distributions(splice_adata, top_factors, age_column="age", plot_type="boxenplot", log_scale=False):
    """
    Plot distributions of cell factor contributions for top N factors with the largest coefficients
    from multinomial logistic regression across different age groups and save the plot as a PDF file.
    
    Parameters:
    - splice_adata: AnnData object containing the cell factor contributions.
    - top_factors: List of top factors to plot.
    - age_column: Column in splice_adata.obs representing the age groups (default="age").
    - plot_type: Type of plot to generate ("boxplot", "violinplot", "stripplot", "boxenplot", or "swarmplot").
    - log_scale: Boolean, whether to apply a logarithmic scale to the y-axis (default=False).
    """

    # Step 1: Extract factor contributions and age groups from splice_adata.obs
    data_combined = splice_adata.obs[top_factors + [age_column]]

    # Step 2: Melt the DataFrame for easier plotting
    data_melted = pd.melt(data_combined, id_vars=[age_column], var_name="Factor", value_name="Contribution")

    # Step 3: Plot distributions using the selected plot type
    plt.figure(figsize=(6, 5))

    if plot_type == "boxenplot":
        sns.boxenplot(x="Factor", y="Contribution", hue=age_column, data=data_melted)

        # Calculate and plot medians as red dots
        medians = data_melted.groupby(['Factor', age_column])['Contribution'].median().reset_index()
        # Iterate over each factor and age group and plot the median as a red dot
        for i, factor in enumerate(top_factors):
            for j, age in enumerate(medians[age_column].unique()):
                median_value = medians[(medians['Factor'] == factor) & (medians[age_column] == age)]['Contribution'].values
                if median_value.size > 0:
                    # Use plt.scatter to plot the median for each age group on the boxenplot
                    plt.scatter([i + (j * 0.1)], median_value, color='red', zorder=5, s=20)

    # Step 4: Apply logarithmic scale if requested
    if log_scale:
        plt.yscale("log")

    # Customize font sizes for labels and ticks
    plt.xlabel("Factors", fontsize=15)
    plt.ylabel("Factor Cell Activity", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tight_layout()

    # Step 5: Save plot as a PDF file with current date and factors in the filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    factors_str = "_".join(top_factors)
    filename = f"factor_distributions_{factors_str}_{date_str}.pdf"
    
    # Save the figure to a PDF
    plt.savefig(filename, format='pdf')

    # Display the plot
    plt.show()
    print(f"Plot saved as {filename}")

# Step 1: Create the age_category column
def create_age_category(splice_adata):
    splice_adata.obs['age_category'] = splice_adata.obs['age'].map({
        '3m': 'young', 
        '18m': 'old', 
        '24m': 'old'
    })

# Step 2: Define function to calculate delta medians, average, and median activity scores
def compare_factors_by_age(splice_adata, factors, cell_type_col="cell_type_grouped"):
    results = []
    
    # Iterate over unique cell types
    for cell_type in splice_adata.obs[cell_type_col].unique():
        # Subset cells by cell type
        cell_type_data = splice_adata[splice_adata.obs[cell_type_col] == cell_type]

        # Further subset into young and old groups
        young_cells = cell_type_data[cell_type_data.obs['age_category'] == 'young']
        old_cells = cell_type_data[cell_type_data.obs['age_category'] == 'old']

        # Iterate over factors
        for factor in factors:
            # Extract factor activity scores for young and old
            young_scores = young_cells.obs[factor].values
            old_scores = old_cells.obs[factor].values

            # Compute median for young and old
            young_median = pd.Series(young_scores).median()
            old_median = pd.Series(old_scores).median()
            
            # Compute the delta in medians
            delta_median = old_median - young_median

            # Perform Wilcoxon rank-sum test
            _, p_value = ranksums(young_scores, old_scores)

            # Calculate average and median factor activity across all cells in the cell type
            avg_factor_activity = pd.Series(cell_type_data.obs[factor].values).mean()
            median_factor_activity = pd.Series(cell_type_data.obs[factor].values).median()

            # Append result to the list
            results.append({
                'cell_type': cell_type,
                'factor': factor,
                'delta_median': delta_median,
                'p_value': p_value,
                'avg_factor_activity': avg_factor_activity,
                'median_factor_activity': median_factor_activity
            })

    # Convert results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df


# Function to plot the distribution of factor activity for young and old mice
def plot_factor_distribution(splice_adata, cell_type, factor):
    # Filter data by the given cell type
    cell_type_data = splice_adata[splice_adata.obs['cell_type_grouped'] == cell_type]
    
    # Split the data into young and old groups
    young_data = cell_type_data[cell_type_data.obs['age_category'] == 'young']
    old_data = cell_type_data[cell_type_data.obs['age_category'] == 'old']
    
    # Extract factor values for both groups
    young_scores = young_data.obs[factor].values
    old_scores = old_data.obs[factor].values
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the distribution for young mice
    sns.kdeplot(young_scores, label="Young", color="blue", shade=True)
    
    # Plot the distribution for old mice
    sns.kdeplot(old_scores, label="Old", color="red", shade=True)
    
    # Plot the median for young
    young_median = pd.Series(young_scores).median()
    plt.axvline(young_median, color="blue", linestyle="--", label=f'Young Median: {young_median:.2f}')
    
    # Plot the median for old
    old_median = pd.Series(old_scores).median()
    plt.axvline(old_median, color="red", linestyle="--", label=f'Old Median: {old_median:.2f}')
    
    # Set plot title and labels
    plt.title(f"Factor {factor} Distribution for {cell_type} (Young vs Old)", fontsize=16)
    plt.xlabel(f"{factor} Activity Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.show()


# Function to plot the distribution of factor activity for 3m, 18m, and 24m age groups
def plot_factor_distribution_by_age(splice_adata, factor, cell_type=None):
    
    if cell_type is not None:
        # Filter data by the given cell type
        cell_type_data = splice_adata[splice_adata.obs['cell_type_grouped'] == cell_type]
    
    else:
        cell_type_data = splice_adata

    # Define age groups
    age_groups = ['3m', '18m', '24m']
    colors = {'3m': 'blue', '18m': 'green', '24m': 'red'}
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    
    # Plot the distribution for each age group
    for age in age_groups:
        age_data = cell_type_data[cell_type_data.obs['age'] == age]
        scores = age_data.obs[factor].values
        
        # Plot the distribution (KDE)
        sns.kdeplot(scores, label=f"{age}", color=colors[age], shade=True)
        
        # Plot the median for each age group
        median = pd.Series(scores).median()
        plt.axvline(median, color=colors[age], linestyle="--", label=f'{age} Median: {median:.2f}')
    
    # Set plot title and labels
    plt.title(f"Factor {factor} Distribution for {cell_type}", fontsize=17)
    plt.xlabel(f"{factor} Activity Score", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.xlim(0, 1)
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(f"Factor_{factor}_{cell_type}_Cell_Distribution_Activity.pdf", format="pdf")  # Save as PDF
    plt.show()


def linear_regression_age_prediction(splice_adata, K=50, test_size=0.2, random_state=42):
    """
    Train and evaluate a multivariate linear regression model using latent factors
    from splice_adata to predict age. Returns RMSE, trained model, and coefficients
    with confidence intervals.
    
    Parameters:
    - splice_adata: AnnData object containing the data
    - K: int, number of latent factors to use for regression
    - test_size: float, proportion of the data to use for testing
    - random_state: int, seed for reproducibility
    
    Returns:
    - rmse: float, Root Mean Squared Error on the test set
    - coefficients_df: Pandas DataFrame with linear regression coefficients
    """
        
    # Extract latent factors and target (age) variable
    data_combined = splice_adata.obs
    latent_factors = [f'factor_{i}' for i in range(1, K + 1)]
    X = data_combined[latent_factors]
    y = data_combined['age_numeric']

    # Add intercept for statsmodels
    X = sm.add_constant(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train a multivariate linear regression model
    linreg = sm.OLS(y_train, X_train).fit()

    # Evaluate the model on the test set
    y_pred = linreg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Age Prediction RMSE: {rmse:.4f}")

    # Train the model on the full dataset for coefficient analysis
    linreg_full = sm.OLS(y, X).fit()

    # Get the linear regression coefficients, confidence intervals, and p-values
    coefficients = linreg_full.params
    conf_intervals = linreg_full.conf_int()
    std_errors = linreg_full.bse

    # Create a DataFrame with coefficients and confidence intervals
    coefficients_df = pd.DataFrame({
        'Factor': ['Intercept'] + latent_factors,
        'Coefficient': coefficients,
        'Lower CI': conf_intervals[0],
        'Upper CI': conf_intervals[1],
        'Standard Error': std_errors
    })

    return rmse, coefficients_df


# Create a bar plot to visualize the delta_median
def plot_factor_medians(factor_data, factor_name):
    plt.figure(figsize=(7,6))

    # Create a bar plot with delta_median values
    sns.barplot(
        x='delta_median', 
        y='cell_type', 
        data=factor_data, 
        palette='coolwarm',
        edgecolor='black'
    )
    
    # Set plot title and labels
    plt.title(f"Factor {factor_name} - Delta Median Activity", fontsize=15)
    plt.xlabel('Delta Median (Old - Young)', fontsize=16)
    plt.ylabel('Cell Type', fontsize=16)

   # Adjust tick font sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=11)

    # Invert y-axis to display cell types top to bottom in the order of the DataFrame
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"Factor_{factor_name}_Delta_Median_Activity.pdf", format="pdf")  # Save as PDF
    plt.show()


def logistic_regression_feature_prediction(
    splice_adata,
    gene_expression_adata,
    feature: str,
    K: int = 50,
    n_pcs: int = 50,
    covariate_column = 'tissue',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train and evaluate multinomial logistic regression models using splicing factors,
    gene expression PCs, tissue covariate, and their combination to predict the feature of interest.
    
    Parameters:
    - splice_adata: AnnData object containing splicing data with latent factors in `obs`.
    - gene_expression_adata: AnnData object containing gene expression data.
    - feature: str, name of the feature to predict (e.g., 'age', 'cell_type_grouped').
    - K: int, number of splicing latent factors to use.
    - n_pcs: int, number of gene expression PCs to use.
    - test_size: float, proportion of the data to use for testing.
    - random_state: int, seed for reproducibility.
    
    Returns:
    - accuracies: dict, accuracy scores for each model.
    - models: dict, trained LogisticRegression models.
    - label_encoder: LabelEncoder instance used to encode the target variable.
    - coefficients_dfs: dict, DataFrames with logistic regression coefficients for each model.
    """
    
    # Step 1: Extract splicing latent factors
    latent_factors = [f'factor_{i}' for i in range(1, K + 1)]
    X_splicing = splice_adata.obs[latent_factors]
    
    # Step 2: Compute gene expression PCs
    pcs = gene_expression_adata.obsm['X_pca'][:, :n_pcs]
    pc_columns = [f'PC_{i+1}' for i in range(n_pcs)]
    X_gene_expression = pd.DataFrame(pcs, index=gene_expression_adata.obs_names, columns=pc_columns)
    
    # Step 3: One-hot encode the "tissue" covariate
    tissue_encoder = OneHotEncoder()
    tissue_covariate = tissue_encoder.fit_transform(splice_adata.obs[[covariate_column]])
    tissue_columns = [f'{covariate_column}_{cat}' for cat in tissue_encoder.categories_[0]]
    
    # Convert tissue covariate to DataFrame
    X_tissue = pd.DataFrame(tissue_covariate.toarray(), index=splice_adata.obs_names, columns=tissue_columns)

    # Add tissue covariate to splicing and gene expression datasets
    X_splicing = pd.concat([X_splicing, X_tissue], axis=1)
    X_gene_expression = pd.concat([X_gene_expression, X_tissue], axis=1)
    
    # Step 4: Prepare combined dataset and scale it
    X_combined = pd.concat([X_splicing, X_gene_expression], axis=1)
    scaler = StandardScaler()
    X_combined_scaled = pd.DataFrame(scaler.fit_transform(X_combined), index=X_combined.index, columns=X_combined.columns)

    # Step 5: Encode the target variable
    label_encoder = LabelEncoder()
    y = splice_adata.obs[feature].astype(str)  # Ensure the feature is a string type
    y_encoded = label_encoder.fit_transform(y)
    
    # Step 6: Split the data
    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_splicing, y_encoded, test_size=test_size, random_state=random_state
    )
    X_train_g, X_test_g, _, _ = train_test_split(
        X_gene_expression, y_encoded, test_size=test_size, random_state=random_state
    )
    X_train_c, X_test_c, _, _ = train_test_split(
        X_combined_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    X_train_t, X_test_t, _, _ = train_test_split(
        X_tissue, y_encoded, test_size=test_size, random_state=random_state
    )
    
    # Step 7: Train logistic regression models
    logreg_params = {
        'multi_class': 'multinomial',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': random_state
    }
    model_splicing = LogisticRegression(**logreg_params)
    model_gene_expression = LogisticRegression(**logreg_params)
    model_combined = LogisticRegression(**logreg_params)
    model_tissue = LogisticRegression(**logreg_params)  # New model for tissue
    
    model_splicing.fit(X_train_s, y_train)
    model_gene_expression.fit(X_train_g, y_train)
    model_combined.fit(X_train_c, y_train)
    model_tissue.fit(X_train_t, y_train)  # Train tissue-only model
    
    # Step 8: Evaluate the models
    y_pred_s = model_splicing.predict(X_test_s)
    y_pred_g = model_gene_expression.predict(X_test_g)
    y_pred_c = model_combined.predict(X_test_c)
    y_pred_t = model_tissue.predict(X_test_t)  # Predictions for tissue-only model

    # Predict probabilities for AUROC and AUPR
    y_prob_s = model_splicing.predict_proba(X_test_s)
    y_prob_g = model_gene_expression.predict_proba(X_test_g)
    y_prob_c = model_combined.predict_proba(X_test_c)
    y_prob_t = model_tissue.predict_proba(X_test_t)

    accuracy_s = accuracy_score(y_test, y_pred_s)
    accuracy_g = accuracy_score(y_test, y_pred_g)
    accuracy_c = accuracy_score(y_test, y_pred_c)
    accuracy_t = accuracy_score(y_test, y_pred_t)  # Accuracy for tissue-only model
    
    # AUROC and AUPR scores
    try:
        auroc_s = roc_auc_score(y_test, y_prob_s, multi_class='ovr')
        auroc_g = roc_auc_score(y_test, y_prob_g, multi_class='ovr')
        auroc_c = roc_auc_score(y_test, y_prob_c, multi_class='ovr')
        auroc_t = roc_auc_score(y_test, y_prob_t, multi_class='ovr')

        aupr_s = average_precision_score(y_test, y_prob_s, average='macro')
        aupr_g = average_precision_score(y_test, y_prob_g, average='macro')
        aupr_c = average_precision_score(y_test, y_prob_c, average='macro')
        aupr_t = average_precision_score(y_test, y_prob_t, average='macro')

    except ValueError as e:
        # Handle any exception that might occur due to the number of classes
        print(f"Error calculating AUROC/AUPR: {e}")
        auroc_s = auroc_g = auroc_c = auroc_t = None
        aupr_s = aupr_g = aupr_c = aupr_t = None

    print(f"{feature.capitalize()} Prediction Accuracy using Splicing Factors: {accuracy_s:.4f}")
    print(f"{feature.capitalize()} Prediction AUROC using Splicing Factors: {auroc_s:.4f}")
    print(f"{feature.capitalize()} Prediction AUPR using Splicing Factors: {aupr_s:.4f}")
    
    print(f"{feature.capitalize()} Prediction Accuracy using Gene Expression PCs: {accuracy_g:.4f}")
    print(f"{feature.capitalize()} Prediction AUROC using Gene Expression PCs: {auroc_g:.4f}")
    print(f"{feature.capitalize()} Prediction AUPR using Gene Expression PCs: {aupr_g:.4f}")
    
    print(f"{feature.capitalize()} Prediction Accuracy using Combined Features: {accuracy_c:.4f}")
    print(f"{feature.capitalize()} Prediction AUROC using Combined Features: {auroc_c:.4f}")
    print(f"{feature.capitalize()} Prediction AUPR using Combined Features: {aupr_c:.4f}")
    
    print(f"{feature.capitalize()} Prediction Accuracy using Tissue Only: {accuracy_t:.4f}")
    print(f"{feature.capitalize()} Prediction AUROC using Tissue Only: {auroc_t:.4f}")
    print(f"{feature.capitalize()} Prediction AUPR using Tissue Only: {aupr_t:.4f}")
        
    # Step 9: Retrain models on full dataset for coefficient analysis
    model_splicing.fit(X_splicing, y_encoded)
    model_gene_expression.fit(X_gene_expression, y_encoded)
    model_combined.fit(X_combined_scaled, y_encoded)
    model_tissue.fit(X_tissue, y_encoded)  # Retrain tissue-only model
    
    # Step 10: Prepare coefficients DataFrames
    coef_splicing = pd.DataFrame(
        model_splicing.coef_, columns=latent_factors + tissue_columns, index=label_encoder.classes_
    )
    coef_gene_expression = pd.DataFrame(
        model_gene_expression.coef_, columns=pc_columns + tissue_columns, index=label_encoder.classes_
    )
    coef_combined = pd.DataFrame(
        model_combined.coef_, columns=X_combined.columns, index=label_encoder.classes_
    )
    coef_tissue = pd.DataFrame(
        model_tissue.coef_, columns=tissue_columns, index=label_encoder.classes_  # Coefficients for tissue-only model
    )
    
    # Collect results
    accuracies = {
        'splicing': accuracy_s,
        'gene_expression': accuracy_g,
        'combined': accuracy_c,
        'tissue_only': accuracy_t  # Include tissue-only accuracy
    }

    aurocs = {
        'splicing': auroc_s,
        'gene_expression': auroc_g,
        'combined': auroc_c,
        'tissue_only': auroc_t
    }
    
    auprs = {
        'splicing': aupr_s,
        'gene_expression': aupr_g,
        'combined': aupr_c,
        'tissue_only': aupr_t
    }
    
    models = {
        'splicing': model_splicing,
        'gene_expression': model_gene_expression,
        'combined': model_combined,
        'tissue_only': model_tissue  # Include tissue-only model
    }
    
    coefficients_dfs = {
        'splicing': coef_splicing,
        'gene_expression': coef_gene_expression,
        'combined': coef_combined,
        'tissue_only': coef_tissue  # Include tissue-only coefficients
    }
    
    return accuracies, aurocs, auprs, models, label_encoder, coefficients_dfs

def plot_clustermap(coefficients, highlighted_factors=None, cmap="seismic", figsize=(6, 7), center=0):
    """
    Plots a clustermap of the given coefficients and applies custom formatting.
    
    Parameters:
        coefficients (pd.DataFrame): DataFrame containing the coefficients (rows=factors, columns=age groups).
        highlighted_factors (list): List of row labels to highlight (bold and red).
        cmap (str): Colormap for the heatmap.
        figsize (tuple): Size of the figure.
        center (float): Center value for diverging colormap.
    """
    # Create the clustermap
    g = sns.clustermap(coefficients.T, cmap=cmap, annot=False, figsize=figsize, 
                       row_cluster=True, col_cluster=False, xticklabels=1, yticklabels=1, center=center)

    # Set y-axis tick labels (rows) font size and rotation
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=14)

    # Set x-axis tick labels (columns) font size
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=15)

    # Disable color bar visibility
    g.cax.set_visible(True)

    # Highlight specific factors (rows)
    if highlighted_factors:
        for label in g.ax_heatmap.yaxis.get_majorticklabels():
            if label.get_text() in highlighted_factors:
                label.set_weight('bold')   # Make it bold
                label.set_color('red')     # Change color to red

    # Set colorbar font size
    g.cax.tick_params(labelsize=15)

    # Save the figure to a PDF
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"factor_age_coefficients_{date_str}.pdf"
    plt.savefig(filename, format='pdf')
    print(f"Figure saved as {filename}")

    # Show the plot
    plt.show()

    # Example usage:
    # plot_clustermap(coefficients_age, highlighted_factors=['factor_17', 'factor_18', 'factor_19'])
