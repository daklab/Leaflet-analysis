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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import umap
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def logistic_regression_feature_prediction_simple(splice_adata, feature, test_size=0.2, random_state=42):
    """
    Train and evaluate a multinomial logistic regression model using latent factors
    from splice_adata to predict the feature of interest.

    Parameters:
    - splice_adata: AnnData object containing the data
    - feature: str, name of the feature to predict (e.g., 'cell_type_grouped', 'age', 'sex')
    - test_size: float, proportion of the data to use for testing
    - random_state: int, seed for reproducibility
    
    Returns:
    - accuracy: float, prediction accuracy on the test set
    - model: trained LogisticRegression model
    - label_encoder: LabelEncoder instance used to encode the target variable
    - coefficients_df: Pandas DataFrame with logistic regression coefficients
    """

    # Extract latent factor matrix X from obsm
    X = splice_adata.obsm["X_PHI"]
    factor_labels = [f"factor_{i}" for i in range(X.shape[1])]
    
    # Encode the categorical feature
    y = splice_adata.obs[feature]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

    # Train multinomial logistic regression
    logreg = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200, n_jobs=-1)
    logreg.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{feature.capitalize()} Prediction Accuracy: {accuracy:.4f}")

    # Train model on full dataset for coefficient analysis
    logreg.fit(X, y_encoded)

    # Extract coefficients
    coefficients = logreg.coef_
    coefficients_df = pd.DataFrame(coefficients.T, index=factor_labels, columns=label_encoder.classes_)

    return accuracy, logreg, label_encoder, coefficients_df

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

def plot_top_factor_distributions(splice_adata, top_factors, age_column="age", plot_type="boxenplot", log_scale=False, save_plot=False):
    """
    Plots the distribution of top latent factors across different age groups using boxen plots.
    
    Parameters:
        splice_adata: AnnData object containing latent factor activities in `.obsm["X_PHI"]`.
        top_factors: List of factor indices to plot.
        age_column: Column name in `.obs` that contains age group labels.
        plot_type: Type of plot ("boxenplot" or "boxplot").
        log_scale: Whether to apply logarithmic scaling to y-axis.
        save_plot: Whether to save the plot as a PDF file.
    """
    
    # Extract factor activities
    top_factors_activities = splice_adata.obsm["X_PHI"][:, top_factors]
    
    # Create DataFrame
    factor_labels = [f"factor_{i}" for i in top_factors]
    data_factors = pd.DataFrame(top_factors_activities, columns=factor_labels)
    data_age = pd.DataFrame(splice_adata.obs[age_column].values, columns=[age_column])
    data_combined = pd.concat([data_age, data_factors], axis=1)
    
    # Melt DataFrame for Seaborn
    data_melted = pd.melt(data_combined, id_vars=[age_column], var_name="Factor", value_name="Contribution")
    
    # Create plot
    plt.figure(figsize=(7, 6))

    if plot_type == "boxenplot":
        sns.boxenplot(x="Factor", y="Contribution", hue=age_column, data=data_melted, dodge=True, palette="Set2")

    elif plot_type == "boxplot":
        sns.boxplot(x="Factor", y="Contribution", hue=age_column, data=data_melted, dodge=True, palette="Set2")

    # Add median values as red dots
    medians = data_melted.groupby(['Factor', age_column])['Contribution'].median().reset_index()
    
    for i, factor in enumerate(factor_labels):  
        for j, age in enumerate(medians[age_column].unique()):
            median_value = medians[(medians['Factor'] == factor) & (medians[age_column] == age)]['Contribution'].values
            if median_value.size > 0:
                plt.scatter(i + (j * 0.2 - 0.2), median_value, color='red', zorder=5, s=25)

    # Apply logarithmic scale if needed
    if log_scale:
        plt.yscale("log")

    # Formatting
    plt.xlabel("Factors", fontsize=14)
    plt.ylabel("Factor Cell Activity", fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.axhline(0, color="black", linestyle="dashed", alpha=0.5)  # Dashed line at 0 for reference
    plt.tight_layout()

    # Save the plot
    if save_plot:
        date_str = datetime.now().strftime("%Y-%m-%d")
        factors_str = "_".join([str(f) for f in top_factors])
        filename = f"factor_distributions_{factors_str}_{date_str}.pdf"
        plt.savefig(filename, format='pdf')
        print(f"Plot saved as {filename}")

    plt.show()

# Step 1: Create the age_category column
def create_age_category(splice_adata):
    splice_adata.obs['age_category'] = splice_adata.obs['age'].map({
        '2m': 'young',
        '3m': 'young', 
        '18m': 'old', 
        '24m': 'old'
    })

def compare_factors_by_age(splice_adata, factors, cell_type_col="cell_type_grouped"):
    """
    Compare the activity of selected latent factors across age groups (young vs. old)
    within each cell type.

    Parameters:
    - splice_adata: AnnData object containing latent factor data in `obsm["X_PHI"]`.
    - factors: List of factor indices to compare (e.g., [0, 1, 2]).
    - cell_type_col: Column name in `obs` that defines cell type groups.

    Returns:
    - result_df: DataFrame summarizing factor differences across age categories.
    """
    
    results = []
    
    # Extract factor matrix
    X_PHI = splice_adata.obsm["X_PHI"]
    factor_labels = [f"factor_{i}" for i in factors]

    # Iterate over unique cell types
    for cell_type in splice_adata.obs[cell_type_col].unique():
        # Create boolean mask for current cell type
        cell_mask = splice_adata.obs[cell_type_col] == cell_type

        # Apply mask to get indices
        cell_indices = splice_adata.obs.index[cell_mask]
        
        # Subset X_PHI to only include selected cells
        cell_type_X = X_PHI[cell_mask, :]

        # Get the corresponding age labels
        age_labels = splice_adata.obs.loc[cell_indices, "age_category"]

        # Create boolean masks for young and old within the selected cell type
        young_mask = age_labels == 'young'
        old_mask = age_labels == 'old'

        if young_mask.sum() == 0 or old_mask.sum() == 0:
            # Skip if there are no young or old cells for this cell type
            continue

        young_X = cell_type_X[young_mask, :][:, factors]  # Extract factors for young cells
        old_X = cell_type_X[old_mask, :][:, factors]      # Extract factors for old cells
        all_X = cell_type_X[:, factors]                   # Extract factors for all cells in this cell type

        # Iterate over factors
        for i, factor in enumerate(factor_labels):
            young_scores = young_X[:, i]
            old_scores = old_X[:, i]
            all_scores = all_X[:, i]

            # Compute median values
            young_median = np.median(young_scores)
            old_median = np.median(old_scores)
            delta_median = old_median - young_median

            # Perform Wilcoxon rank-sum test
            _, p_value = ranksums(young_scores, old_scores)

            # Compute mean and median factor activity across all cells in the cell type
            avg_factor_activity = np.mean(all_scores)
            median_factor_activity = np.median(all_scores)

            # Append results
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
def plot_factor_distribution(splice_adata, cell_type, factor): # to-do, add option to specify which column name contains the "cell_type" or tissue or whatever feature you want to plot distribution of relative to factor activity 
    """
    Plots the distribution of a given latent factor for a specific cell type,
    comparing young vs. old age groups.

    Parameters:
    - splice_adata: AnnData object containing latent factors in `obsm["X_PHI"]`
    - cell_type: str, cell type to filter data by
    - factor: int, index of the latent factor to visualize
    """
    
    # Extract factor matrix and metadata
    X_PHI = splice_adata.obsm["X_PHI"]
    factor_labels = [f"factor_{i}" for i in range(X_PHI.shape[1])]
    
    # Ensure factor index is valid
    if factor >= X_PHI.shape[1]:
        raise ValueError(f"Factor {factor} is out of bounds. Max available factor index: {X_PHI.shape[1] - 1}")

    # Create mask for the selected cell type
    cell_mask = splice_adata.obs['cell_type_grouped'] == cell_type

    if cell_mask.sum() == 0:
        raise ValueError(f"No cells found for cell type '{cell_type}'.")

    # Get the corresponding cell indices
    cell_indices = splice_adata.obs.index[cell_mask]
    
    # Extract factor values for the selected cell type
    cell_type_X = X_PHI[cell_mask, :]
    age_labels = splice_adata.obs.loc[cell_indices, "age_category"]

    # Create masks for young and old
    young_mask = age_labels == 'young'
    old_mask = age_labels == 'old'

    if young_mask.sum() == 0 or old_mask.sum() == 0:
        raise ValueError(f"Missing data: No young or old samples available for '{cell_type}'.")

    # Extract factor scores for young and old
    young_scores = cell_type_X[young_mask, factor]
    old_scores = cell_type_X[old_mask, factor]

    # Create the plot
    plt.figure(figsize=(8, 5))

    # Plot KDE distributions
    sns.kdeplot(young_scores, label="Young", color="blue", shade=True, common_norm=False)
    sns.kdeplot(old_scores, label="Old", color="red", shade=True, common_norm=False)

    # Plot median lines
    young_median = np.median(young_scores)
    old_median = np.median(old_scores)
    plt.axvline(young_median, color="blue", linestyle="--", label=f'Young Median: {young_median:.2f}')
    plt.axvline(old_median, color="red", linestyle="--", label=f'Old Median: {old_median:.2f}')

    # Set labels and title
    plt.title(f"Factor {factor} Distribution for {cell_type} (Young vs Old)", fontsize=14)
    plt.xlabel(f"{factor_labels[factor]} Activity Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # Add legend and layout fixes
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Function to plot the distribution of factor activity for 3m, 18m, and 24m age groups
def plot_factor_distribution_by_age(splice_adata, factor, cell_type=None, save_plot=False):
    
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
        scores = age_data.obsm['X_PHI'][:, factor]
        
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
    if save_plot:
        plt.savefig(f"Factor_{factor}_{cell_type}_Cell_Distribution_Activity.pdf", format="pdf")
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

def plot_factor_medians(result_table, factor_name, save_plot=False):
    """
    Plots the delta median (Old - Young) activity for a given factor across cell types.

    Parameters:
    - result_table (pd.DataFrame): DataFrame containing `factor`, `delta_median`, and `cell_type`.
    - factor_name (str): The name of the factor to plot (e.g., 'factor_20').
    - save_path (str, optional): File path to save the plot (if None, does not save).
    """

    # Filter data for the selected factor and sort by delta_median
    factor_data = result_table[result_table["factor"] == factor_name].sort_values(by="delta_median")

    # Ensure the factor exists in the dataset
    if factor_data.empty:
        print(f"Warning: Factor '{factor_name}' not found in the dataset.")
        return

    # Create figure
    plt.figure(figsize=(7, 6))

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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)

    # Invert y-axis to display cell types top to bottom in the order of the DataFrame
    plt.gca().invert_yaxis()

    # Adjust layout
    plt.tight_layout()

    # Save if save_path is provided
    if save_plot:
        file_name = f"factor_{factor_name}_delta_median_plot.pdf"
        plt.savefig(file_name, format='pdf')
        print(f"Saved plot to '{file_name}'.")

    # Show the plot
    plt.show()

def plot_factors_with_delta(coefficients_with_deltas, highlight_factors=None, save_path=None):
    """
    Plots two side-by-side bar plots:
    1. Coefficients with error bars for aging prediction.
    2. Factor delta (Old - Young) in cell activity.

    Parameters:
    - coefficients_with_deltas (pd.DataFrame): DataFrame containing factor coefficients, confidence intervals, and delta values.
    - highlight_factors (list): List of factors to highlight in red (e.g., ['factor_17', 'factor_2']).
    - save_path (str, optional): File path to save the plot (if None, does not save).
    """

    # Set global font size
    plt.rcParams.update({'font.size': 12})

    # Sort DataFrame by coefficient in descending order
    coefficients_with_deltas = coefficients_with_deltas.sort_values(by='Coefficient', ascending=False)

    # Calculate Standard Error
    coefficients_with_deltas["Standard Error"] = (coefficients_with_deltas["CI_Upper"] - coefficients_with_deltas["CI_Lower"]) / 3.92

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    # Plot 1: Coefficients with error bars
    ax1.errorbar(coefficients_with_deltas['Coefficient'], coefficients_with_deltas['Factor'],
                 xerr=1.96 * coefficients_with_deltas['Standard Error'], fmt='o', capsize=3, markersize=4, color='lightblue')

    # Add a vertical line at x = 0 for reference
    ax1.axvline(0, color='gray', linestyle='--')

    # Set axis labels
    ax1.set_xlabel('Factor Aging Coefficient \n(Linear Regression)', fontsize=14)
    ax1.set_ylabel('Factors', fontsize=13)

    # Highlight specific factors in red
    if highlight_factors:
        y_ticks = ax1.get_yticklabels()
        for tick in y_ticks:
            if tick.get_text() in highlight_factors:
                tick.set_color('red')

    # Plot 2: Delta (Old - Young)
    ax2.barh(y=coefficients_with_deltas['Factor'], width=coefficients_with_deltas['Delta_OY'],
             color='lightcoral', label='Delta YO')
    ax2.axvline(0, color='gray', linestyle='--')

    # Set axis labels and limits for delta plot
    ax2.set_xlabel('Factor Cell Activity\nDelta (Old - Young)', fontsize=13)
    ax2.set_xlim(-0.08, 0.08)

    # Adjust tick labels for both axes
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"Plot saved at: {save_path}")

    # Show the plot
    plt.show()

def logistic_regression_feature_prediction(
    splice_adata,
    gene_expression_adata,
    feature: str,
    K: int = 50,
    n_pcs: int = 50,
    covariate_column='tissue',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train and evaluate multinomial logistic regression models using splicing factors,
    gene expression PCs, tissue covariate, and their combination to predict the feature of interest.

    Returns:
    - accuracies: dict, accuracy scores for each model.
    - models: dict, trained LogisticRegression models.
    - label_encoder: LabelEncoder instance used to encode the target variable.
    - coefficients_dfs: dict, DataFrames with logistic regression coefficients for each model.
    """

    ### **Step 1: Extract Splicing Latent Factors**
    if "X_PHI" not in splice_adata.obsm:
        raise ValueError("Splicing data is missing 'X_PHI' in obsm.")
    X_splicing = pd.DataFrame(splice_adata.obsm["X_PHI"], index=splice_adata.obs.index)
    X_splicing.columns = [f'factor_{i}' for i in range(X_splicing.shape[1])]

    ### **Step 2: Extract Gene Expression Principal Components**
    if "X_pca" not in gene_expression_adata.obsm:
        raise ValueError("Gene expression data is missing 'X_pca' in obsm.")
    pcs = gene_expression_adata.obsm["X_pca"][:, :n_pcs]
    X_gene_expression = pd.DataFrame(pcs, index=gene_expression_adata.obs.index)
    X_gene_expression.columns = [f'PC_{i+1}' for i in range(n_pcs)]

    ### **Step 3: Encode Tissue Covariate (if available)**
    if covariate_column in splice_adata.obs:
        tissue_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        tissue_covariate = tissue_encoder.fit_transform(splice_adata.obs[[covariate_column]])
        tissue_columns = [f'{covariate_column}_{cat}' for cat in tissue_encoder.categories_[0]]
        X_tissue = pd.DataFrame(tissue_covariate, index=splice_adata.obs.index, columns=tissue_columns)
    else:
        print(f"Warning: '{covariate_column}' not found in splice_adata. Skipping tissue covariate.")
        X_tissue = pd.DataFrame(index=splice_adata.obs.index)  # Empty DataFrame

    ### **Step 4: Merge Features**
    X_splicing = pd.concat([X_splicing, X_tissue], axis=1)
    X_gene_expression = pd.concat([X_gene_expression, X_tissue], axis=1)
    X_combined = pd.concat([X_splicing, X_gene_expression], axis=1)

    ### **Step 5: Standardize Features**
    scaler = StandardScaler()
    X_combined_scaled = pd.DataFrame(scaler.fit_transform(X_combined), index=X_combined.index, columns=X_combined.columns)

    ### **Step 6: Encode Target Feature**
    if feature not in splice_adata.obs:
        raise ValueError(f"Feature '{feature}' not found in splice_adata.obs.")

    label_encoder = LabelEncoder()
    y = splice_adata.obs[feature].astype(str)  # Convert to string before encoding
    y_encoded = label_encoder.fit_transform(y)

    ### **Step 7: Train-Test Split**
    data_splits = {
        "splicing": train_test_split(X_splicing, y_encoded, test_size=test_size, random_state=random_state),
        "gene_expression": train_test_split(X_gene_expression, y_encoded, test_size=test_size, random_state=random_state),
        "combined": train_test_split(X_combined_scaled, y_encoded, test_size=test_size, random_state=random_state),
        "tissue_only": train_test_split(X_tissue, y_encoded, test_size=test_size, random_state=random_state),
    }

    ### **Step 8: Train Logistic Regression Models**
    logreg_params = {'multi_class': 'multinomial', 'solver': 'saga', 'max_iter': 100, 'random_state': random_state}
    models = {name: LogisticRegression(**logreg_params) for name in data_splits.keys()}

    for name, (X_train, X_test, y_train, y_test) in data_splits.items():
        models[name].fit(X_train, y_train)

    ### **Step 9: Evaluate Models**
    accuracies = {
        name: accuracy_score(y_test, models[name].predict(X_test))
        for name, (_, X_test, _, y_test) in data_splits.items()
    }

    ### **Step 10: Train Models on Full Dataset for Coefficients**
    for name, (X_train, _, y_train, _) in data_splits.items():
        models[name].fit(X_train, y_train)

    ### **Step 11: Extract Coefficients**
    coefficients_dfs = {
        name: pd.DataFrame(models[name].coef_, columns=X_train.columns, index=label_encoder.classes_)
        for name, (X_train, _, _, _) in data_splits.items()
    }

    ### **Step 12: Return Results**
    return accuracies, models, label_encoder, coefficients_dfs


def plot_clustermap(coefficients, highlighted_factors=None, cmap="seismic", figsize=(6, 7), center=0, col_cluster=False, save_plot=False):
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
                       row_cluster=True, col_cluster=col_cluster, xticklabels=1, yticklabels=1, center=center)

    # Set y-axis tick labels (rows) font size and rotation
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=14)

    # Set x-axis tick labels (columns) font size
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=15)

    # Disable color bar visibility
    g.cax.set_visible(True)

    # Highlight specific factors (rows)
    if highlighted_factors:
        for label in g.ax_heatmap.xaxis.get_majorticklabels():
            if label.get_text() in highlighted_factors:
                label.set_weight('bold')   # Make it bold
                label.set_color('red')     # Change color to red

    # Set colorbar font size
    g.cax.tick_params(labelsize=15)

    if save_plot:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"factor_age_coefficients_{date_str}.pdf"
        plt.savefig(filename, format='pdf')
        print(f"Figure saved as {filename}")

    # Show the plot
    plt.show()

def plot_umap_with_junctions_and_factors(adata, junction_ids, factors, PSI_layer="PSI_CELLS", PHI_matrix=None, meta_columns=["tissue", "age"], size=8, ncols=2, wspace=0.1):
    """
    Plots UMAP with selected junction PSI values and factor usage across cells.
    
    Parameters:
    adata : AnnData
        The AnnData object containing UMAP coordinates and PSI values.
    junction_ids : list
        List of junction IDs to extract from `adata.layers[PSI_layer]` and store in `adata.obs`.
    factors : list
        List of factor indices to extract from `PHI_matrix` and store in `adata.obs`.
    PSI_layer : str, optional
        The layer containing PSI values (default is "PSI_CELLS").
    PHI_matrix : numpy.ndarray, optional
        The matrix containing factor usage (should match cell count in `adata`).
    meta_columns : list, optional
        List of column names in `adata.obs` to include in the UMAP plot (default is ["tissue", "age"]).
    size : float, optional
        Marker size for UMAP plot (default is 8).
    ncols : int, optional
        Number of columns for subplot arrangement (default is 2).
    wspace : float, optional
        Space between subplots (default is 0.1).
    """
    
    # Add junction PSI values to adata.obs
    for junction_id in junction_ids:
        adata.obs[f"junction_{junction_id}"] = adata.layers[PSI_layer][:, junction_id]
    
    # Add factor usage to adata.obs if PHI_matrix is provided
    if PHI_matrix is not None:
        for factor in factors:
            adata.obs[f"factor_{factor}"] = PHI_matrix[:, factor]
    
    # Define colors for UMAP visualization
    color_vars = meta_columns + [f"junction_{j}" for j in junction_ids] + [f"factor_{f}" for f in factors]
    
    # Plot UMAP
    sc.pl.umap(adata, color=color_vars, wspace=wspace, size=size, ncols=ncols)
    plt.show()

    # Remove newly made obs columns 
    for junction_id in junction_ids:
        del adata.obs[f"junction_{junction_id}"]
    for factor in factors:
        del adata.obs[f"factor_{factor}"]
    
def plot_violin_by_cell_type(adata, feature="junction", factor_idx=None, junction_idx=None, gene_name=None, cell_type_column="cell_type_grouped", cell_type=None, group_column="age", size=(6, 6)):
    """
    Plots a violin plot of a given feature (factor or junction activity) in a specified cell type across a grouping variable.
    """
    
    # Add junction or factor activity to adata.obs for easy plotting 
    if feature == "junction":
        if junction_idx is None:
            raise ValueError("Please provide a junction index.")
        adata.obs[feature] = adata.layers["PSI_CELLS"][:, junction_idx]
    elif feature == "factor":
        if factor_idx is None:
            raise ValueError("Please provide a factor index.")
        adata.obs[feature] = adata.obsm["X_PHI"][:, factor_idx]
    elif feature == "gene":
        if gene_name is None:
            raise ValueError("Please provide a gene name.")
        adata.obs[feature] = adata[:, gene_name].X.toarray()        
    else:
        raise ValueError("Invalid feature. Please use 'junction' or 'factor'.")
   
    # Filter for the specified cell type if cell_type is not None
    if cell_type is not None:
        subset_cells = adata[adata.obs[cell_type_column] == cell_type]
    else:
        subset_cells = adata
    
    # Create the violin plot
    plt.figure(figsize=size)
    ax = sns.violinplot(x=group_column, y=feature, data=subset_cells.obs, hue=group_column, inner=None)
    
    # Calculate median and mean per group
    medians = subset_cells.obs.groupby(group_column)[feature].median()
    means = subset_cells.obs.groupby(group_column)[feature].mean()
    
    # Annotate median and mean values on the plot
    for i, (median, mean) in enumerate(zip(medians, means)):
        plt.text(i, median, f"Median: {median:.2f}", ha="center", va="bottom", fontsize=10, color="black", fontweight="bold")
    
    # Labels and title
    if feature == "junction":
        plt.xlabel(group_column.capitalize(), fontsize=12)
        plt.ylabel(f"{feature} {junction_idx} PSI", fontsize=12)
        plt.title(f"{feature} {junction_idx} PSI in {cell_type}", fontsize=14)
    elif feature == "factor":
        plt.xlabel(group_column.capitalize(), fontsize=12)
        plt.ylabel(f"{feature} {factor_idx} Activity", fontsize=12)
        plt.title(f"{feature} {factor_idx} Activity in {cell_type}", fontsize=14)
    elif feature == "gene":
        plt.xlabel(group_column.capitalize(), fontsize=12)
        plt.ylabel(f"{gene_name} Expression", fontsize=12)
        plt.title(f"{gene_name} Expression in {cell_type}", fontsize=14)
        print(f"The number of cells that have non-zero expression of {gene_name} is {len(subset_cells.obs[subset_cells.obs[feature] > 0])}")
    plt.show()

# Example usage:
# plot_violin_by_cell_type(adata, feature="factor_10", cell_type_column="cell_type_grouped", cell_type="MICROGLIA", group_column="age", size=(6,6))
# plot_violin_by_cell_type(adata, feature="junction_30510", cell_type_column="cell_type_grouped", cell_type="MICROGLIA", group_column="age", size=(6,6))
