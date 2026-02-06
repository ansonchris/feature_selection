import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for display (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. Prepare data (Replace this with your own data)
def prepare_data():
    """Prepare example data"""
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate feature data
    X = np.random.randn(n_samples, n_features)
    
    # Create correlated features (first 5 features are related to target)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 3 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(n_samples) * 0.5
    
    # Add noise features
    X[:, 5:] = X[:, 5:] + np.random.randn(n_samples, n_features-5) * 2
    
    # Create feature names
    feature_names = [f'X{i}' for i in range(n_features)]
    
    return X, y, feature_names

# 2. L1 Regularization (Lasso) for feature selection
def lasso_feature_selection(X, y, feature_names, alpha=0.1):
    """Use Lasso for feature selection"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create Lasso model
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Get coefficients
    coefficients = lasso.coef_
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Filter features with non-zero coefficients
    selected_features = result_df[result_df['coefficient'] != 0]['feature'].tolist()
    
    return lasso, result_df, selected_features

# 3. L2 Regularization (Ridge) for feature importance assessment
def ridge_feature_importance(X, y, feature_names, alpha=1.0):
    """Use Ridge to assess feature importance"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create Ridge model
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_scaled, y)
    
    # Get coefficients
    coefficients = ridge.coef_
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    return ridge, result_df

# 4. ElasticNet for feature selection
def elasticnet_feature_selection(X, y, feature_names, alpha=0.1, l1_ratio=0.5):
    """Use ElasticNet for feature selection"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create ElasticNet model
    elastic_net = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=10000
    )
    elastic_net.fit(X_scaled, y)
    
    # Get coefficients
    coefficients = elastic_net.coef_
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Filter features with non-zero coefficients
    selected_features = result_df[result_df['coefficient'] != 0]['feature'].tolist()
    
    return elastic_net, result_df, selected_features

# 5. Cross-validation to find optimal parameters
def find_best_parameters(X, y, model_type='lasso'):
    """Use cross-validation to find optimal parameters"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grid
    if model_type == 'lasso':
        model = Lasso(random_state=42, max_iter=10000)
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    elif model_type == 'ridge':
        model = Ridge(random_state=42)
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    elif model_type == 'elasticnet':
        model = ElasticNet(random_state=42, max_iter=10000)
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    
    # Grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_scaled, y)
    
    return grid_search.best_params_, grid_search.best_score_

# 6. Visualization of results
def visualize_results(lasso_df, ridge_df, elasticnet_df):
    """Visualize feature coefficients from three methods"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Lasso coefficients plot
    ax1 = axes[0]
    lasso_df_sorted = lasso_df.sort_values('abs_coefficient', ascending=False)
    colors = ['red' if coef == 0 else 'blue' for coef in lasso_df_sorted['coefficient']]
    ax1.bar(range(len(lasso_df_sorted)), lasso_df_sorted['coefficient'], color=colors)
    ax1.set_xticks(range(len(lasso_df_sorted)))
    ax1.set_xticklabels(lasso_df_sorted['feature'], rotation=45, ha='right')
    ax1.set_title('Lasso Feature Coefficients (Red = Eliminated Features)')
    ax1.set_ylabel('Coefficient Value')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    # Ridge coefficients plot
    ax2 = axes[1]
    ridge_df_sorted = ridge_df.sort_values('abs_coefficient', ascending=False)
    ax2.bar(range(len(ridge_df_sorted)), ridge_df_sorted['coefficient'], color='green')
    ax2.set_xticks(range(len(ridge_df_sorted)))
    ax2.set_xticklabels(ridge_df_sorted['feature'], rotation=45, ha='right')
    ax2.set_title('Ridge Feature Coefficients')
    ax2.set_ylabel('Coefficient Value')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    # ElasticNet coefficients plot
    ax3 = axes[2]
    elasticnet_df_sorted = elasticnet_df.sort_values('abs_coefficient', ascending=False)
    colors = ['red' if coef == 0 else 'orange' for coef in elasticnet_df_sorted['coefficient']]
    ax3.bar(range(len(elasticnet_df_sorted)), elasticnet_df_sorted['coefficient'], color=colors)
    ax3.set_xticks(range(len(elasticnet_df_sorted)))
    ax3.set_xticklabels(elasticnet_df_sorted['feature'], rotation=45, ha='right')
    ax3.set_title('ElasticNet Feature Coefficients (Red = Eliminated Features)')
    ax3.set_ylabel('Coefficient Value')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

# 7. Main function
def main():
    """Main function"""
    print("=" * 60)
    print("L1/L2/ElasticNet Feature Selection Demo")
    print("=" * 60)
    
    # Prepare data
    X, y, feature_names = prepare_data()
    print(f"Data Shape: X={X.shape}, y={y.shape}")
    print(f"Number of Features: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n1. Finding optimal parameters...")
    # Find optimal Lasso parameters
    best_lasso_params, lasso_score = find_best_parameters(X_train, y_train, 'lasso')
    print(f"Lasso Best Parameters: {best_lasso_params}, Best Score: {-lasso_score:.4f}")
    
    # Find optimal Ridge parameters
    best_ridge_params, ridge_score = find_best_parameters(X_train, y_train, 'ridge')
    print(f"Ridge Best Parameters: {best_ridge_params}, Best Score: {-ridge_score:.4f}")
    
    # Find optimal ElasticNet parameters
    best_en_params, en_score = find_best_parameters(X_train, y_train, 'elasticnet')
    print(f"ElasticNet Best Parameters: {best_en_params}, Best Score: {-en_score:.4f}")
    
    print("\n2. Using Lasso for feature selection...")
    lasso_model, lasso_df, lasso_selected = lasso_feature_selection(
        X_train, y_train, feature_names, alpha=best_lasso_params['alpha']
    )
    print(f"Number of Features Selected by Lasso: {len(lasso_selected)}")
    print(f"Lasso Selected Features: {lasso_selected}")
    
    print("\n3. Using Ridge for feature importance assessment...")
    ridge_model, ridge_df = ridge_feature_importance(
        X_train, y_train, feature_names, alpha=best_ridge_params['alpha']
    )
    print("Ridge Feature Importance Ranking:")
    print(ridge_df.head(10))
    
    print("\n4. Using ElasticNet for feature selection...")
    elasticnet_model, elasticnet_df, elasticnet_selected = elasticnet_feature_selection(
        X_train, y_train, feature_names, 
        alpha=best_en_params['alpha'], 
        l1_ratio=best_en_params['l1_ratio']
    )
    print(f"Number of Features Selected by ElasticNet: {len(elasticnet_selected)}")
    print(f"ElasticNet Selected Features: {elasticnet_selected}")
    
    print("\n5. Evaluating model performance on test set...")
    # Standardize test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lasso performance
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    
    # Ridge performance
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    # ElasticNet performance
    elasticnet_model.fit(X_train_scaled, y_train)
    y_pred_en = elasticnet_model.predict(X_test_scaled)
    mse_en = mean_squared_error(y_test, y_pred_en)
    r2_en = r2_score(y_test, y_pred_en)
    
    print(f"\nModel Performance Comparison:")
    print(f"{'Model':<15} {'MSE':<15} {'R²':<15}")
    print(f"{'-'*45}")
    print(f"{'Lasso':<15} {mse_lasso:<15.4f} {r2_lasso:<15.4f}")
    print(f"{'Ridge':<15} {mse_ridge:<15.4f} {r2_ridge:<15.4f}")
    print(f"{'ElasticNet':<15} {mse_en:<15.4f} {r2_en:<15.4f}")
    
    print("\n6. Feature coefficient comparison analysis...")
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'lasso_coef': lasso_df['coefficient'],
        'ridge_coef': ridge_df['coefficient'],
        'elasticnet_coef': elasticnet_df['coefficient']
    })
    
    # Add flags for feature selection status
    comparison_df['lasso_selected'] = comparison_df['lasso_coef'] != 0
    comparison_df['elasticnet_selected'] = comparison_df['elasticnet_coef'] != 0
    
    print("\nFeature Coefficient Comparison (Top 10 Features):")
    print(comparison_df.head(10))
    
    print("\n7. Common selected features:")
    common_features = set(lasso_selected) & set(elasticnet_selected)
    print(f"Common Features Selected by Lasso and ElasticNet: {list(common_features)}")
    
    # Visualization
    visualize_results(lasso_df, ridge_df, elasticnet_df)
    
    return {
        'lasso': {
            'model': lasso_model,
            'selected_features': lasso_selected,
            'coefficients': lasso_df,
            'performance': {'mse': mse_lasso, 'r2': r2_lasso}
        },
        'ridge': {
            'model': ridge_model,
            'coefficients': ridge_df,
            'performance': {'mse': mse_ridge, 'r2': r2_ridge}
        },
        'elasticnet': {
            'model': elasticnet_model,
            'selected_features': elasticnet_selected,
            'coefficients': elasticnet_df,
            'performance': {'mse': mse_en, 'r2': r2_en}
        },
        'comparison': comparison_df
    }

# 8. Usage example
if __name__ == "__main__":
    # Run main function
    results = main()
    
    # Save results to file
    results['comparison'].to_csv('feature_selection_results.csv', index=False)
    print("\nResults saved to 'feature_selection_results.csv'")