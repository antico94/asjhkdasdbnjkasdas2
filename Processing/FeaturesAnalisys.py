import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import tqdm
from typing import Dict, List, Tuple, Optional, Set, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.DataStorage import DataStorage


class FeatureAnalyzer:
    def __init__(self, config: Config, logger: Logger, data_storage: DataStorage):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.output_dir = "FeatureAnalysis"
        self.sql_table = "SelectedFeatures"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_theme(style="darkgrid")

    def load_data(self, pair: str, timeframe: str, dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for feature analysis"""
        print(f"Loading {dataset_type} data for {pair} {timeframe}...")
        self.logger.info(f"Loading {dataset_type} data for {pair} {timeframe}")

        try:
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X.empty or y.empty:
                self.logger.error(f"No data found for {pair} {timeframe} {dataset_type}")
                print(f"Error: No data found for {pair} {timeframe} {dataset_type}")
                return pd.DataFrame(), pd.DataFrame()

            # Store time column if present for temporal validation
            time_col = None
            if 'time' in X.columns:
                time_col = X['time'].copy()
                X = X.drop('time', axis=1)

            # Handle missing values
            if X.isna().any().any():
                self.logger.warning(f"Found missing values in features, filling with zeros")
                X = X.fillna(0)

            if y.isna().any().any():
                self.logger.warning(f"Found missing values in targets, filling with zeros")
                y = y.fillna(0)

            print(f"Loaded {len(X)} rows with {X.shape[1]} features")
            return X, y

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            print(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def analyze_correlation(self, X: pd.DataFrame, threshold: float = 0.85) -> Dict[str, List[str]]:
        """Identify highly correlated feature pairs"""
        print("Analyzing feature correlation...")
        self.logger.info("Analyzing feature correlation")

        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Find highly correlated features
            correlated_features = {}

            # Use tqdm for progress indication
            for i in tqdm.tqdm(range(len(corr_matrix.columns)), desc="Correlation Analysis"):
                col_name = corr_matrix.columns[i]
                # Look at correlation with features that come after this one
                corr_values = corr_matrix.iloc[i + 1:, i]
                high_corr = corr_values[corr_values > threshold]

                if not high_corr.empty:
                    correlated_cols = high_corr.index.tolist()
                    correlated_features[col_name] = correlated_cols

            # Plot correlation heatmap for top features
            self._plot_correlation_heatmap(X)

            print(f"Found {sum(len(v) for v in correlated_features.values())} highly correlated feature pairs")
            return correlated_features

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            print(f"Error in correlation analysis: {e}")
            return {}

    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.DataFrame, target_col: str,
                                   is_classification: bool = False) -> pd.DataFrame:
        """Calculate feature importance using ensemble methods"""
        task_type = "classification" if is_classification else "regression"
        print(f"Calculating feature importance for {task_type} target: {target_col}...")
        self.logger.info(f"Calculating feature importance for {task_type} target: {target_col}")

        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Select appropriate model based on task type
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                               random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                              random_state=42, n_jobs=-1)

            # Time series cross-validation for more robust importance
            tscv = TimeSeriesSplit(n_splits=5)
            importance_scores = np.zeros(X.shape[1])

            for train_idx, test_idx in tqdm.tqdm(tscv.split(X_scaled),
                                                 total=5,
                                                 desc="Cross-validation"):
                X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx][target_col], y.iloc[test_idx][target_col]

                model.fit(X_train, y_train)
                importance_scores += model.feature_importances_

            # Average importance across folds
            importance_scores /= 5

            # Create and sort importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance_scores
            }).sort_values(by='Importance', ascending=False)

            # Plot top 20 features
            self._plot_feature_importance(feature_importance, target_col, task_type)

            print(f"Top 5 features: {', '.join(feature_importance['Feature'].head(5).tolist())}")
            return feature_importance

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            print(f"Error calculating feature importance: {e}")
            return pd.DataFrame(columns=['Feature', 'Importance'])

    def select_features_rfe(self, X: pd.DataFrame, y: pd.DataFrame, target_col: str,
                            n_features: int = 30, is_classification: bool = False) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        task_type = "classification" if is_classification else "regression"
        print(f"Running RFE to select top {n_features} features for {task_type} target: {target_col}...")
        self.logger.info(f"Running RFE to select top {n_features} features for {task_type}")

        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Select appropriate model based on task type
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, max_depth=10,
                                               random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10,
                                              random_state=42, n_jobs=-1)

            # Apply RFE with progress reporting via verbose
            rfe = RFE(estimator=model,
                      n_features_to_select=min(n_features, X.shape[1]),
                      step=3,
                      verbose=1)

            # Fit RFE
            print("Fitting RFE (this may take a while)...")
            rfe.fit(X_scaled, y[target_col])

            # Get selected features
            selected_features = X.columns[rfe.support_].tolist()

            print(f"RFE selected {len(selected_features)} features")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in RFE feature selection: {e}")
            print(f"Error in RFE feature selection: {e}")
            return []

    def select_features_statistical(self, X: pd.DataFrame, y: pd.DataFrame, target_col: str,
                                    k: int = 30, is_classification: bool = False) -> List[str]:
        """Select features using statistical tests (F-test)"""
        task_type = "classification" if is_classification else "regression"
        print(f"Selecting top {k} features using statistical tests for {task_type}...")
        self.logger.info(f"Selecting features using statistical tests for {task_type}")

        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Select appropriate scoring function based on task type
            score_func = f_classif if is_classification else f_regression

            # Apply SelectKBest
            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            selector.fit(X_scaled, y[target_col])

            # Get selected features
            selected_features = X.columns[selector.get_support()].tolist()

            # Get scores for visualization
            scores = selector.scores_
            feature_scores = pd.DataFrame({
                'Feature': X.columns,
                'Score': scores
            }).sort_values(by='Score', ascending=False)

            # Plot statistical scores
            self._plot_statistical_scores(feature_scores, target_col, task_type)

            print(f"Statistical tests selected {len(selected_features)} features")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in statistical feature selection: {e}")
            print(f"Error in statistical feature selection: {e}")
            return []

    def remove_redundant_features(self, features: List[str], correlated_features: Dict[str, List[str]],
                                  importance_df: pd.DataFrame) -> List[str]:
        """Remove redundant features based on correlation and importance"""
        print("Removing redundant features...")
        self.logger.info("Removing redundant features")

        try:
            features_to_keep = set(features)
            removed_features = set()

            # Filter correlation dict to only include features in our selected set
            filtered_corr = {}
            for feature, correlated in correlated_features.items():
                if feature in features_to_keep:
                    correlated_in_set = [f for f in correlated if f in features_to_keep]
                    if correlated_in_set:
                        filtered_corr[feature] = correlated_in_set

            print(f"Processing {len(filtered_corr)} feature groups for redundancy...")
            # Process each group of correlated features
            for feature, correlated in tqdm.tqdm(filtered_corr.items(), desc="Redundancy Removal"):
                if feature not in features_to_keep or feature in removed_features:
                    continue

                # Create a group of the feature and its correlated features
                group = [feature] + correlated
                group = [f for f in group if f in features_to_keep and f not in removed_features]

                if len(group) <= 1:
                    continue

                # Find the most important feature in the group
                group_importances = importance_df[importance_df['Feature'].isin(group)]

                if group_importances.empty:
                    continue

                most_important = group_importances.iloc[0]['Feature']

                # Keep only the most important feature from the group
                for feat in group:
                    if feat != most_important:
                        if feat in features_to_keep:
                            features_to_keep.remove(feat)
                            removed_features.add(feat)

            # Convert back to list and sort by importance
            sorted_features = []
            for feature in importance_df['Feature']:
                if feature in features_to_keep:
                    sorted_features.append(feature)

            print(f"Reduced from {len(features)} to {len(sorted_features)} features after redundancy removal")
            return sorted_features

        except Exception as e:
            self.logger.error(f"Error removing redundant features: {e}")
            print(f"Error removing redundant features: {e}")
            return features

    def validate_feature_set(self, X: pd.DataFrame, y: pd.DataFrame, target_col: str,
                             selected_features: List[str], is_classification: bool = False) -> float:
        """Validate the selected feature set using time series cross-validation"""
        task_type = "classification" if is_classification else "regression"
        print(f"Validating selected features for {task_type}...")
        self.logger.info(f"Validating selected features for {task_type}")

        try:
            if not selected_features:
                self.logger.warning("No features to validate")
                return 0.0

            # Subset to selected features
            X_selected = X[selected_features]

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Select appropriate model and metric based on task type
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, max_depth=10,
                                               random_state=42, n_jobs=-1)
                scores = []

                for train_idx, test_idx in tqdm.tqdm(tscv.split(X_selected),
                                                     total=5,
                                                     desc="CV Validation"):
                    X_train, X_test = (X_selected.iloc[train_idx],
                                       X_selected.iloc[test_idx])
                    y_train, y_test = (y.iloc[train_idx][target_col],
                                       y.iloc[test_idx][target_col])

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    scores.append(accuracy_score(y_test, y_pred))

                avg_score = np.mean(scores)
                print(f"Classification accuracy with selected features: {avg_score:.4f}")

            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10,
                                              random_state=42, n_jobs=-1)
                scores = []

                for train_idx, test_idx in tqdm.tqdm(tscv.split(X_selected),
                                                     total=5,
                                                     desc="CV Validation"):
                    X_train, X_test = (X_selected.iloc[train_idx],
                                       X_selected.iloc[test_idx])
                    y_train, y_test = (y.iloc[train_idx][target_col],
                                       y.iloc[test_idx][target_col])

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

                avg_score = np.mean(scores)
                print(f"Regression RMSE with selected features: {avg_score:.4f}")

            return avg_score

        except Exception as e:
            self.logger.error(f"Error validating feature set: {e}")
            print(f"Error validating feature set: {e}")
            return 0.0

    def ensure_sql_table_exists(self) -> bool:
        """Ensure the SQL table for storing selected features exists"""
        print("Checking database tables...")
        self.logger.info("Ensuring SQL table for features exists")

        try:
            conn_str = self.config.get_sql_connection_string()
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{self.sql_table}')
                    CREATE TABLE {self.sql_table} (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        model_type VARCHAR(20) NOT NULL,
                        feature_name VARCHAR(100) NOT NULL,
                        importance FLOAT DEFAULT 0,
                        created_at DATETIME DEFAULT GETDATE()
                    )
                """)
                conn.commit()
                print("Database table ready")
                return True

        except Exception as e:
            self.logger.error(f"Error ensuring SQL table exists: {e}")
            print(f"Error ensuring SQL table exists: {e}")
            return False

    def save_features_to_sql(self, features: List[str], importance_df: pd.DataFrame,
                             symbol: str, timeframe: str, model_type: str) -> bool:
        """Save selected features to SQL database with importance values"""
        print(f"Saving {len(features)} features to database...")
        self.logger.info(f"Saving {len(features)} features to database for {model_type}")

        if not features:
            self.logger.warning(f"No features to save for {symbol} {timeframe} {model_type}")
            return False

        try:
            conn_str = self.config.get_sql_connection_string()
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()

                # Delete existing features for this configuration
                cursor.execute(f"""
                    DELETE FROM {self.sql_table}
                    WHERE symbol = ? AND timeframe = ? AND model_type = ?
                """, (symbol, timeframe, model_type))

                # Prepare importance lookup
                importance_lookup = {row['Feature']: row['Importance']
                                     for _, row in importance_df.iterrows()}

                # Insert each feature with its importance
                for i, feature in enumerate(tqdm.tqdm(features, desc="Saving to DB")):
                    importance = importance_lookup.get(feature, 0.0)
                    cursor.execute(f"""
                        INSERT INTO {self.sql_table} 
                        (symbol, timeframe, model_type, feature_name, importance)
                        VALUES (?, ?, ?, ?, ?)
                    """, (symbol, timeframe, model_type, feature, importance))

                conn.commit()
                print(f"Successfully saved {len(features)} features to database")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save features to SQL: {e}")
            print(f"Error: Failed to save features to database: {e}")
            return False

    def cleanup_legacy_outputs(self) -> None:
        """Remove old output files before generating new ones"""
        print("Cleaning up old analysis files...")
        if os.path.exists(self.output_dir):
            try:
                # Only remove files, not the directory itself
                for file_name in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                self.logger.info("Removed old report files")
            except Exception as e:
                self.logger.warning(f"Could not remove old files: {e}")
                print(f"Warning: Could not remove old files: {e}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def save_summary_report(self, pair: str, timeframe: str,
                            direction_features: List[str], magnitude_features: List[str],
                            direction_importance: pd.DataFrame, magnitude_importance: pd.DataFrame) -> None:
        """Save a text summary report of the feature analysis"""
        report_path = os.path.join(self.output_dir, f"feature_analysis_report.txt")

        try:
            with open(report_path, 'w') as f:
                f.write(f"FEATURE ANALYSIS REPORT\n")
                f.write(f"======================\n\n")
                f.write(f"Symbol: {pair}\n")
                f.write(f"Timeframe: {timeframe}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write(f"DIRECTION MODEL FEATURES ({len(direction_features)})\n")
                f.write(f"-----------------------------\n")
                direction_importance = direction_importance[direction_importance['Feature'].isin(direction_features)]
                for i, row in direction_importance.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']:.6f}\n")

                f.write(f"\nMAGNITUDE MODEL FEATURES ({len(magnitude_features)})\n")
                f.write(f"-------------------------------\n")
                magnitude_importance = magnitude_importance[magnitude_importance['Feature'].isin(magnitude_features)]
                for i, row in magnitude_importance.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']:.6f}\n")

                f.write(f"\nTOP 20 FEATURES BY IMPORTANCE:\n")
                f.write(f"-----------------------------\n")
                combined_importance = pd.concat([direction_importance, magnitude_importance])
                combined_importance = combined_importance.drop_duplicates(subset='Feature')
                combined_importance = combined_importance.sort_values('Importance', ascending=False).head(20)
                for i, row in combined_importance.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']:.6f}\n")

                f.write(f"\nSELECTED FEATURES AFTER REDUNDANCY REMOVAL:\n")
                f.write(f"---------------------------------------\n")
                for feature in sorted(set(direction_features + magnitude_features)):
                    f.write(f"{feature}\n")

                f.write(f"\nHIGHLY CORRELATED FEATURE GROUPS:\n")
                f.write(f"------------------------------\n")
                f.write("See correlation heatmap image for details.\n")

            print(f"Feature analysis report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving summary report: {e}")
            print(f"Error saving summary report: {e}")

    def _plot_correlation_heatmap(self, X: pd.DataFrame) -> None:
        """Plot correlation heatmap for top features"""
        try:
            # Take top 20 features by variance for readability
            top_vars = X.var().sort_values(ascending=False).head(20).index
            X_top = X[top_vars]

            # Calculate correlation
            corr = X_top.corr()

            # Plot
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f",
                        square=True, linewidths=.5, vmin=-1, vmax=1)
            plt.title('Feature Correlation Heatmap (Top 20 by Variance)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'), dpi=300)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting correlation heatmap: {e}")

    def _plot_feature_importance(self, importance_df: pd.DataFrame, target: str, task_type: str) -> None:
        """Plot feature importance from ensemble model"""
        try:
            plt.figure(figsize=(12, 8))

            # Plot top 20 features
            top_features = importance_df.head(20).copy()
            top_features = top_features.sort_values('Importance')

            sns.barplot(x='Importance', y='Feature', data=top_features,
                        palette='viridis')

            plt.title(f'Feature Importance for {target} ({task_type.capitalize()})')
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(self.output_dir, f'{target}_{task_type}_importance.png'), dpi=300)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")

    def _plot_statistical_scores(self, scores_df: pd.DataFrame, target: str, task_type: str) -> None:
        """Plot statistical test scores for features"""
        try:
            plt.figure(figsize=(12, 8))

            # Plot top 20 features
            top_features = scores_df.head(20).copy()
            top_features = top_features.sort_values('Score')

            sns.barplot(x='Score', y='Feature', data=top_features,
                        palette='magma')

            test_type = "F-Classification" if task_type == "classification" else "F-Regression"
            plt.title(f'{test_type} Scores for {target}')
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(self.output_dir, f'{target}_{task_type}_fscores.png'), dpi=300)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting statistical scores: {e}")

    def run_complete_analysis(self, pair: str, timeframe: str, dataset_type: str,
                              target_col: str, is_classification: bool = False) -> Tuple[List[str], pd.DataFrame]:
        """Run complete feature selection workflow for a target"""
        print(f"{'-' * 60}")
        task_type = "Classification" if is_classification else "Regression"
        print(f"Starting {task_type} feature analysis for {target_col}")
        print(f"{'-' * 60}")

        self.logger.info(f"Starting complete feature analysis for {target_col} ({task_type})")

        try:
            # Load data
            X, y = self.load_data(pair, timeframe, dataset_type)
            if X.empty or y.empty or target_col not in y.columns:
                self.logger.warning(f"Skipping: No data or missing target {target_col}")
                print(f"Error: Target {target_col} not found in dataset")
                return [], pd.DataFrame()

            # Step 1: Analyze correlation between features
            correlated = self.analyze_correlation(X)

            # Step 2: Calculate feature importance
            importance_df = self.analyze_feature_importance(X, y, target_col, is_classification)

            # Step 3: Select features using RFE
            rfe_features = self.select_features_rfe(X, y, target_col, n_features=30,
                                                    is_classification=is_classification)

            # Step 4: Select features using statistical tests
            stat_features = self.select_features_statistical(X, y, target_col, k=30,
                                                             is_classification=is_classification)

            # Step 5: Find common features between RFE and statistical methods
            common_features = list(set(rfe_features).intersection(set(stat_features)))
            print(f"Found {len(common_features)} features common to both selection methods")

            # Step 6: Remove redundant features based on correlation
            reduced_corr = {k: [f for f in v if f in common_features]
                            for k, v in correlated.items() if k in common_features}
            final_features = self.remove_redundant_features(common_features, reduced_corr, importance_df)

            # Step 7: Validate final feature set
            self.validate_feature_set(X, y, target_col, final_features, is_classification)

            print(f"Selected {len(final_features)} final features for {target_col}")
            return final_features, importance_df

        except Exception as e:
            self.logger.error(f"Error in complete feature analysis: {e}")
            print(f"Error: Feature analysis failed: {e}")
            return [], pd.DataFrame()

    def run_dual_analysis(self, pair: str, timeframe: str, dataset_type: str,
                          direction_target: str = "direction_1",
                          magnitude_target: str = "future_price_1") -> Dict[str, List[str]]:
        """Run feature analysis for both direction and magnitude prediction"""
        print(f"\n{'=' * 80}")
        print(f"STARTING DUAL FEATURE ANALYSIS FOR {pair} {timeframe}")
        print(f"{'=' * 80}\n")

        self.logger.info(f"Starting dual feature analysis for {pair} {timeframe}")

        # Initial setup
        self.ensure_sql_table_exists()
        self.cleanup_legacy_outputs()

        # Run direction analysis (classification)
        direction_features, direction_importance = self.run_complete_analysis(
            pair, timeframe, dataset_type, direction_target, is_classification=True)

        # Run magnitude analysis (regression)
        magnitude_features, magnitude_importance = self.run_complete_analysis(
            pair, timeframe, dataset_type, magnitude_target, is_classification=False)

        # Save results to database
        if direction_features:
            self.save_features_to_sql(direction_features, direction_importance,
                                      pair, timeframe, "direction")

        if magnitude_features:
            self.save_features_to_sql(magnitude_features, magnitude_importance,
                                      pair, timeframe, "magnitude")

        # Save summary report
        self.save_summary_report(pair, timeframe,
                                 direction_features, magnitude_features,
                                 direction_importance, magnitude_importance)

        # Final summary
        print(f"\n{'=' * 80}")
        print(f"FEATURE ANALYSIS COMPLETE")
        print(f"Direction model: {len(direction_features)} features selected")
        print(f"Magnitude model: {len(magnitude_features)} features selected")
        print(f"Analysis results saved to {self.output_dir}/feature_analysis_report.txt")
        print(f"{'=' * 80}\n")

        # Return all selected features
        return {
            "direction": direction_features,
            "magnitude": magnitude_features
        }