import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisCategory(Enum):
    """Categories for analysis patterns"""
    STATISTICAL = "statistical"
    TIME_SERIES = "time_series"
    MACHINE_LEARNING = "machine_learning"
    VISUALIZATION = "visualization"
    DATA_QUALITY = "data_quality"
    
@dataclass
class AnalysisPattern:
    """Structure for analysis patterns"""
    category: str
    subcategory: str
    pattern_name: str
    description: str
    required_columns: Dict
    example_prompt: str
    analysis_steps: List[str]
    code_template: str
    library_imports: List[str]
    tags: List[str]

class BaseKnowledgeGenerator:
    def __init__(self, storage):
        """Initialize the knowledge base generator"""
        self.storage = storage
        # Use the data directory from storage
        self.output_dir = storage.data_dir / "base_knowledge" 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # MiniLM embedding dimension
        
        # Initialize storage paths
        self.db_path = self.output_dir / "base_knowledge.db"
        self.index_path = self.output_dir / "base_vectors.index"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BaseKnowledgeGenerator with output dir: {self.output_dir}")

    def initialize_database(self):
        """Create the SQLite database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_patterns (
                        id INTEGER PRIMARY KEY,
                        category TEXT NOT NULL,
                        subcategory TEXT NOT NULL,
                        pattern_name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        required_columns JSON NOT NULL,
                        example_prompt TEXT NOT NULL,
                        analysis_steps JSON NOT NULL,
                        code_template TEXT NOT NULL,
                        library_imports JSON NOT NULL,
                        tags JSON NOT NULL,
                        vector_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(category, subcategory, pattern_name)
                    )
                """)
                conn.commit()
                logger.info("Initialized database schema")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def generate_knowledge_base(self):
        """Generate complete knowledge base with all patterns"""
        try:
            logger.info("Starting knowledge base generation")
            self.initialize_database()
            
            # Initialize FAISS index
            index = faiss.IndexFlatL2(self.dimension)
            
            # Collect all patterns
            patterns = []
            pattern_generators = [
                self._generate_statistical_patterns(),
                self._generate_time_series_patterns(),
                self._generate_ml_patterns(),
                self._generate_visualization_patterns(),
                self._generate_data_quality_patterns(),
                self._generate_business_patterns()
            ]
            
            for generator in pattern_generators:
                if generator is not None:  # Add check for None
                    patterns.extend(generator)
            
            # Store patterns and build index
            with sqlite3.connect(self.db_path) as conn:
                for i, pattern in enumerate(patterns):
                    try:
                        # Generate embedding for the pattern
                        text_to_embed = f"{pattern.pattern_name} {pattern.description} {pattern.example_prompt}"
                        embedding = self.model.encode([text_to_embed])[0]
                        
                        # Add to FAISS index
                        index.add(np.array([embedding]).astype('float32'))
                        
                        # Store in database
                        conn.execute("""
                            INSERT INTO analysis_patterns (
                                category, subcategory, pattern_name, description,
                                required_columns, example_prompt, analysis_steps,
                                code_template, library_imports, tags, vector_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pattern.category,
                            pattern.subcategory,
                            pattern.pattern_name,
                            pattern.description,
                            json.dumps(pattern.required_columns),
                            pattern.example_prompt,
                            json.dumps(pattern.analysis_steps),
                            pattern.code_template,
                            json.dumps(pattern.library_imports),
                            json.dumps(pattern.tags),
                            i
                        ))
                        
                        logger.debug(f"Added pattern: {pattern.pattern_name}")
                        
                    except Exception as e:
                        logger.error(f"Error processing pattern {pattern.pattern_name}: {str(e)}")
                        continue
                
                conn.commit()
            
            # Save FAISS index
            faiss.write_index(index, str(self.index_path))
            
            logger.info(f"Successfully generated knowledge base with {len(patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error generating knowledge base: {str(e)}")
            raise

    def _generate_statistical_patterns(self) -> List[AnalysisPattern]:
        """Generate statistical analysis patterns"""
        patterns = []
        
        # Basic Statistical Analysis
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.STATISTICAL.value,
            subcategory="descriptive",
            pattern_name="Basic Statistical Analysis",
            description="Comprehensive statistical summary of numeric columns",
            required_columns={"numeric": {"min_count": 1}},
            example_prompt="Show me basic statistics for all numeric columns",
            analysis_steps=[
                "Calculate descriptive statistics",
                "Generate distribution plots",
                "Identify outliers",
                "Test for normality"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import seaborn as sns
                import matplotlib.pyplot as plt
                from scipy import stats
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Basic statistics
                stats_df = df[numeric_cols].describe()
                
                # Add additional statistics
                stats_df.loc['skew'] = df[numeric_cols].skew()
                stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
                
                # Distribution plots
                for col in numeric_cols:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.show()
                
                # Outlier detection
                z_scores = np.abs(stats.zscore(df[numeric_cols]))
                outliers = (z_scores > 3).any(axis=1)
                
                # Normality tests
                normality_tests = {}
                for col in numeric_cols:
                    _, p_value = stats.normaltest(df[col].dropna())
                    normality_tests[col] = {
                        'is_normal': p_value > 0.05,
                        'p_value': p_value
                    }
                
                return {
                    'statistics': stats_df,
                    'outliers_count': outliers.sum(),
                    'normality_tests': normality_tests
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'seaborn',
                'scipy'
            ],
            tags=[
                'statistics',
                'visualization',
                'outliers',
                'distribution'
            ]
        ))
        
        # Correlation Analysis
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.STATISTICAL.value,
            subcategory="correlation",
            pattern_name="Correlation Analysis",
            description="Analyze relationships between numeric variables",
            required_columns={"numeric": {"min_count": 2}},
            example_prompt="Show me correlations between variables",
            analysis_steps=[
                "Calculate correlation matrix",
                "Generate correlation heatmap",
                "Identify strong correlations",
                "Create scatter plots"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Calculate correlations
                corr_matrix = df[numeric_cols].corr()
                
                # Plot correlation heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f'
                )
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.show()
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i,j]) > 0.7:
                            strong_corr.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': corr_matrix.iloc[i,j]
                            })
                
                # Scatter plots for strong correlations
                for corr in strong_corr:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        data=df,
                        x=corr['var1'],
                        y=corr['var2']
                    )
                    plt.title(f"Correlation: {corr['correlation']:.2f}")
                    plt.tight_layout()
                    plt.show()
                
                return {
                    'correlation_matrix': corr_matrix,
                    'strong_correlations': strong_corr
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'seaborn',
                'matplotlib'
            ],
            tags=[
                'correlation',
                'relationships',
                'visualization',
                'heatmap'
            ]
        ))
        
        # Hypothesis Testing
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.STATISTICAL.value,
            subcategory="hypothesis_testing",
            pattern_name="Statistical Hypothesis Testing",
            description="Perform various statistical tests to compare groups and verify assumptions",
            required_columns={
                "numeric": {"min_count": 1},
                "categorical": {"min_count": 1, "optional": True}
            },
            example_prompt="Test if there are significant differences between groups",
            analysis_steps=[
                "Check normality assumptions",
                "Perform appropriate statistical tests",
                "Calculate effect sizes",
                "Visualize group differences"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                results = {}
                
                # Single sample tests
                for col in numeric_cols:
                    data = df[col].dropna()
                    
                    # Normality test
                    _, normality_pval = stats.normaltest(data)
                    
                    # One-sample t-test or Wilcoxon signed-rank test
                    population_mean = data.mean()  # Example: test against mean
                    if normality_pval > 0.05:
                        t_stat, t_pval = stats.ttest_1samp(data, population_mean)
                        test_type = 'one_sample_ttest'
                        test_stat = t_stat
                        p_val = t_pval
                    else:
                        w_stat, w_pval = stats.wilcoxon(data - population_mean)
                        test_type = 'wilcoxon'
                        test_stat = w_stat
                        p_val = w_pval
                    
                    results[f'single_sample_{col}'] = {
                        'test_type': test_type,
                        'test_statistic': test_stat,
                        'p_value': p_val,
                        'is_normal': normality_pval > 0.05
                    }
                
                # Group comparison tests (if categorical columns exist)
                if len(categorical_cols) > 0:
                    for cat_col in categorical_cols:
                        for num_col in numeric_cols:
                            groups = [group for _, group in df.groupby(cat_col)[num_col]]
                            n_groups = len(groups)
                            
                            if n_groups < 2:
                                continue
                            
                            # Test normality for each group
                            normality_results = [stats.normaltest(group) for group in groups]
                            groups_normal = all(p > 0.05 for _, p in normality_results)
                            
                            # Test homogeneity of variance
                            _, levene_p = stats.levene(*groups)
                            equal_var = levene_p > 0.05
                            
                            if n_groups == 2:
                                # Two groups: t-test or Mann-Whitney U
                                if groups_normal and equal_var:
                                    stat, pval = stats.ttest_ind(*groups)
                                    test_type = 'ttest'
                                else:
                                    stat, pval = stats.mannwhitneyu(*groups)
                                    test_type = 'mann_whitney'
                                

                                
                                results[f'comparison_{cat_col}_{num_col}'] = {
                                    'test_type': test_type,
                                    'test_statistic': stat,
                                    'p_value': pval,
                                    'effect_size': d
                                }
                                
                                # Visualization
                                plt.figure(figsize=(10, 6))
                                sns.boxplot(data=df, x=cat_col, y=num_col)
                                plt.title(f'{test_type}: p={pval:.4f}, d={d:.2f}')
                                plt.show()
                                
                            else:
                                # Multiple groups: ANOVA or Kruskal-Wallis
                                if groups_normal and equal_var:
                                    stat, pval = stats.f_oneway(*groups)
                                    test_type = 'anova'
                                    
                                    # Post-hoc Tukey HSD
                                    tukey = pairwise_tukeyhsd(
                                        df[num_col].values,
                                        df[cat_col].values
                                    )
                                    posthoc_results = pd.DataFrame(
                                        data=tukey._results_table.data[1:],
                                        columns=tukey._results_table.data[0]
                                    )
                                else:
                                    stat, pval = stats.kruskal(*groups)
                                    test_type = 'kruskal'
                                    posthoc_results = None
                                
                                # Effect size (Eta-squared)
                                ss_between = sum(len(g) * (np.mean(g) - np.mean(df[num_col])) ** 2 
                                               for g in groups)
                                ss_total = sum((df[num_col] - np.mean(df[num_col])) ** 2)
                                eta_squared = ss_between / ss_total
                                
                                results[f'comparison_{cat_col}_{num_col}'] = {
                                    'test_type': test_type,
                                    'test_statistic': stat,
                                    'p_value': pval,
                                    'effect_size': eta_squared,
                                    'posthoc': (
                                        posthoc_results.to_dict() 
                                        if posthoc_results is not None else None
                                    )
                                }
                                
                                # Visualization
                                plt.figure(figsize=(12, 6))
                                sns.boxplot(data=df, x=cat_col, y=num_col)
                                plt.title(f'{test_type}: p={pval:.4f}, η²={eta_squared:.2f}')
                                plt.xticks(rotation=45)
                                plt.show()
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'seaborn',
                'statsmodels'
            ],
            tags=[
                'statistics',
                'hypothesis_testing',
                'anova',
                't_test',
                'effect_size'
            ]
        ))
        
        # Statistical Power Analysis
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.STATISTICAL.value,
            subcategory="power_analysis",
            pattern_name="Statistical Power Analysis",
            description="Calculate statistical power and required sample sizes for different tests",
            required_columns={"numeric": {"min_count": 1}},
            example_prompt="Calculate the power of my statistical tests and required sample sizes",
            analysis_steps=[
                "Calculate effect sizes",
                "Perform power analysis",
                "Generate power curves",
                "Estimate required sample sizes"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                from statsmodels.stats.power import TTestPower, FTestPower
                
                # Read data
                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                results = {}
                
                for col in numeric_cols:
                    data = df[col].dropna()
                    
                    # Calculate observed effect size (Cohen's d)
                    effect_size = (data.mean() - data.mean()) / data.std()
                    
                    # Power analysis for different sample sizes
                    n_range = np.linspace(10, 1000, 100)
                    power_analysis = TTestPower()
                    power = [power_analysis.power(effect_size=effect_size, 
                                                nobs=n, 
                                                alpha=0.05) 
                            for n in n_range]
                    
                    # Required sample size for different power levels
                    power_levels = [0.8, 0.9, 0.95]
                    required_n = {
                        f'power_{p}': power_analysis.solve_power(
                            effect_size=effect_size,
                            power=p,
                            alpha=0.05
                        )
                        for p in power_levels
                    }
                    
                    # Power curve plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(n_range, power)
                    plt.axhline(y=0.8, color='r', linestyle='--', label='0.8 power threshold')
                    plt.xlabel('Sample Size')
                    plt.ylabel('Statistical Power')
                    plt.title(f'Power Analysis Curve for {col}')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    
                    # Store results
                    results[col] = {
                        'effect_size': effect_size,
                        'required_sample_sizes': required_n,
                        'current_power': power_analysis.power(
                            effect_size=effect_size,
                            nobs=len(data),
                            alpha=0.05
                        )
                    }
                    
                    # Power vs Effect Size plot
                    effect_sizes = np.linspace(0.1, 1.0, 100)
                    power_by_effect = [power_analysis.power(
                        effect_size=es,
                        nobs=len(data),
                        alpha=0.05
                    ) for es in effect_sizes]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(effect_sizes, power_by_effect)
                    plt.axhline(y=0.8, color='r', linestyle='--', label='0.8 power threshold')
                    plt.xlabel('Effect Size')
                    plt.ylabel('Statistical Power')
                    plt.title(f'Power vs Effect Size for {col} (n={len(data)})')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    
                    # For two-sample tests
                    if len(df.columns) > 1:
                        # Example with another column
                        other_cols = [c for c in numeric_cols if c != col]
                        for other_col in other_cols:
                            other_data = df[other_col].dropna()
                            
                            # Calculate effect size between groups
                            d = (data.mean() - other_data.mean()) / \
                                np.sqrt((data.var() + other_data.var()) / 2)
                            
                            # Two-sample power analysis
                            two_sample_power = TTestPower().power(
                                effect_size=d,
                                nobs=min(len(data), len(other_data)),
                                alpha=0.05
                            )
                            
                            results[f'{col}_vs_{other_col}'] = {
                                'effect_size': d,
                                'power': two_sample_power,
                                'required_n_each_group': TTestPower().solve_power(
                                    effect_size=d,
                                    power=0.8,
                                    alpha=0.05
                                )
                            }
                
                # ANOVA power analysis if multiple groups exist
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for cat_col in categorical_cols:
                        for num_col in numeric_cols:
                            groups = df.groupby(cat_col)[num_col].apply(list)
                            if len(groups) > 2:
                                # Calculate effect size (f) for ANOVA
                                grand_mean = df[num_col].mean()
                                n = len(df[num_col])
                                n_groups = len(groups)
                                
                                # Between group variance
                                ss_between = sum(
                                    len(group) * (np.mean(group) - grand_mean) ** 2 
                                    for group in groups
                                )
                                
                                # Within group variance
                                ss_within = sum(
                                    sum((x - np.mean(group)) ** 2) 
                                    for group in groups
                                )
                                
                                df_between = n_groups - 1
                                df_within = n - n_groups
                                
                                ms_between = ss_between / df_between
                                ms_within = ss_within / df_within
                                
                                f_stat = ms_between / ms_within
                                
                                # Power analysis for ANOVA
                                power_analysis = FTestPower()
                                anova_power = power_analysis.power(
                                    f_stat,
                                    df_between,
                                    df_within,
                                    alpha=0.05
                                )
                                
                                results[f'anova_{cat_col}_{num_col}'] = {
                                    'f_statistic': f_stat,
                                    'power': anova_power,
                                    'required_n_per_group': power_analysis.solve_power(
                                        effect_size=f_stat,
                                        df_num=df_between,
                                        df_denom=df_within,
                                        power=0.8,
                                        alpha=0.05
                                    )
                                }
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'statsmodels'
            ],
            tags=[
                'statistics',
                'power_analysis',
                'effect_size',
                'sample_size'
            ]
        ))

        # Regression Analysis
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.STATISTICAL.value,
            subcategory="regression",
            pattern_name="Comprehensive Regression Analysis",
            description="Perform detailed regression analysis with diagnostics and assumptions testing",
            required_columns={
                "dependent": {"min_count": 1},
                "independent": {"min_count": 1}
            },
            example_prompt="Run a regression analysis with full diagnostics",
            analysis_steps=[
                "Prepare data and check assumptions",
                "Fit regression models",
                "Analyze residuals",
                "Calculate diagnostics",
                "Generate prediction intervals"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                from statsmodels.stats.diagnostic import linear_rainbow
                from statsmodels.stats.stattools import durbin_watson
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools.tools import add_constant
                
                # Read data
                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Assuming last numeric column is dependent variable
                y_col = numeric_cols[-1]
                X_cols = numeric_cols[:-1]
                
                y = df[y_col]
                X = df[X_cols]
                
                results = {}
                
                # Check multicollinearity
                X_with_const = add_constant(X)
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X_with_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                                  for i in range(X_with_const.shape[1])]
                
                # Fit model
                model = OLS(y, X_with_const).fit()
                predictions = model.predict(X_with_const)
                residuals = y - predictions
                
                # Basic regression results
                results['model_summary'] = {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue,
                    'coefficients': model.params.to_dict(),
                    'std_errors': model.bse.to_dict(),
                    'p_values': model.pvalues.to_dict()
                }
                
                # Residual analysis
                results['residual_analysis'] = {
                    'normality_test': stats.normaltest(residuals),
                    'durbin_watson': durbin_watson(residuals),
                    'homoscedasticity_test': linear_rainbow(model),
                    'residual_stats': {
                        'mean': residuals.mean(),
                        'std': residuals.std(),
                        'skew': stats.skew(residuals),
                        'kurtosis': stats.kurtosis(residuals)
                    }
                }
                
                # Diagnostic plots
                # 1. Residuals vs Fitted
                plt.figure(figsize=(12, 8))
                plt.subplot(221)
                plt.scatter(predictions, residuals)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Fitted values')
                plt.ylabel('Residuals')
                plt.title('Residuals vs Fitted')
                
                # 2. Q-Q plot
                plt.subplot(222)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q plot')
                
                # 3. Scale-Location
                plt.subplot(223)
                plt.scatter(predictions, np.sqrt(np.abs(stats.zscore(residuals))))
                plt.xlabel('Fitted values')
                plt.ylabel('Sqrt(|Standardized residuals|)')
                plt.title('Scale-Location')
                
                # 4. Leverage plot
                plt.subplot(224)
                influence = model.get_influence()
                leverage = influence.hat_matrix_diag
                plt.scatter(leverage, stats.zscore(residuals))
                plt.xlabel('Leverage')
                plt.ylabel('Standardized residuals')
                plt.title('Residuals vs Leverage')
                
                plt.tight_layout()
                plt.show()
                
                # Coefficients plot
                plt.figure(figsize=(10, 6))
                coef_df = pd.DataFrame({
                    'coef': model.params.values[1:],
                    'std': model.bse.values[1:],
                    'variable': X_cols
                })
                
                plt.errorbar(
                    x=coef_df['coef'],
                    y=range(len(coef_df)),
                    xerr=coef_df['std'] * 1.96,
                    fmt='o'
                )
                plt.yticks(range(len(coef_df)), coef_df['variable'])
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Coefficient Plot (95% CI)')
                plt.xlabel('Coefficient value')
                plt.tight_layout()
                plt.show()
                
                # Prediction intervals
                prediction_interval = model.get_prediction(X_with_const).conf_int()
                
                results['predictions'] = {
                    'fitted_values': predictions.to_dict(),
                    'prediction_intervals': {
                        'lower': prediction_interval[:, 0].tolist(),
                        'upper': prediction_interval[:, 1].tolist()
                    }
                }
                
                # Feature importance
                results['feature_importance'] = {
                    'standardized_coef': (
                        model.params[1:] * X.std().values / y.std()
                    ).to_dict(),
                    'vif': vif_data.to_dict()
                }
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'seaborn',
                'statsmodels'
            ],
            tags=[
                'regression',
                'diagnostics',
                'statistical_testing',
                'visualization'
            ]
        ))
        
        return patterns

    def _generate_time_series_patterns(self) -> List[AnalysisPattern]:
        """Generate time series analysis patterns"""
        patterns = []
        
        # Time Series Decomposition
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.TIME_SERIES.value,
            subcategory="decomposition",
            pattern_name="Time Series Decomposition",
            description="Decompose time series into trend, seasonal, and residual components",
            required_columns={
                "datetime": {"min_count": 1},
                "numeric": {"min_count": 1}
            },
            example_prompt="Show me trends and seasonality in the time series",
            analysis_steps=[
                "Convert to datetime index",
                "Perform seasonal decomposition",
                "Plot components",
                "Calculate trend statistics"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from statsmodels.tsa.seasonal import seasonal_decompose
                import matplotlib.pyplot as plt
                
                # Read and prepare data
                df = pd.read_csv(csv_path)
                
                # Find datetime and numeric columns
                datetime_cols = [col for col in df.columns 
                               if pd.api.types.is_datetime64_any_dtype(df[col])]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if not datetime_cols:
                    # Try to convert string columns to datetime
                    for col in df.select_dtypes(include=['object']):
                        try:
                            df[col] = pd.to_datetime(df[col])
                            datetime_cols.append(col)
                        except:
                            continue
                
                if not datetime_cols:
                    raise ValueError("No datetime column found")
                
                # Set datetime index
                df.set_index(datetime_cols[0], inplace=True)
                
                # Perform decomposition for each numeric column
                results = {}
                for col in numeric_cols:
                    try:
                        # Ensure series is complete
                        series = df[col].asfreq('D')
                        
                        # Interpolate missing values if needed
                        if series.isnull().any():
                            series = series.interpolate()
                        
                        # Determine period
                        if len(series) >= 365:
                            period = 365  # Annual
                        elif len(series) >= 52:
                            period = 52   # Weekly
                        elif len(series) >= 12:
                            period = 12   # Monthly
                        else:
                            period = 7    # Weekly
                        
                        # Decompose
                        decomposition = seasonal_decompose(
                            series,
                            period=period
                        )
                        
                        # Plot
                        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(12,10))
                        decomposition.observed.plot(ax=ax1)
                        ax1.set_title('Observed')
                        decomposition.trend.plot(ax=ax2)
                        ax2.set_title('Trend')
                        decomposition.seasonal.plot(ax=ax3)
                        ax3.set_title('Seasonal')
                        decomposition.resid.plot(ax=ax4)
                        ax4.set_title('Residual')
                        plt.tight_layout()
                        plt.show()
                        
                        # Calculate trend statistics
                        trend = decomposition.trend.dropna()
                        trend_change = (trend[-1] - trend[0]) / trend[0] * 100
                        
                        results[col] = {
                            'trend': decomposition.trend,
                            'seasonal': decomposition.seasonal,
                            'resid': decomposition.resid,
                            'trend_change_percent': trend_change,
                            'seasonality_strength': np.std(decomposition.seasonal) / np.std(series)
                        }
                        
                    except Exception as e:
                        results[col] = {'error': str(e)}
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'statsmodels',
                'matplotlib'
            ],
            tags=[
                'time_series',
                'decomposition',
                'trend',
                'seasonality'
            ]
        ))

        # Moving Averages and Trends
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.TIME_SERIES.value,
            subcategory="trends",
            pattern_name="Moving Averages and Trends Analysis",
            description="Analyze trends using various moving average techniques",
            required_columns={
                "datetime": {"min_count": 1},
                "numeric": {"min_count": 1}
            },
            example_prompt="Show me the trends in the data using moving averages",
            analysis_steps=[
                "Calculate simple moving averages",
                "Calculate exponential moving averages",
                "Identify trend changes",
                "Visualize trends"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Read and prepare data
                df = pd.read_csv(csv_path)
                
                # Convert date column if needed
                date_cols = [col for col in df.columns 
                           if df[col].dtype == 'datetime64[ns]' 
                           or 'date' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                    df.sort_index(inplace=True)
                
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                results = {}
                for col in numeric_cols:
                    # Calculate moving averages
                    windows = [7, 14, 30]  # Adjust based on data frequency
                    mas = {}
                    
                    for window in windows:
                        # Simple moving average
                        mas[f'SMA_{window}'] = df[col].rolling(window=window).mean()
                        
                        # Exponential moving average
                        mas[f'EMA_{window}'] = df[col].ewm(span=window).mean()
                    
                    # Plot original data with moving averages
                    plt.figure(figsize=(12, 6))
                    plt.plot(df.index, df[col], label='Original', alpha=0.5)
                    
                    for name, ma in mas.items():
                        plt.plot(df.index, ma, label=name)
                    
                    plt.title(f'Moving Averages - {col}')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                    
                    # Calculate trend changes
                    trend_changes = {}
                    for name, ma in mas.items():
                        # Calculate daily changes
                        changes = ma.diff()
                        
                        # Identify trend changes
                        trend_changes[name] = {
                            'uptrend_days': (changes > 0).sum(),
                            'downtrend_days': (changes < 0).sum(),
                            'trend_strength': abs(changes.mean() / ma.std())
                        }
                    
                    results[col] = {
                        'moving_averages': {
                            name: ma.tolist() 
                            for name, ma in mas.items()
                        },
                        'trend_changes': trend_changes
                    }
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'matplotlib',
                'seaborn'
            ],
            tags=[
                'time_series',
                'moving_average',
                'trends',
                'visualization'
            ]
        ))

        # Time Series Forecasting
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.TIME_SERIES.value,
            subcategory="forecasting",
            pattern_name="Time Series Forecasting",
            description="Forecast future values using multiple methods including SARIMA and Prophet",
            required_columns={
                "datetime": {"min_count": 1},
                "numeric": {"min_count": 1}
            },
            example_prompt="Forecast the next 30 days of values",
            analysis_steps=[
                "Prepare time series data",
                "Test for stationarity",
                "Apply multiple forecasting methods",
                "Compare forecast accuracy",
                "Generate prediction intervals"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                from statsmodels.tsa.stattools import adfuller
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                import matplotlib.pyplot as plt
                import seaborn as sns
                from datetime import datetime, timedelta

                def test_stationarity(series):
                    result = adfuller(series.dropna())
                    return {
                        'test_statistic': result[0],
                        'p_value': result[1],
                        'is_stationary': result[1] < 0.05
                    }

                # Read and prepare data
                df = pd.read_csv(csv_path)
                
                # Convert date column
                date_col = [col for col in df.columns 
                           if df[col].dtype == 'datetime64[ns]' 
                           or 'date' in col.lower()][0]
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Set date as index
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
                
                # Get numeric columns for forecasting
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                results = {}
                for col in numeric_cols:
                    try:
                        series = df[col]
                        
                        # Test stationarity
                        stationarity_test = test_stationarity(series)
                        
                        # Prepare train/test split
                        train_size = int(len(series) * 0.8)
                        train = series[:train_size]
                        test = series[train_size:]
                        
                        # Fit SARIMA model
                        model = SARIMAX(
                            train,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12)
                        )
                        fitted_model = model.fit(disp=False)
                        
                        # Make predictions
                        forecast = fitted_model.get_forecast(
                            steps=len(test) + 30,  # Test period + 30 days
                            alpha=0.05  # 95% confidence interval
                        )
                        
                        # Extract predictions and confidence intervals
                        predictions = forecast.predicted_mean
                        conf_int = forecast.conf_int()
                        
                        # Plot results
                        plt.figure(figsize=(12, 6))
                        plt.plot(train.index, train, label='Training Data')
                        plt.plot(test.index, test, label='Test Data')
                        plt.plot(predictions.index, predictions, label='Forecast')
                        plt.fill_between(
                            conf_int.index,
                            conf_int.iloc[:, 0],
                            conf_int.iloc[:, 1],
                            alpha=0.3
                        )
                        plt.title(f'Time Series Forecast - {col}')
                        plt.legend()
                        plt.show()
                        
                        # Calculate error metrics
                        test_predictions = predictions[:len(test)]
                        mae = mean_absolute_error(test, test_predictions)
                        rmse = np.sqrt(mean_squared_error(test, test_predictions))
                        
                        results[col] = {
                            'stationarity_test': stationarity_test,
                            'forecast_values': predictions.tolist(),
                            'confidence_intervals': conf_int.to_dict(),
                            'metrics': {
                                'mae': mae,
                                'rmse': rmse
                            }
                        }
                        
                    except Exception as e:
                        results[col] = {'error': str(e)}
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'statsmodels',
                'sklearn',
                'matplotlib'
            ],
            tags=[
                'time_series',
                'forecasting',
                'prediction',
                'SARIMA'
            ]
        ))

        return patterns

    def _generate_ml_patterns(self) -> List[AnalysisPattern]:
        """Generate machine learning analysis patterns"""
        patterns = []
        
        # Automated ML Pipeline
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.MACHINE_LEARNING.value,
            subcategory="automated",
            pattern_name="Automated ML Pipeline",
            description="Complete automated machine learning pipeline with preprocessing, model selection, and evaluation",
            required_columns={
                "features": {"min_count": 1},
                "target": {"min_count": 1}
            },
            example_prompt="Build a model to predict the target variable",
            analysis_steps=[
                "Analyze data types",
                "Preprocess features",
                "Handle missing values",
                "Encode categorical variables",
                "Select and train models",
                "Evaluate and compare models",
                "Generate feature importance"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.impute import SimpleImputer
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from xgboost import XGBClassifier, XGBRegressor
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    mean_squared_error, r2_score, classification_report
                )
                import matplotlib.pyplot as plt
                import seaborn as sns
                import shap

                # Read data
                df = pd.read_csv(csv_path)

                # Identify target variable
                if 'target' in df.columns:
                    target_col = 'target'
                else:
                    # Ask user or use last column
                    target_col = df.columns[-1]

                # Separate features and target
                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Identify numerical and categorical columns
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object', 'category']).columns

                # Create preprocessing pipelines
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', LabelEncoder())
                ])

                # Combine preprocessing steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])

                # Determine if classification or regression
                unique_targets = len(np.unique(y))
                is_classification = unique_targets < 10

                if is_classification:
                    # Classification models
                    models = {
                        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'xgboost': XGBClassifier(n_estimators=100, random_state=42)
                    }
                else:
                    # Regression models
                    models = {
                        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'xgboost': XGBRegressor(n_estimators=100, random_state=42)
                    }

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                results = {}
                feature_importance = {}
                shap_values = {}

                # Train and evaluate models
                for name, model in models.items():
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])

                    # Fit model
                    pipeline.fit(X_train, y_train)

                    # Make predictions
                    y_pred = pipeline.predict(X_test)

                    # Store metrics
                    if is_classification:
                        results[name] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted'),
                            'recall': recall_score(y_test, y_pred, average='weighted'),
                            'f1': f1_score(y_test, y_pred, average='weighted'),
                            'classification_report': classification_report(y_test, y_pred)
                        }
                    else:
                        results[name] = {
                            'mse': mean_squared_error(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'r2': r2_score(y_test, y_pred)
                        }

                    # Feature importance
                    feature_importance[name] = pd.DataFrame({
                        'feature': X.columns,
                        'importance': pipeline.named_steps['classifier'].feature_importances_
                    }).sort_values('importance', ascending=False)

                    # SHAP values for feature importance
                    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
                    shap_values[name] = explainer.shap_values(X_test)

                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    sns.barplot(
                        data=feature_importance[name].head(10),
                        x='importance',
                        y='feature'
                    )
                    plt.title(f'{name} - Top 10 Feature Importance')
                    plt.tight_layout()
                    plt.show()

                    # Plot SHAP summary
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values[name],
                        X_test,
                        plot_type='bar',
                        show=False
                    )
                    plt.title(f'{name} - SHAP Feature Importance')
                    plt.tight_layout()
                    plt.show()

                return {
                    'model_results': results,
                    'feature_importance': feature_importance,
                    'is_classification': is_classification,
                    'target_column': target_col
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'sklearn',
                'xgboost',
                'shap',
                'matplotlib',
                'seaborn'
            ],
            tags=[
                'machine_learning',
                'automation',
                'classification',
                'regression',
                'feature_importance'
            ]
        ))

        # Clustering Analysis
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.MACHINE_LEARNING.value,
            subcategory="clustering",
            pattern_name="Advanced Clustering Analysis",
            description="Comprehensive clustering analysis with multiple algorithms and visualization",
            required_columns={"numeric": {"min_count": 2}},
            example_prompt="Find natural clusters or segments in the data",
            analysis_steps=[
                "Preprocess and scale data",
                "Determine optimal clusters",
                "Apply multiple clustering algorithms",
                "Visualize clusters",
                "Profile cluster characteristics"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
                from sklearn.metrics import silhouette_score
                import matplotlib.pyplot as plt
                import seaborn as sns
                from scipy.cluster.hierarchy import dendrogram, linkage
                
                # Read and prepare data
                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Scale the features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols])
                scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
                
                results = {}
                
                # K-means with elbow method
                inertias = []
                silhouette_scores = []
                K = range(2, 11)
                
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_data)
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
                
                # Plot elbow curve
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(K, inertias, 'bx-')
                plt.xlabel('k')
                plt.ylabel('Inertia')
                plt.title('Elbow Method for Optimal k')
                
                plt.subplot(1, 2, 2)
                plt.plot(K, silhouette_scores, 'rx-')
                plt.xlabel('k')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score vs. k')
                plt.tight_layout()
                plt.show()
                
                # Optimal K from silhouette score
                optimal_k = K[np.argmax(silhouette_scores)]
                
                # Apply different clustering algorithms
                clustering_algorithms = {
                    'kmeans': KMeans(n_clusters=optimal_k, random_state=42),
                    'dbscan': DBSCAN(eps=0.5, min_samples=5),
                    'hierarchical': AgglomerativeClustering(n_clusters=optimal_k)
                }
                
                for name, algorithm in clustering_algorithms.items():
                    # Fit algorithm
                    clusters = algorithm.fit_predict(scaled_data)
                    
                    # Store clusters in original dataframe
                    df[f'{name}_cluster'] = clusters
                    
                    # Calculate cluster profiles
                    cluster_profiles = df.groupby(f'{name}_cluster')[numeric_cols].agg([
                        'mean', 'std', 'min', 'max'
                    ])
                    
                    # Store results
                    results[name] = {
                        'n_clusters': len(np.unique(clusters)),
                        'silhouette_score': silhouette_score(scaled_data, clusters),
                        'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
                        'cluster_profiles': cluster_profiles
                    }
                    
                    # Visualize clusters (first two features)
                    plt.figure(figsize=(10, 6))
                    scatter = plt.scatter(
                        scaled_data[:, 0],
                        scaled_data[:, 1],
                        c=clusters,
                        cmap='viridis'
                    )
                    plt.colorbar(scatter)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title(f'Clusters using {name}')
                    plt.show()
                
                # Hierarchical clustering dendrogram
                plt.figure(figsize=(12, 8))
                linkage_matrix = linkage(scaled_data, 'ward')
                dendrogram(linkage_matrix)
                plt.title('Hierarchical Clustering Dendrogram')
                plt.xlabel('Sample Index')
                plt.ylabel('Distance')
                plt.show()
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'sklearn',
                'matplotlib',
                'seaborn',
                'scipy'
            ],
            tags=[
                'clustering',
                'segmentation',
                'kmeans',
                'dbscan',
                'hierarchical'
            ]
        ))

        # Anomaly Detection
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.MACHINE_LEARNING.value,
            subcategory="anomaly_detection",
            pattern_name="Multi-method Anomaly Detection",
            description="Detect anomalies using multiple methods including statistical, isolation forest, and LOF",
            required_columns={"numeric": {"min_count": 1}},
            example_prompt="Find unusual or anomalous patterns in the data",
            analysis_steps=[
                "Prepare and scale data",
                "Apply multiple anomaly detection methods",
                "Compare results across methods",
                "Visualize anomalies",
                "Profile anomalous records"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.covariance import EllipticEnvelope
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Read and prepare data
                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Scale the features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols])
                scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
                
                results = {}
                
                # Statistical method (Z-score)
                z_scores = np.abs(scaled_data)
                statistical_outliers = (z_scores > 3).any(axis=1)
                
                # Isolation Forest
                iso_forest = IsolationForest(random_state=42, contamination=0.1)
                iso_forest_outliers = iso_forest.fit_predict(scaled_data) == -1
                
                # Local Outlier Factor
                lof = LocalOutlierFactor(contamination=0.1)
                lof_outliers = lof.fit_predict(scaled_data) == -1
                
                # Robust covariance
                robust_cov = EllipticEnvelope(contamination=0.1, random_state=42)
                robust_cov_outliers = robust_cov.fit_predict(scaled_data) == -1
                
                # Store results for each method
                methods = {
                    'statistical': statistical_outliers,
                    'isolation_forest': iso_forest_outliers,
                    'lof': lof_outliers,
                    'robust_covariance': robust_cov_outliers
                }
                
                for method_name, outliers in methods.items():
                    # Calculate anomaly statistics
                    n_outliers = outliers.sum()
                    outlier_idx = np.where(outliers)[0]
                    
                    # Profile anomalies
                    if n_outliers > 0:
                        anomaly_profile = df.iloc[outlier_idx][numeric_cols].describe()
                        normal_profile = df.iloc[~outliers][numeric_cols].describe()
                    else:
                        anomaly_profile = pd.DataFrame()
                        normal_profile = df[numeric_cols].describe()
                    
                    results[method_name] = {
                        'n_outliers': n_outliers,
                        'outlier_percentage': (n_outliers / len(df)) * 100,
                        'anomaly_profile': anomaly_profile,
                        'normal_profile': normal_profile
                    }
                    
                    # Visualize results for first two features
                    plt.figure(figsize=(10, 6))
                    plt.scatter(
                        scaled_data[~outliers, 0],
                        scaled_data[~outliers, 1],
                        c='blue',
                        label='Normal'
                    )
                    plt.scatter(
                        scaled_data[outliers, 0],
                        scaled_data[outliers, 1],
                        c='red',
                        label='Anomaly'
                    )
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title(f'Anomalies detected by {method_name}')
                    plt.legend()
                    plt.show()
                
                # Compare methods
                method_comparison = pd.DataFrame({
                    method: outliers.astype(int)
                    for method, outliers in methods.items()
                })
                
                # Correlation between methods
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    method_comparison.corr(),
                    annot=True,
                    cmap='coolwarm',
                    center=0
                )
                plt.title('Correlation between anomaly detection methods')
                plt.show()
                
                # Consensus anomalies (detected by majority of methods)
                consensus_threshold = len(methods) / 2
                consensus_anomalies = method_comparison.sum(axis=1) >= consensus_threshold
                
                results['consensus'] = {
                    'n_outliers': consensus_anomalies.sum(),
                    'outlier_percentage': (consensus_anomalies.sum() / len(df)) * 100,
                    'anomaly_indices': np.where(consensus_anomalies)[0].tolist()
                }
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'sklearn',
                'matplotlib',
                'seaborn'
            ],
            tags=[
                'anomaly_detection',
                'outliers',
                'isolation_forest',
                'lof'
            ]
        ))

        return patterns

    def _generate_data_quality_patterns(self) -> List[AnalysisPattern]:
        """Generate data quality analysis patterns"""
        patterns = []

        # Comprehensive Data Quality Assessment
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.DATA_QUALITY.value,
            subcategory="assessment",
            pattern_name="Comprehensive Data Quality Assessment",
            description="Perform thorough data quality analysis including missing values, duplicates, outliers, and pattern validation",
            required_columns={"any": {"min_count": 1}},
            example_prompt="Analyze the quality of the dataset",
            analysis_steps=[
                "Check data completeness",
                "Identify duplicates",
                "Detect outliers",
                "Validate patterns",
                "Assess consistency"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                import re
                from datetime import datetime

                def validate_email(email):
                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                return bool(re.match(pattern, str(email)))

                def validate_date(date_str):
                    try:
                        pd.to_datetime(date_str)
                        return True
                    except:
                        return False

                # Read data
                df = pd.read_csv(csv_path)

                results = {
                    'basic_info': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_usage': df.memory_usage().sum() / 1024**2  # MB
                    },
                    'column_analysis': {},
                    'duplicates': {},
                    'pattern_validation': {},
                    'consistency_checks': {}
                }

                # 1. Column-level Analysis
                for column in df.columns:
                    col_data = df[column]
                    col_analysis = {
                        'dtype': str(col_data.dtype),
                        'unique_values': col_data.nunique(),
                        'missing_count': col_data.isnull().sum(),
                        'missing_percentage': (col_data.isnull().sum() / len(df)) * 100
                    }
                    
                    # Numeric column specific analysis
                    if pd.api.types.is_numeric_dtype(col_data):
                        clean_data = col_data.dropna()
                        if len(clean_data) > 0:
                            z_scores = np.abs(stats.zscore(clean_data))
                            col_analysis.update({
                                'mean': col_data.mean(),
                                'std': col_data.std(),
                                'min': col_data.min(),
                                'max': col_data.max(),
                                'outliers_count': (z_scores > 3).sum()
                            })
                    
                    # String column specific analysis
                    elif pd.api.types.is_string_dtype(col_data):
                        clean_data = col_data.dropna()
                        if len(clean_data) > 0:
                            col_analysis.update({
                                'empty_strings': (clean_data == '').sum(),
                                'avg_length': clean_data.str.len().mean(),
                                'max_length': clean_data.str.len().max(),
                                'min_length': clean_data.str.len().min()
                            })
                            
                            # Pattern validation for specific types
                            if 'email' in column.lower():
                                valid_emails = clean_data.apply(validate_email)
                                col_analysis['valid_email_percentage'] = (valid_emails.sum() / len(clean_data)) * 100
                            
                            elif 'date' in column.lower():
                                valid_dates = clean_data.apply(validate_date)
                                col_analysis['valid_date_percentage'] = (valid_dates.sum() / len(clean_data)) * 100
                    
                    results['column_analysis'][column] = col_analysis

                # 2. Missing Value Analysis
                plt.figure(figsize=(12, 6))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
                plt.title('Missing Value Patterns')
                plt.tight_layout()
                plt.show()

                # 3. Missing Value Correlations
                missing_correlations = df.isnull().corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    missing_correlations,
                    annot=True,
                    cmap='coolwarm',
                    center=0
                )
                plt.title('Missing Value Correlations')
                plt.show()

                # 4. Duplicate Analysis
                duplicates = df.duplicated()
                results['duplicates'].update({
                    'total_duplicates': duplicates.sum(),
                    'duplicate_percentage': (duplicates.sum() / len(df)) * 100,
                    'columns_with_duplicates': {
                        col: df[col].duplicated().sum()
                        for col in df.columns
                    }
                })

                # 5. Consistency Checks
                consistency_checks = {
                    'data_types': {
                        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                        'text_columns': len(df.select_dtypes(include=['object']).columns),
                        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                        'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
                    },
                    'value_ranges': {}
                }

                # Check value ranges for numeric columns
                for col in df.select_dtypes(include=[np.number]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    consistency_checks['value_ranges'][col] = {
                        'expected_range': [lower_bound, upper_bound],
                        'actual_range': [df[col].min(), df[col].max()],
                        'out_of_range_count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                    }

                # 6. Generate Summary Visualizations
                # Distribution of missing values
                plt.figure(figsize=(12, 6))
                missing_percentages = df.isnull().sum() / len(df) * 100
                missing_percentages.sort_values(ascending=False).plot(kind='bar')
                plt.title('Percentage of Missing Values by Column')
                plt.xlabel('Columns')
                plt.ylabel('Missing Percentage')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                # 7. Data Type Consistency
                type_consistency = {}
                for col in df.columns:
                    # Check if mixed types exist in the column
                    types = df[col].apply(type).value_counts()
                    type_consistency[col] = {
                        'types_found': types.index.map(str).tolist(),
                        'type_counts': types.tolist(),
                        'is_consistent': len(types) == 1
                    }

                results['consistency_checks']['type_consistency'] = type_consistency

                # 8. Pattern Validation
                pattern_validation = {}
                
                # Email validation for relevant columns
                email_columns = [col for col in df.columns if 'email' in col.lower()]
                for col in email_columns:
                    valid_emails = df[col].dropna().apply(validate_email)
                    pattern_validation[f'{col}_email_pattern'] = {
                        'valid_count': valid_emails.sum(),
                        'invalid_count': (~valid_emails).sum(),
                        'validity_percentage': (valid_emails.sum() / len(valid_emails)) * 100
                    }
                
                # Date validation for relevant columns
                date_columns = [col for col in df.columns if 'date' in col.lower()]
                for col in date_columns:
                    valid_dates = df[col].dropna().apply(validate_date)
                    pattern_validation[f'{col}_date_pattern'] = {
                        'valid_count': valid_dates.sum(),
                        'invalid_count': (~valid_dates).sum(),
                        'validity_percentage': (valid_dates.sum() / len(valid_dates)) * 100
                    }

                results['pattern_validation'] = pattern_validation

                # 9. Generate Quality Score
                def calculate_column_quality_score(col_analysis):
                    scores = []
                    
                    # Completeness score
                    missing_pct = col_analysis['missing_percentage']
                    completeness_score = 1 - (missing_pct / 100)
                    scores.append(completeness_score)
                    
                    # Consistency score (if applicable)
                    if 'is_consistent' in col_analysis:
                        scores.append(1.0 if col_analysis['is_consistent'] else 0.5)
                    
                    # Validity score (if applicable)
                    if 'valid_email_percentage' in col_analysis:
                        scores.append(col_analysis['valid_email_percentage'] / 100)
                    elif 'valid_date_percentage' in col_analysis:
                        scores.append(col_analysis['valid_date_percentage'] / 100)
                    
                    return np.mean(scores) * 100

                quality_scores = {
                    col: calculate_column_quality_score(analysis)
                    for col, analysis in results['column_analysis'].items()
                }
                
                results['quality_scores'] = {
                    'column_scores': quality_scores,
                    'overall_score': np.mean(list(quality_scores.values()))
                }

                # 10. Generate Recommendations
                recommendations = []
                
                # Missing value recommendations
                high_missing_cols = [
                    col for col in df.columns 
                    if (df[col].isnull().sum() / len(df)) > 0.2
                ]
                if high_missing_cols:
                    recommendations.append({
                        'type': 'missing_values',
                        'description': 'High missing value rate',
                        'columns': high_missing_cols,
                        'suggestion': 'Consider imputation or column removal'
                    })

                # Duplicate recommendations
                if results['duplicates']['duplicate_percentage'] > 5:
                    recommendations.append({
                        'type': 'duplicates',
                        'description': 'High percentage of duplicate rows',
                        'suggestion': 'Investigate source of duplicates and consider deduplication'
                    })

                # Type consistency recommendations
                inconsistent_types = [
                    col for col, check in type_consistency.items() 
                    if not check['is_consistent']
                ]
                if inconsistent_types:
                    recommendations.append({
                        'type': 'type_consistency',
                        'description': 'Mixed data types detected',
                        'columns': inconsistent_types,
                        'suggestion': 'Standardize data types'
                    })

                # Pattern validation recommendations
                for col, validation in pattern_validation.items():
                    if validation['validity_percentage'] < 90:
                        recommendations.append({
                            'type': 'pattern_validation',
                            'description': f'Low pattern validity in {col}',
                            'validity_rate': validation['validity_percentage'],
                            'suggestion': 'Review and clean invalid values'
                        })

                # Value range recommendations
                for col, range_check in consistency_checks['value_ranges'].items():
                    if range_check['out_of_range_count'] > len(df) * 0.05:
                        recommendations.append({
                            'type': 'value_ranges',
                            'description': f'High number of outliers in {col}',
                            'outlier_count': range_check['out_of_range_count'],
                            'suggestion': 'Investigate and validate extreme values'
                        })

                results['recommendations'] = recommendations

                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'seaborn'
            ],
            tags=[
                'data_quality',
                'validation',
                'cleaning',
                'assessment'
            ]
        ))

        # Data Profiling Pattern
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.DATA_QUALITY.value,
            subcategory="profiling",
            pattern_name="Advanced Data Profiling",
            description="Generate comprehensive data profile including distributions, patterns, and relationships",
            required_columns={"any": {"min_count": 1}},
            example_prompt="Create a detailed profile of my dataset",
            analysis_steps=[
                "Analyze column distributions",
                "Identify patterns and correlations",
                "Generate summary statistics",
                "Create profile visualizations"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                from collections import Counter
                import re

                # Read data
                df = pd.read_csv(csv_path)
                
                results = {
                    'overview': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_usage': df.memory_usage().sum() / 1024**2,  # MB
                        'dtypes': df.dtypes.value_counts().to_dict()
                    },
                    'column_profiles': {},
                    'correlations': {},
                    'patterns': {}
                }

                # Analyze each column
                for column in df.columns:
                    col_data = df[column]
                    profile = {
                        'dtype': str(col_data.dtype),
                        'unique_count': col_data.nunique(),
                        'missing_count': col_data.isnull().sum(),
                        'missing_percentage': (col_data.isnull().sum() / len(df)) * 100
                    }
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        # Numeric column analysis
                        clean_data = col_data.dropna()
                        if len(clean_data) > 0:
                            profile.update({
                                'mean': col_data.mean(),
                                'median': col_data.median(),
                                'std': col_data.std(),
                                'min': col_data.min(),
                                'max': col_data.max(),
                                'skew': stats.skew(clean_data),
                                'kurtosis': stats.kurtosis(clean_data)
                            })
                            
                            # Distribution plot
                            plt.figure(figsize=(10, 6))
                            sns.histplot(clean_data, kde=True)
                            plt.title(f'Distribution of {column}')
                            plt.show()
                            
                            # Box plot
                            plt.figure(figsize=(8, 4))
                            sns.boxplot(x=clean_data)
                            plt.title(f'Box Plot of {column}')
                            plt.show()
                            
                            # Check for outliers
                            z_scores = np.abs(stats.zscore(clean_data))
                            profile['outliers_count'] = (z_scores > 3).sum()
                            
                            # Test for normality
                            _, normality_pval = stats.normaltest(clean_data)
                            profile['is_normal'] = normality_pval > 0.05
                    
                    elif pd.api.types.is_string_dtype(col_data):
                        # String column analysis
                        clean_data = col_data.dropna()
                        if len(clean_data) > 0:
                            # Length statistics
                            lengths = clean_data.str.len()
                            profile.update({
                                'avg_length': lengths.mean(),
                                'max_length': lengths.max(),
                                'min_length': lengths.min(),
                                'empty_strings': (clean_data == '').sum()
                            })
                            
                            # Most common values
                            value_counts = col_data.value_counts()
                            profile['top_values'] = value_counts.head(10).to_dict()
                            
                            # Character type analysis
                            char_types = {
                                'alphabetic': clean_data.str.contains(r'[A-Za-z]').sum(),
                                'numeric': clean_data.str.contains(r'\d').sum(),
                                'special': clean_data.str.contains(r'[^A-Za-z0-9\s]').sum()
                            }
                            profile['character_types'] = char_types
                            
                            # Value length distribution
                            plt.figure(figsize=(10, 6))
                            sns.histplot(lengths, bins=30)
                            plt.title(f'String Length Distribution for {column}')
                            plt.show()
                            
                            # Bar plot of top values
                            plt.figure(figsize=(12, 6))
                            value_counts.head(10).plot(kind='bar')
                            plt.title(f'Top 10 Values in {column}')
                            plt.xticks(rotation=45)
                            plt.show()
                            
                            # Pattern detection
                            common_patterns = []
                            for value in clean_data.unique():
                                pattern = re.sub(r'\d', '9', str(value))
                                pattern = re.sub(r'[A-Za-z]', 'x', pattern)
                                common_patterns.append(pattern)
                            
                            pattern_counts = Counter(common_patterns)
                            profile['common_patterns'] = dict(pattern_counts.most_common(5))
                    
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        # Datetime column analysis
                        clean_data = col_data.dropna()
                        if len(clean_data) > 0:
                            profile.update({
                                'min_date': clean_data.min(),
                                'max_date': clean_data.max(),
                                'date_range_days': (clean_data.max() - clean_data.min()).days
                            })
                            
                            # Temporal patterns
                            profile['temporal_patterns'] = {
                                'weekday_distribution': clean_data.dt.dayofweek.value_counts().to_dict(),
                                'month_distribution': clean_data.dt.month.value_counts().to_dict(),
                                'year_distribution': clean_data.dt.year.value_counts().to_dict()
                            }
                            
                            # Time series plot
                            plt.figure(figsize=(12, 6))
                            clean_data.value_counts().sort_index().plot()
                            plt.title(f'Time Series Distribution of {column}')
                            plt.xticks(rotation=45)
                            plt.show()
                    
                    results['column_profiles'][column] = profile

                # Correlation analysis for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlation_matrix = df[numeric_cols].corr()
                    
                    # Plot correlation heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        correlation_matrix,
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        fmt='.2f'
                    )
                    plt.title('Correlation Matrix')
                    plt.show()
                    
                    results['correlations'] = correlation_matrix.to_dict()

                # Pattern analysis
                for col1 in df.columns:
                    for col2 in df.columns:
                        if col1 < col2:  # Avoid duplicate comparisons
                            # Check for functional dependencies
                            grouped = df.groupby(col1)[col2].nunique()
                            if (grouped == 1).all():
                                results['patterns'][f'{col1}_determines_{col2}'] = True
                            
                            # Check for inclusion dependencies
                            if df[col1].dtype == df[col2].dtype:
                                set1 = set(df[col1].dropna())
                                set2 = set(df[col2].dropna())
                                if set1.issubset(set2):
                                    results['patterns'][f'{col1}_subset_of_{col2}'] = True
                                elif set2.issubset(set1):
                                    results['patterns'][f'{col2}_subset_of_{col1}'] = True

                # Generate insights
                insights = []
                
                # Missing value insights
                for col, profile in results['column_profiles'].items():
                    if profile['missing_percentage'] > 0:
                        insights.append({
                            'type': 'missing_values',
                            'column': col,
                            'description': (f"Column has {profile['missing_percentage']:.1f}% "
                                          f"missing values ({profile['missing_count']} rows)")
                        })

                # Distribution insights
                for col, profile in results['column_profiles'].items():
                    if 'is_normal' in profile:
                        insights.append({
                            'type': 'distribution',
                            'column': col,
                            'description': (f"Distribution is {'normal' if profile['is_normal'] else 'non-normal'} "
                                          f"with skewness {profile.get('skew', 0):.2f}")
                        })

                # Correlation insights
                if 'correlations' in results:
                    for col1 in correlation_matrix.columns:
                        for col2 in correlation_matrix.columns:
                            if col1 < col2:
                                corr = correlation_matrix.loc[col1, col2]
                                if abs(corr) > 0.7:
                                    insights.append({
                                        'type': 'correlation',
                                        'columns': [col1, col2],
                                        'description': f"Strong correlation detected ({corr:.2f})"
                                    })

                # Pattern insights
                for pattern, _ in results['patterns'].items():
                    insights.append({
                        'type': 'pattern',
                        'description': f"Detected pattern: {pattern.replace('_', ' ')}"
                    })

                results['insights'] = insights

                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'seaborn',
                'collections',
                're'
            ],
            tags=[
                'data_profiling',
                'pattern_detection',
                'visualization',
                'statistics'
            ]
        ))

        # Data Quality Monitoring Pattern
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.DATA_QUALITY.value,
            subcategory="monitoring",
            pattern_name="Data Quality Monitoring",
            description="Monitor data quality metrics over time and detect anomalies",
            required_columns={
                "datetime": {"min_count": 1},
                "metrics": {"min_count": 1}
            },
            example_prompt="Monitor the quality of my data over time",
            analysis_steps=[
                "Calculate quality metrics",
                "Track temporal changes",
                "Detect anomalies",
                "Generate alerts"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import matplotlib.pyplot as plt
                import seaborn as sns
                from datetime import datetime, timedelta

                # Read data
                df = pd.read_csv(csv_path)
                
                # Convert date column
                date_cols = [col for col in df.columns 
                           if df[col].dtype == 'datetime64[ns]' 
                           or 'date' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                
                results = {
                    'metrics_over_time': {},
                    'anomalies': {},
                    'trends': {},
                    'alerts': []
                }

                # Calculate daily quality metrics
                daily_metrics = pd.DataFrame()
                
                # Missing value rates
                daily_metrics['missing_rate'] = df.resample('D').apply(
                    lambda x: x.isnull().mean().mean()
                )

                # Duplicate rates
                daily_metrics['duplicate_rate'] = df.resample('D').apply(
                    lambda x: x.duplicated().mean()
                )

                # Data type consistency
                def type_consistency(x):
                    return np.mean([
                        len(x[col].apply(type).unique()) == 1
                        for col in x.columns
                    ])
                
                daily_metrics['type_consistency'] = df.resample('D').apply(type_consistency)

                # Value range violations
                def range_violations(x):
                    violations = 0
                    numeric_cols = x.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        q1 = x[col].quantile(0.25)
                        q3 = x[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        violations += ((x[col] < lower) | (x[col] > upper)).mean()
                    return violations / len(numeric_cols) if numeric_cols.size > 0 else 0

                daily_metrics['range_violations'] = df.resample('D').apply(range_violations)

                # Calculate overall quality score
                daily_metrics['quality_score'] = 100 * (
                    1 - daily_metrics['missing_rate']
                ) * (
                    1 - daily_metrics['duplicate_rate']
                ) * daily_metrics['type_consistency'] * (
                    1 - daily_metrics['range_violations']
                )

                # Plot metrics over time
                fig, axes = plt.subplots(5, 1, figsize=(12, 15))
                fig.suptitle('Data Quality Metrics Over Time')
                
                metrics = ['missing_rate', 'duplicate_rate', 'type_consistency',
                          'range_violations', 'quality_score']
                
                for ax, metric in zip(axes, metrics):
                    daily_metrics[metric].plot(ax=ax)
                    ax.set_title(metric.replace('_', ' ').title())
                    ax.grid(True)
                
                plt.tight_layout()
                plt.show()

                # Detect anomalies in metrics
                for metric in metrics:
                    # Calculate rolling statistics
                    rolling_mean = daily_metrics[metric].rolling(window=7).mean()
                    rolling_std = daily_metrics[metric].rolling(window=7).std()
                    
                    # Define anomaly thresholds
                    upper_bound = rolling_mean + 3 * rolling_std
                    lower_bound = rolling_mean - 3 * rolling_std
                    
                    # Detect anomalies
                    anomalies = daily_metrics[
                        (daily_metrics[metric] > upper_bound) |
                        (daily_metrics[metric] < lower_bound)
                    ]
                    
                    if not anomalies.empty:
                        results['anomalies'][metric] = {
                            'dates': anomalies.index.tolist(),
                            'values': anomalies[metric].tolist(),
                            'expected_range': {
                                'lower': lower_bound[anomalies.index].tolist(),
                                'upper': upper_bound[anomalies.index].tolist()
                            }
                        }
                        
                        # Generate alerts for anomalies
                        for date, value in zip(anomalies.index, anomalies[metric]):
                            results['alerts'].append({
                                'timestamp': date,
                                'metric': metric,
                                'value': value,
                                'expected_range': [
                                    lower_bound[date],
                                    upper_bound[date]
                                ],
                                'severity': 'high' if abs(
                                    (value - rolling_mean[date]) / rolling_std[date]
                                ) > 5 else 'medium'
                            })

                # Analyze trends
                for metric in metrics:
                    # Calculate trend using linear regression
                    x = np.arange(len(daily_metrics))
                    y = daily_metrics[metric].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    results['trends'][metric] = {
                        'slope': slope,
                        'p_value': p_value,
                        'r_squared': r_value ** 2,
                        'trend_direction': 'improving' if (
                            (slope > 0 and metric in ['type_consistency', 'quality_score']) or
                            (slope < 0 and metric in ['missing_rate', 'duplicate_rate', 'range_violations'])
                        ) else 'deteriorating'
                    }

                # Store metrics
                results['metrics_over_time'] = daily_metrics.to_dict(orient='index')

                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'matplotlib',
                'seaborn'
            ],
            tags=[
                'monitoring',
                'time_series',
                'anomaly_detection',
                'quality_metrics'
            ]
        ))

        return patterns

    def _generate_business_patterns(self) -> List[AnalysisPattern]:
        """Generate business analytics patterns"""
        patterns = []
        
        # Revenue Analysis Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="revenue",
            pattern_name="Comprehensive Revenue Analysis",
            description="Complete revenue analysis including trends, segmentation, and customer metrics",
            required_columns={
                "revenue": {"min_count": 1},
                "date": {"min_count": 1},
                "customer_id": {"optional": True},
                "segment": {"optional": True}
            },
            example_prompt="Analyze our revenue trends and patterns",
            analysis_steps=[
                "Calculate revenue metrics",
                "Analyze revenue trends",
                "Segment revenue analysis",
                "Customer revenue metrics",
                "Generate revenue forecasts"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from scipy import stats
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Read and prepare data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                revenue_col = [col for col in df.columns if 'revenue' in col.lower()][0]
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                customer_col = [col for col in df.columns if 'customer' in col.lower() or 'client' in col.lower()]
                segment_col = [col for col in df.columns if 'segment' in col.lower() or 'category' in col.lower()]
                
                # Convert date
                df[date_col] = pd.to_datetime(df[date_col])
                
                results = {
                    'metrics': {},
                    'trends': {},
                    'segments': {},
                    'customer_metrics': {},
                    'forecasts': {}
                }
                
                # 1. Revenue Metrics
                # Basic metrics
                total_revenue = df[revenue_col].sum()
                avg_revenue = df[revenue_col].mean()
                median_revenue = df[revenue_col].median()
                
                results['metrics'].update({
                    'total_revenue': total_revenue,
                    'average_revenue': avg_revenue,
                    'median_revenue': median_revenue,
                    'revenue_std': df[revenue_col].std(),
                    'revenue_growth': (df[revenue_col].iloc[-1] - df[revenue_col].iloc[0]) / df[revenue_col].iloc[0] * 100
                })
                
                # 2. Time-based Analysis
                df['year'] = df[date_col].dt.year
                df['month'] = df[date_col].dt.month
                df['quarter'] = df[date_col].dt.quarter
                
                # Monthly trends
                monthly_revenue = df.groupby(df[date_col].dt.to_period('M'))[revenue_col].agg(['sum', 'count', 'mean'])
                monthly_growth = monthly_revenue['sum'].pct_change() * 100
                
                # Quarterly trends
                quarterly_revenue = df.groupby(df[date_col].dt.to_period('Q'))[revenue_col].agg(['sum', 'count', 'mean'])
                quarterly_growth = quarterly_revenue['sum'].pct_change() * 100
                
                results['trends'].update({
                    'monthly_revenue': monthly_revenue.to_dict(),
                    'monthly_growth': monthly_growth.to_dict(),
                    'quarterly_revenue': quarterly_revenue.to_dict(),
                    'quarterly_growth': quarterly_growth.to_dict()
                })
                
                # Visualize trends
                fig_trends = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Monthly Revenue', 'Monthly Growth', 
                                'Quarterly Revenue', 'Quarterly Growth')
                )
                
                # Monthly revenue trend
                fig_trends.add_trace(
                    go.Scatter(
                        x=monthly_revenue.index.astype(str),
                        y=monthly_revenue['sum'],
                        name='Monthly Revenue'
                    ),
                    row=1, col=1
                )
                
                # Monthly growth
                fig_trends.add_trace(
                    go.Bar(
                        x=monthly_growth.index.astype(str),
                        y=monthly_growth,
                        name='Monthly Growth %'
                    ),
                    row=1, col=2
                )
                
                # Quarterly revenue
                fig_trends.add_trace(
                    go.Scatter(
                        x=quarterly_revenue.index.astype(str),
                        y=quarterly_revenue['sum'],
                        name='Quarterly Revenue'
                    ),
                    row=2, col=1
                )
                
                # Quarterly growth
                fig_trends.add_trace(
                    go.Bar(
                        x=quarterly_growth.index.astype(str),
                        y=quarterly_growth,
                        name='Quarterly Growth %'
                    ),
                    row=2, col=2
                )
                
                fig_trends.update_layout(height=800, title_text="Revenue Trends Analysis")
                
                # 3. Segment Analysis (if segment column exists)
                if segment_col:
                    segment_col = segment_col[0]
                    segment_analysis = df.groupby(segment_col).agg({
                        revenue_col: ['sum', 'mean', 'count'],
                        'customer_id': 'nunique' if customer_col else 'count'
                    }).round(2)
                    
                    # Calculate segment percentages
                    segment_analysis['revenue_percentage'] = (
                        segment_analysis[(revenue_col, 'sum')] / total_revenue * 100
                    ).round(2)
                    
                    results['segments'] = segment_analysis.to_dict()
                    
                    # Visualize segment distribution
                    fig_segments = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Revenue by Segment', 'Customer Count by Segment')
                    )
                    
                    fig_segments.add_trace(
                        go.Pie(
                            labels=segment_analysis.index,
                            values=segment_analysis[(revenue_col, 'sum')],
                            name='Revenue'
                        ),
                        row=1, col=1
                    )
                    
                    fig_segments.add_trace(
                        go.Bar(
                            x=segment_analysis.index,
                            y=segment_analysis[('customer_id', 'nunique' if customer_col else 'count')],
                            name='Customers'
                        ),
                        row=1, col=2
                    )
                    
                    fig_segments.update_layout(title_text="Segment Analysis")
                
                # 4. Customer Analysis (if customer_id exists)
                if customer_col:
                    customer_col = customer_col[0]
                    customer_analysis = df.groupby(customer_col)[revenue_col].agg([
                        'sum', 'count', 'mean', 'median', 'std'
                    ]).round(2)
                    
                    # Calculate customer metrics
                    results['customer_metrics'].update({
                        'total_customers': len(customer_analysis),
                        'avg_revenue_per_customer': customer_analysis['sum'].mean(),
                        'median_revenue_per_customer': customer_analysis['sum'].median(),
                        'top_10_customers': customer_analysis.nlargest(10, 'sum').to_dict()
                    })
                    
                    # Customer concentration analysis
                    customer_concentration = customer_analysis['sum'].sort_values(ascending=False)
                    customer_concentration_pct = (
                        customer_concentration.cumsum() / customer_concentration.sum() * 100
                    )
                    
                    # Plot customer concentration
                    fig_concentration = go.Figure()
                    fig_concentration.add_trace(
                        go.Scatter(
                            x=np.arange(len(customer_concentration_pct)) / len(customer_concentration_pct) * 100,
                            y=customer_concentration_pct,
                            name='Customer Concentration'
                        )
                    )
                    fig_concentration.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=100, y1=100,
                        line=dict(color='red', dash='dash'),
                        name='Equal Distribution'
                    )
                    fig_concentration.update_layout(
                        title='Customer Revenue Concentration',
                        xaxis_title='Cumulative % of Customers',
                        yaxis_title='Cumulative % of Revenue'
                    )
                    
                # 5. Forecasting
                # Simple moving average forecast
                ma_periods = [3, 6, 12]
                forecasts = {}
                
                for period in ma_periods:
                    ma = monthly_revenue['sum'].rolling(window=period).mean()
                    forecast = ma.iloc[-1]  # Use last MA value as forecast
                    forecasts[f'{period}_month_ma'] = forecast
                
                # Seasonal decomposition for pattern analysis
                if len(monthly_revenue) >= 12:
                    decomposition = seasonal_decompose(
                        monthly_revenue['sum'],
                        period=12,
                        extrapolate_trend='freq'
                    )
                    
                    # Plot decomposition
                    fig_decomp = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual']
                    )
                    
                    components = [
                        decomposition.observed,
                        decomposition.trend,
                        decomposition.seasonal,
                        decomposition.resid
                    ]
                    
                    for i, component in enumerate(components, 1):
                        fig_decomp.add_trace(
                            go.Scatter(
                                x=component.index.astype(str),
                                y=component,
                                name=['Observed', 'Trend', 'Seasonal', 'Residual'][i-1]
                            ),
                            row=i, col=1
                        )
                    
                    fig_decomp.update_layout(height=1000, title_text="Revenue Decomposition")
                
                results['forecasts'].update({
                    'moving_averages': forecasts,
                    'seasonality_strength': np.std(decomposition.seasonal) / np.std(monthly_revenue['sum'])
                    if len(monthly_revenue) >= 12 else None
                })
                
                return {
                    'results': results,
                    'visualizations': {
                        'trends': fig_trends,
                        'segments': fig_segments if segment_col else None,
                        'concentration': fig_concentration if customer_col else None,
                        'decomposition': fig_decomp if len(monthly_revenue) >= 12 else None
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'scipy',
                'plotly',
                'statsmodels'
            ],
            tags=[
                'revenue',
                'business_metrics',
                'trends',
                'forecasting',
                'segmentation',
                'customer_analysis'
            ]
        ))

        # Customer Segmentation Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="segmentation",
            pattern_name="Customer Segmentation Analysis",
            description="Segment customers using multiple methods including RFM, k-means, and behavioral clustering",
            required_columns={
                "customer_id": {"min_count": 1},
                "transaction_date": {"min_count": 1},
                "amount": {"min_count": 1},
                "product_id": {"optional": True},
                "behavioral_metrics": {"optional": True}
            },
            example_prompt="Segment our customers based on their behavior and value",
            analysis_steps=[
                "Calculate RFM metrics",
                "Perform behavioral analysis",
                "Apply clustering algorithms",
                "Profile segments",
                "Generate segment insights"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from datetime import datetime
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                customer_col = [col for col in df.columns if 'customer' in col.lower()][0]
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                amount_col = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower()][0]
                
                # Convert date
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Calculate RFM metrics
                today = df[date_col].max()
                
                rfm = df.groupby(customer_col).agg({
                    date_col: lambda x: (today - x.max()).days,  # Recency
                    'order_id': 'count',  # Frequency
                    amount_col: 'sum'  # Monetary
                }).reset_index()
                
                rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
                
                # Scale RFM metrics
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
                
                # Perform k-means clustering
                k = 5  # Number of segments
                kmeans = KMeans(n_clusters=k, random_state=42)
                rfm['segment'] = kmeans.fit_predict(rfm_scaled)
                
                # Profile segments
                segment_profiles = rfm.groupby('segment').agg({
                    'recency': ['mean', 'min', 'max'],
                    'frequency': ['mean', 'min', 'max'],
                    'monetary': ['mean', 'min', 'max'],
                    customer_col: 'count'
                }).round(2)
                
                # Calculate segment metrics
                segment_metrics = pd.DataFrame()
                segment_metrics['size'] = rfm.groupby('segment').size()
                segment_metrics['size_percent'] = segment_metrics['size'] / len(rfm) * 100
                segment_metrics['avg_recency'] = rfm.groupby('segment')['recency'].mean()
                segment_metrics['avg_frequency'] = rfm.groupby('segment')['frequency'].mean()
                segment_metrics['avg_monetary'] = rfm.groupby('segment')['monetary'].mean()
                
                # Assign segment labels
                def assign_segment_label(row):
                    if row['avg_monetary'] > rfm['monetary'].quantile(0.75):
                        return 'High Value'
                    elif row['avg_frequency'] > rfm['frequency'].quantile(0.75):
                        return 'Loyal'
                    elif row['avg_recency'] < rfm['recency'].quantile(0.25):
                        return 'Recent'
                    elif row['avg_recency'] > rfm['recency'].quantile(0.75):
                        return 'At Risk'
                    else:
                        return 'Average'
                
                segment_metrics['label'] = segment_metrics.apply(assign_segment_label, axis=1)
                
                # Visualizations
                # 1. Segment Size
                fig_size = px.pie(
                    segment_metrics,
                    values='size',
                    names='label',
                    title='Customer Segments Distribution'
                )
                
                # 2. Segment Profiles
                fig_profiles = go.Figure()
                
                metrics = ['avg_recency', 'avg_frequency', 'avg_monetary']
                for label in segment_metrics['label'].unique():
                    segment_data = segment_metrics[segment_metrics['label'] == label]
                    fig_profiles.add_trace(go.Scatterpolar(
                        r=[segment_data[m].values[0] for m in metrics],
                        theta=metrics,
                        name=label
                    ))
                
                fig_profiles.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title='Segment Profiles'
                )
                
                # 3. Segment Comparison
                fig_comparison = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=['Recency', 'Frequency', 'Monetary']
                )
                
                for i, metric in enumerate(['avg_recency', 'avg_frequency', 'avg_monetary'], 1):
                    fig_comparison.add_trace(
                        go.Bar(
                            x=segment_metrics['label'],
                            y=segment_metrics[metric],
                            name=metric
                        ),
                        row=1, col=i
                    )
                
                fig_comparison.update_layout(height=400, title_text="Segment Comparison")
                
                # Additional behavioral analysis if product data exists
                product_metrics = None
                if 'product_id' in df.columns:
                    product_metrics = df.groupby([customer_col, 'product_id']).size().unstack(fill_value=0)
                    product_metrics_scaled = StandardScaler().fit_transform(product_metrics)
                    
                    # Product-based clustering
                    product_clusters = KMeans(n_clusters=4, random_state=42).fit_predict(product_metrics_scaled)
                    
                    # Combine with RFM segments
                    combined_segments = pd.DataFrame({
                        'rfm_segment': rfm['segment'],
                        'product_segment': product_clusters
                    })
                    
                    # Create segment matrix
                    segment_matrix = pd.crosstab(
                        combined_segments['rfm_segment'],
                        combined_segments['product_segment']
                    )
                
                return {
                    'rfm_metrics': rfm.to_dict(),
                    'segment_profiles': segment_profiles.to_dict(),
                    'segment_metrics': segment_metrics.to_dict(),
                    'product_segments': product_metrics.to_dict() if product_metrics is not None else None,
                    'visualizations': {
                        'segment_size': fig_size,
                        'segment_profiles': fig_profiles,
                        'segment_comparison': fig_comparison
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'sklearn',
                'plotly',
                'datetime'
            ],
            tags=[
                'segmentation',
                'clustering',
                'rfm_analysis',
                'customer_analytics',
                'behavioral_analysis'
            ]
        ))

        # Cohort Analysis Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="cohort",
            pattern_name="Cohort Retention Analysis",
            description="Analyze user retention and behavior across different cohorts over time",
            required_columns={
                "user_id": {"min_count": 1},
                "date": {"min_count": 1},
                "event_type": {"optional": True},
                "revenue": {"optional": True}
            },
            example_prompt="Show me retention rates for different user cohorts",
            analysis_steps=[
                "Create cohort groups",
                "Calculate retention metrics",
                "Generate cohort matrix",
                "Analyze revenue by cohort",
                "Visualize retention trends"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from datetime import datetime
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                user_col = [col for col in df.columns if 'user' in col.lower()][0]
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Create cohort data
                df['cohort_date'] = df.groupby(user_col)[date_col].transform('min')
                df['cohort_month'] = df['cohort_date'].dt.to_period('M')
                df['activity_month'] = df[date_col].dt.to_period('M')
                df['months_since_join'] = (
                    df['activity_month'] - 
                    df['cohort_month']
                ).apply(lambda x: x.n)
                
                # Calculate retention matrix
                cohort_data = df.groupby(['cohort_month', 'months_since_join'])[user_col].nunique()
                cohort_sizes = df.groupby('cohort_month')[user_col].nunique()
                retention_matrix = cohort_data.unstack(1).fillna(0)
                retention_pcts = retention_matrix.div(cohort_sizes, axis=0) * 100
                
                # Calculate average retention by month
                avg_retention = retention_pcts.mean()
                
                # If revenue data exists, calculate revenue metrics
                revenue_metrics = None
                if 'revenue' in df.columns:
                    revenue_col = 'revenue'
                    revenue_by_cohort = df.groupby(['cohort_month', 'months_since_join'])[revenue_col].sum()
                    revenue_matrix = revenue_by_cohort.unstack(1).fillna(0)
                    arpu_matrix = revenue_matrix.div(cohort_data.unstack(1))
                    
                    revenue_metrics = {
                        'total_revenue': revenue_matrix.to_dict(),
                        'arpu': arpu_matrix.to_dict(),
                        'cumulative_revenue': revenue_matrix.cumsum(axis=1).to_dict()
                    }
                
                # Visualizations
                # 1. Retention Heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=retention_pcts.values,
                    x=['Month ' + str(x) for x in retention_pcts.columns],
                    y=[str(x) for x in retention_pcts.index],
                    colorscale='Blues',
                    text=np.round(retention_pcts.values, 1),
                    texttemplate='%{text}%'
                ))
                
                fig_heatmap.update_layout(
                    title='Cohort Retention Heatmap',
                    xaxis_title='Months Since Acquisition',
                    yaxis_title='Cohort Month'
                )
                
                # 2. Retention Curves
                fig_curves = go.Figure()
                
                for cohort in retention_pcts.index[-6:]:  # Last 6 cohorts
                    fig_curves.add_trace(go.Scatter(
                        x=retention_pcts.columns,
                        y=retention_pcts.loc[cohort],
                        name=str(cohort)
                    ))
                
                fig_curves.update_layout(
                    title='Retention Curves by Cohort',
                    xaxis_title='Months Since Acquisition',
                    yaxis_title='Retention Rate (%)'
                )
                
                # 3. Cohort Size Trend
                fig_size = go.Figure(data=go.Bar(
                    x=[str(x) for x in cohort_sizes.index],
                    y=cohort_sizes.values,
                    name='Cohort Size'
                ))
                
                fig_size.update_layout(
                    title='Cohort Size Trends',
                    xaxis_title='Cohort Month',
                    yaxis_title='Number of Users'
                )
                
                return {
                    'retention_matrix': retention_pcts.to_dict(),
                    'cohort_sizes': cohort_sizes.to_dict(),
                    'average_retention': avg_retention.to_dict(),
                    'revenue_metrics': revenue_metrics,
                    'visualizations': {
                        'heatmap': fig_heatmap,
                        'curves': fig_curves,
                        'cohort_size': fig_size
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'plotly',
                'datetime'
            ],
            tags=[
                'cohort_analysis',
                'retention',
                'user_behavior',
                'time_based_analysis'
            ]
        ))

        # Churn Prediction Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="churn",
            pattern_name="Churn Prediction Analysis",
            description="Predict and analyze customer churn using machine learning",
            required_columns={
                "user_id": {"min_count": 1},
                "activity_date": {"min_count": 1},
                "features": {"min_count": 1}
            },
            example_prompt="Predict which customers are likely to churn",
            analysis_steps=[
                "Calculate churn labels",
                "Generate features",
                "Train prediction model",
                "Identify risk factors",
                "Score customers"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import (
                    classification_report, 
                    roc_auc_score, 
                    confusion_matrix
                )
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import shap
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                user_col = [col for col in df.columns if 'user' in col.lower()][0]
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Calculate churn (example: no activity in last 30 days)
                last_activity = df.groupby(user_col)[date_col].max()
                max_date = last_activity.max()
                churn_threshold = pd.Timedelta(days=30)
                is_churned = (max_date - last_activity) > churn_threshold
                
                # Prepare features
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in feature_cols 
                            if col not in [user_col, 'churn']]
                
                # Aggregate user features
                user_features = df.groupby(user_col)[feature_cols].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).fillna(0)
                
                # Add churn labels
                user_features['churn'] = is_churned
                
                # Split features and target
                X = user_features.drop('churn', axis=1)
                y = user_features['churn']
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.2, 
                    random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'classification_report': classification_report(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # SHAP values for feature impact
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # Visualizations
                # 1. ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                fig_roc = go.Figure(data=go.Scatter(
                    x=fpr, y=tpr,
                    name='ROC Curve (AUC = %0.2f)' % metrics['roc_auc']
                ))
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    line=dict(dash='dash'),
                    name='Random'
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                
                # 2. Feature Importance
                fig_importance = px.bar(
                    feature_importance.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 20 Feature Importance'
                )
                
                # 3. Confusion Matrix
                cm = metrics['confusion_matrix']
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted No Churn', 'Predicted Churn'],
                    y=['Actual No Churn', 'Actual Churn'],
                    text=cm,
                    texttemplate='%{text}',
                    colorscale='Reds'
                ))
                
                fig_cm.update_layout(title='Confusion Matrix')
                
                # Calculate risk scores for all users
                risk_scores = model.predict_proba(scaler.transform(X))[:, 1]
                risk_data = pd.DataFrame({
                    'user_id': X.index,
                    'risk_score': risk_scores
                })
                
                # Identify high-risk users
                high_risk = risk_data[risk_data['risk_score'] > 0.7]
                
                return {
                    'metrics': metrics,
                    'feature_importance': feature_importance.to_dict(),
                    'risk_scores': risk_data.to_dict(),
                    'high_risk_users': high_risk.to_dict(),
                    'model': {
                        'type': 'random_forest',
                        'features': feature_cols,
                        'parameters': model.get_params()
                    },
                    'visualizations': {
                        'roc_curve': fig_roc,
                        'feature_importance': fig_importance,
                        'confusion_matrix': fig_cm
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'sklearn',
                'plotly',
                'shap'
            ],
            tags=[
                'churn_prediction',
                'machine_learning',
                'customer_retention',
                'risk_analysis'
            ]
        ))

        # Funnel Analysis Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="funnel",
            pattern_name="Conversion Funnel Analysis",
            description="Analyze user conversion through multiple stages of a funnel",
            required_columns={
                "user_id": {"min_count": 1},
                "event_type": {"min_count": 1},
                "timestamp": {"min_count": 1},
                "properties": {"optional": True}
            },
            example_prompt="Show me conversion rates through our sales funnel",
            analysis_steps=[
                "Define funnel stages",
                "Calculate conversion rates",
                "Analyze drop-offs",
                "Measure time between stages",
                "Segment funnel performance"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from datetime import datetime, timedelta
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                user_col = [col for col in df.columns if 'user' in col.lower()][0]
                event_col = [col for col in df.columns if 'event' in col.lower()][0]
                time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
                
                # Convert timestamp
                df[time_col] = pd.to_datetime(df[time_col])
                
                # Define funnel stages (customize based on your events)
                funnel_stages = [
                    'page_view',
                    'add_to_cart',
                    'begin_checkout',
                    'purchase'
                ]
                
                # Calculate overall funnel metrics
                funnel_metrics = {}
                total_users = df[user_col].nunique()
                
                for i, stage in enumerate(funnel_stages):
                    users_in_stage = df[df[event_col] == stage][user_col].nunique()
                    conversion_rate = users_in_stage / total_users * 100 if i == 0 else \
                                    users_in_stage / prev_users * 100
                    drop_off = prev_users - users_in_stage if i > 0 else total_users - users_in_stage
                    
                    funnel_metrics[stage] = {
                        'users': users_in_stage,
                        'conversion_rate': conversion_rate,
                        'drop_off': drop_off
                    }
                    
                    prev_users = users_in_stage
                
                # Calculate time between stages
                stage_times = {}
                for i in range(len(funnel_stages)-1):
                    current_stage = funnel_stages[i]
                    next_stage = funnel_stages[i+1]
                    
                    # Get timestamps for each stage
                    stage_times_df = df[df[event_col].isin([current_stage, next_stage])].copy()
                    stage_times_df = stage_times_df.sort_values(time_col)
                    
                    # Calculate time difference for users who reached both stages
                    time_diffs = []
                    for user in stage_times_df[user_col].unique():
                        user_events = stage_times_df[stage_times_df[user_col] == user]
                        if current_stage in user_events[event_col].values and \
                        next_stage in user_events[event_col].values:
                            current_time = user_events[user_events[event_col] == current_stage][time_col].iloc[0]
                            next_time = user_events[user_events[event_col] == next_stage][time_col].iloc[0]
                            time_diffs.append((next_time - current_time).total_seconds() / 3600)  # hours
                    
                    stage_times[f"{current_stage}_to_{next_stage}"] = {
                        'mean_hours': np.mean(time_diffs) if time_diffs else 0,
                        'median_hours': np.median(time_diffs) if time_diffs else 0,
                        'min_hours': np.min(time_diffs) if time_diffs else 0,
                        'max_hours': np.max(time_diffs) if time_diffs else 0
                    }
                
                # Visualizations
                # 1. Funnel Chart
                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_stages,
                    x=[funnel_metrics[stage]['users'] for stage in funnel_stages],
                    textinfo="value+percent initial"
                ))
                
                fig_funnel.update_layout(title="Conversion Funnel")
                
                # 2. Stage Times
                fig_times = go.Figure()
                
                for transition, times in stage_times.items():
                    fig_times.add_trace(go.Box(
                        y=[transition],
                        x=times['mean_hours'],
                        name=transition
                    ))
                
                fig_times.update_layout(
                    title="Time Between Stages",
                    xaxis_title="Hours",
                    yaxis_title="Stage Transition"
                )
                
                # 3. Daily Conversion Trends
                daily_conversions = df.groupby([
                    pd.Grouper(key=time_col, freq='D'),
                    event_col
                ])[user_col].nunique().unstack()
                
                fig_trends = go.Figure()
                
                for stage in funnel_stages:
                    fig_trends.add_trace(go.Scatter(
                        x=daily_conversions.index,
                        y=daily_conversions[stage],
                        name=stage,
                        mode='lines'
                    ))
                
                fig_trends.update_layout(
                    title="Daily Funnel Performance",
                    xaxis_title="Date",
                    yaxis_title="Users"
                )
                
                return {
                    'funnel_metrics': funnel_metrics,
                    'stage_times': stage_times,
                    'daily_trends': daily_conversions.to_dict(),
                    'visualizations': {
                        'funnel': fig_funnel,
                        'stage_times': fig_times,
                        'daily_trends': fig_trends
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'plotly',
                'datetime'
            ],
            tags=[
                'funnel_analysis',
                'conversion_optimization',
                'user_journey',
                'engagement'
            ]
        ))

        # Product Analytics Pattern
        patterns.append(AnalysisPattern(
            category="business_analytics",
            subcategory="product",
            pattern_name="Product Performance Analytics",
            description="Comprehensive analysis of product performance, usage, and metrics",
            required_columns={
                "product_id": {"min_count": 1},
                "timestamp": {"min_count": 1},
                "user_id": {"min_count": 1},
                "revenue": {"optional": True},
                "category": {"optional": True}
            },
            example_prompt="Analyze our product performance and usage patterns",
            analysis_steps=[
                "Calculate product metrics",
                "Analyze usage patterns",
                "Measure engagement",
                "Track growth trends",
                "Compare product segments"
            ],
            code_template="""
                import pandas as pd
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from datetime import datetime, timedelta
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify columns
                product_col = [col for col in df.columns if 'product' in col.lower()][0]
                time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
                user_col = [col for col in df.columns if 'user' in col.lower()][0]
                
                # Convert timestamp
                df[time_col] = pd.to_datetime(df[time_col])
                
                # Basic product metrics
                product_metrics = df.groupby(product_col).agg({
                    user_col: 'nunique',  # unique users
                    'order_id': 'count',  # total orders
                    'revenue': 'sum' if 'revenue' in df.columns else 'count'  # revenue or count
                }).round(2)
                
                # Calculate market share
                product_metrics['market_share'] = (
                    product_metrics['revenue' if 'revenue' in df.columns else 'order_id'] / 
                    product_metrics['revenue' if 'revenue' in df.columns else 'order_id'].sum() * 100
                ).round(2)
                
                # Time-based analysis
                df['year_month'] = df[time_col].dt.to_period('M')
                monthly_metrics = df.groupby(['year_month', product_col]).agg({
                    user_col: 'nunique',
                    'order_id': 'count',
                    'revenue': 'sum' if 'revenue' in df.columns else 'count'
                }).round(2)
                
                # Product engagement (repeat usage)
                user_product_freq = df.groupby([user_col, product_col]).size().reset_index(name='frequency')
                product_engagement = user_product_freq.groupby(product_col).agg({
                    'frequency': ['mean', 'median', 'max', 'min']
                }).round(2)
                
                # Category analysis if available
                category_metrics = None
                if 'category' in df.columns:
                    category_metrics = df.groupby('category').agg({
                        product_col: 'nunique',  # products per category
                        user_col: 'nunique',     # users per category
                        'revenue': 'sum' if 'revenue' in df.columns else 'count'
                    }).round(2)
                
                # Visualizations
                # 1. Product Performance Overview
                fig_overview = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Market Share', 'User Distribution',
                        'Order Distribution', 'Revenue Distribution'
                    )
                )
                
                # Market share
                fig_overview.add_trace(
                    go.Pie(
                        labels=product_metrics.index,
                        values=product_metrics['market_share'],
                        name='Market Share'
                    ),
                    row=1, col=1
                )
                
                # User distribution
                fig_overview.add_trace(
                    go.Bar(
                        x=product_metrics.index,
                        y=product_metrics[user_col],
                        name='Users'
                    ),
                    row=1, col=2
                )
                
                # Order distribution
                fig_overview.add_trace(
                    go.Bar(
                        x=product_metrics.index,
                        y=product_metrics['order_id'],
                        name='Orders'
                    ),
                    row=2, col=1
                )
                
                # Revenue distribution
                fig_overview.add_trace(
                    go.Bar(
                        x=product_metrics.index,
                        y=product_metrics['revenue' if 'revenue' in df.columns else 'order_id'],
                        name='Revenue'
                    ),
                    row=2, col=2
                )
                
                fig_overview.update_layout(height=800, title_text="Product Performance Overview")
                
                # 2. Time Series Analysis
                fig_trends = go.Figure()
                
                for product in df[product_col].unique():
                    product_data = monthly_metrics.xs(product, level=1)
                    fig_trends.add_trace(go.Scatter(
                        x=product_data.index.astype(str),
                        y=product_data['revenue' if 'revenue' in df.columns else 'order_id'],
                        name=product,
                        mode='lines+markers'
                    ))
                
                fig_trends.update_layout(
                    title="Product Performance Over Time",
                    xaxis_title="Month",
                    yaxis_title="Revenue/Orders"
                )
                
                # 3. Engagement Analysis
                fig_engagement = px.box(
                    user_product_freq,
                    x=product_col,
                    y='frequency',
                    title="Product Usage Distribution"
                )
                
                return {
                    'product_metrics': product_metrics.to_dict(),
                    'monthly_trends': monthly_metrics.to_dict(),
                    'engagement_metrics': product_engagement.to_dict(),
                    'category_metrics': category_metrics.to_dict() if category_metrics is not None else None,
                    'visualizations': {
                        'overview': fig_overview,
                        'trends': fig_trends,
                        'engagement': fig_engagement
                    }
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'plotly',
                'datetime'
            ],
            tags=[
                'product_analytics',
                'performance_metrics',
                'user_engagement',
                'market_analysis'
            ]
        ))

        return patterns

    def _generate_visualization_patterns(self) -> List[AnalysisPattern]:
        """Generate visualization analysis patterns"""
        patterns = []
        
        # Interactive Dashboard
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.VISUALIZATION.value,
            subcategory="dashboard",
            pattern_name="Interactive Data Dashboard",
            description="Create an interactive dashboard with multiple visualizations",
            required_columns={"any": {"min_count": 1}},
            example_prompt="Create an interactive dashboard to explore the data",
            analysis_steps=[
                "Prepare data summary",
                "Create interactive plots",
                "Build distribution visualizations",
                "Generate correlation plots",
                "Add trend analysis"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Create dashboard layout
                dashboard = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        'Numeric Distributions',
                        'Correlation Heatmap',
                        'Time Series Trends',
                        'Category Distributions',
                        'Scatter Matrix',
                        'Box Plots'
                    ),
                    specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                          [{"type": "scatter"}, {"type": "bar"}],
                          [{"type": "scatter"}, {"type": "box"}]]
                )
                
                # Get column types
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                datetime_cols = pd.to_datetime(df[categorical_cols], errors='ignore').notna().any()
                
                # 1. Numeric Distributions
                for i, col in enumerate(numeric_cols):
                    dashboard.add_trace(
                        go.Histogram(
                            x=df[col],
                            name=col,
                            showlegend=True,
                            nbinsx=30,
                            histnorm='probability'
                        ),
                        row=1, col=1
                    )
                
                # 2. Correlation Heatmap
                corr_matrix = df[numeric_cols].corr()
                dashboard.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        showscale=True,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False
                    ),
                    row=1, col=2
                )
                
                # 3. Time Series if datetime exists
                if datetime_cols.any():
                    date_col = datetime_cols[datetime_cols].index[0]
                    df['date'] = pd.to_datetime(df[date_col])
                    for col in numeric_cols[:3]:  # First 3 numeric columns
                        dashboard.add_trace(
                            go.Scatter(
                                x=df['date'],
                                y=df[col],
                                name=col,
                                mode='lines',
                                line=dict(width=2)
                            ),
                            row=2, col=1
                        )
                
                # 4. Category Distributions
                for i, col in enumerate(categorical_cols[:5]):  # First 5 categorical columns
                    value_counts = df[col].value_counts()
                    dashboard.add_trace(
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            name=col,
                            text=value_counts.values,
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
                
                # 5. Scatter Matrix (first 3 numeric columns)
                if len(numeric_cols) >= 2:
                    for i, col1 in enumerate(numeric_cols[:2]):
                        for j, col2 in enumerate(numeric_cols[i+1:3]):
                            dashboard.add_trace(
                                go.Scatter(
                                    x=df[col1],
                                    y=df[col2],
                                    mode='markers',
                                    name=f'{col1} vs {col2}',
                                    marker=dict(
                                        size=6,
                                        opacity=0.6
                                    )
                                ),
                                row=3, col=1
                            )
                
                # 6. Box Plots
                for i, col in enumerate(numeric_cols):
                    dashboard.add_trace(
                        go.Box(
                            y=df[col],
                            name=col,
                            boxpoints='outliers',
                            boxmean=True
                        ),
                        row=3, col=2
                    )
                
                # Update layout with better formatting
                dashboard.update_layout(
                    height=1200,
                    width=1600,
                    title=dict(
                        text="Interactive Data Dashboard",
                        x=0.5,
                        y=0.99,
                        font=dict(size=24)
                    ),
                    showlegend=True,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                # Update axes labels and formatting
                dashboard.update_xaxes(title_text="Value", row=1, col=1)
                dashboard.update_yaxes(title_text="Frequency", row=1, col=1)
                
                if datetime_cols.any():
                    dashboard.update_xaxes(title_text="Date", row=2, col=1)
                    dashboard.update_yaxes(title_text="Value", row=2, col=1)
                
                dashboard.update_xaxes(title_text="Category", row=2, col=2)
                dashboard.update_yaxes(title_text="Count", row=2, col=2)
                
                if len(numeric_cols) >= 2:
                    dashboard.update_xaxes(title_text=numeric_cols[0], row=3, col=1)
                    dashboard.update_yaxes(title_text=numeric_cols[1], row=3, col=1)
                
                dashboard.update_xaxes(title_text="Variable", row=3, col=2)
                dashboard.update_yaxes(title_text="Value", row=3, col=2)
                
                # Create additional interactive visualizations
                
                # Parallel Coordinates
                fig_parallel = px.parallel_coordinates(
                    df[numeric_cols],
                    color=numeric_cols[0],
                    title="Parallel Coordinates Plot"
                )
                
                # 3D Scatter if enough numeric columns
                if len(numeric_cols) >= 3:
                    fig_3d = px.scatter_3d(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        z=numeric_cols[2],
                        color=numeric_cols[0],
                        title="3D Scatter Plot"
                    )
                
                # Return dictionary with all visualizations
                return {
                    "dashboard": dashboard,
                    "parallel_coordinates": fig_parallel,
                    "scatter_3d": fig_3d if len(numeric_cols) >= 3 else None,
                    "numeric_columns": numeric_cols.tolist(),
                    "categorical_columns": categorical_cols.tolist(),
                    "datetime_columns": datetime_cols.tolist()
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'plotly'
            ],
            tags=[
                'visualization',
                'interactive',
                'dashboard',
                'plotly'
            ]
        ))

        # Advanced Statistical Visualization
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.VISUALIZATION.value,
            subcategory="statistical",
            pattern_name="Advanced Statistical Visualization",
            description="Create comprehensive statistical visualizations using seaborn",
            required_columns={"numeric": {"min_count": 1}},
            example_prompt="Create detailed statistical visualizations",
            analysis_steps=[
                "Generate distribution plots",
                "Create statistical relationships",
                "Visualize categorical relationships",
                "Add regression analysis plots"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                from scipy import stats
                
                # Set style
                sns.set_style("whitegrid")
                plt.style.use("seaborn")
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Get column types
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                results = {
                    'distribution_stats': {},
                    'relationship_stats': {},
                    'regression_stats': {}
                }
                
                # 1. Distribution Analysis
                for col in numeric_cols:
                    # Create figure with subplots
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle(f'Distribution Analysis: {col}', size=16)
                    
                    # Histogram with KDE
                    sns.histplot(data=df, x=col, kde=True, ax=axes[0,0])
                    axes[0,0].set_title('Distribution with KDE')
                    
                    # Box plot
                    sns.boxplot(data=df, y=col, ax=axes[0,1])
                    axes[0,1].set_title('Box Plot')
                    
                    # Violin plot
                    sns.violinplot(data=df, y=col, ax=axes[1,0])
                    axes[1,0].set_title('Violin Plot')
                    
                    # Q-Q plot
                    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1,1])
                    axes[1,1].set_title('Q-Q Plot')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Calculate distribution statistics
                    clean_data = df[col].dropna()
                    _, normality_pvalue = stats.normaltest(clean_data)
                    
                    results['distribution_stats'][col] = {
                        'mean': clean_data.mean(),
                        'median': clean_data.median(),
                        'std': clean_data.std(),
                        'skew': stats.skew(clean_data),
                        'kurtosis': stats.kurtosis(clean_data),
                        'is_normal': normality_pvalue > 0.05
                    }
                
                # 2. Relationship Analysis
                if len(numeric_cols) > 1:
                    # Correlation matrix with different methods
                    correlation_methods = ['pearson', 'spearman', 'kendall']
                    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                    
                    for i, method in enumerate(correlation_methods):
                        corr = df[numeric_cols].corr(method=method)
                        sns.heatmap(
                            corr,
                            annot=True,
                            cmap='coolwarm',
                            center=0,
                            fmt='.2f',
                            ax=axes[i]
                        )
                        axes[i].set_title(f'{method.capitalize()} Correlation')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Pair plot with different representations
                    g = sns.PairGrid(df[numeric_cols])
                    g.map_diag(sns.histplot, kde=True)
                    g.map_upper(sns.scatterplot)
                    g.map_lower(sns.kdeplot)
                    plt.show()
                    
                    # Store correlation statistics
                    results['relationship_stats']['correlations'] = {
                        method: df[numeric_cols].corr(method=method).to_dict()
                        for method in correlation_methods
                    }
                
                # 3. Categorical Relationships
                if len(categorical_cols) > 0:
                    for cat_col in categorical_cols:
                        if df[cat_col].nunique() < 10:  # Only for categories with few unique values
                            fig, axes = plt.subplots(
                                len(numeric_cols), 2,
                                figsize=(15, 5 * len(numeric_cols))
                            )
                            
                            for i, num_col in enumerate(numeric_cols):
                                # Box plot
                                sns.boxplot(
                                    data=df,
                                    x=cat_col,
                                    y=num_col,
                                    ax=axes[i,0]
                                )
                                axes[i,0].set_title(f'Box Plot: {cat_col} vs {num_col}')
                                axes[i,0].tick_params(axis='x', rotation=45)
                                
                                # Violin plot
                                sns.violinplot(
                                    data=df,
                                    x=cat_col,
                                    y=num_col,
                                    ax=axes[i,1]
                                )
                                axes[i,1].set_title(f'Violin Plot: {cat_col} vs {num_col}')
                                axes[i,1].tick_params(axis='x', rotation=45)
                                
                                # Calculate statistics
                                stats_dict = df.groupby(cat_col)[num_col].agg([
                                    'mean', 'median', 'std', 'count'
                                ]).to_dict()
                                
                                # ANOVA test
                                groups = [group for _, group in df.groupby(cat_col)[num_col]]
                                f_stat, p_val = stats.f_oneway(*groups)
                                
                                results['relationship_stats'][f'{cat_col}_vs_{num_col}'] = {
                                    'group_stats': stats_dict,
                                    'anova_f_stat': f_stat,
                                    'anova_p_value': p_val
                                }
                            
                            plt.tight_layout()
                            plt.show()
                
                # 4. Regression Analysis
                if len(numeric_cols) > 1:
                    for i, col1 in enumerate(numeric_cols[:-1]):
                        for col2 in numeric_cols[i+1:]:
                            # Create figure
                            plt.figure(figsize=(12, 8))
                            
                            # Regular regression plot
                            sns.regplot(
                                data=df,
                                x=col1,
                                y=col2,
                                scatter_kws={'alpha':0.5},
                                line_kws={'color': 'red'}
                            )
                            
                            # Calculate regression statistics
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                df[col1].dropna(),
                                df[col2].dropna()
                            )
                            
                            plt.title(
                                f'Regression Analysis: {col1} vs {col2}\n'
                                f'R² = {r_value**2:.3f}, p = {p_value:.3e}'
                            )
                            plt.tight_layout()
                            plt.show()
                            
                            # Store regression statistics
                            results['regression_stats'][f'{col1}_vs_{col2}'] = {
                                'slope': slope,
                                'intercept': intercept,
                                'r_squared': r_value**2,
                                'p_value': p_value,
                                'std_error': std_err
                            }
                            
                            # Residual analysis
                            y_pred = slope * df[col1] + intercept
                            residuals = df[col2] - y_pred
                            
                            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                            
                            # Residual plot
                            axes[0,0].scatter(y_pred, residuals, alpha=0.5)
                            axes[0,0].axhline(y=0, color='r', linestyle='--')
                            axes[0,0].set_xlabel('Predicted values')
                            axes[0,0].set_ylabel('Residuals')
                            axes[0,0].set_title('Residual Plot')
                            
                            # Residual histogram
                            sns.histplot(residuals, kde=True, ax=axes[0,1])
                            axes[0,1].set_title('Residual Distribution')
                            
                            # Q-Q plot of residuals
                            stats.probplot(residuals, dist="norm", plot=axes[1,0])
                            axes[1,0].set_title('Q-Q Plot of Residuals')
                            
                            # Scale-Location plot
                            axes[1,1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
                            axes[1,1].set_xlabel('Predicted values')
                            axes[1,1].set_ylabel('Sqrt(|Residuals|)')
                            axes[1,1].set_title('Scale-Location Plot')
                            
                            plt.tight_layout()
                            plt.show()
                
                return results
            """,
            library_imports=[
                'pandas',
                'numpy',
                'matplotlib',
                'seaborn',
                'scipy'
            ],
            tags=[
                'visualization',
                'statistical',
                'seaborn',
                'matplotlib'
            ]
        ))

        # Advanced Geospatial Visualization
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.VISUALIZATION.value,
            subcategory="geospatial",
            pattern_name="Advanced Geospatial Visualization",
            description="Create interactive geospatial visualizations and maps",
            required_columns={
                "latitude": {"min_count": 1},
                "longitude": {"min_count": 1}
            },
            example_prompt="Create interactive maps from my location data",
            analysis_steps=[
                "Prepare geospatial data",
                "Create base maps",
                "Add data layers",
                "Create choropleth maps",
                "Add interactive features"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify latitude and longitude columns
                lat_cols = [col for col in df.columns if 'lat' in col.lower()]
                lon_cols = [col for col in df.columns if 'lon' in col.lower()]
                
                if not (lat_cols and lon_cols):
                    raise ValueError("Could not identify latitude and longitude columns")
                
                lat_col = lat_cols[0]
                lon_col = lon_cols[0]
                
                results = {}
                
                # 1. Basic scatter mapbox
                fig_scatter = px.scatter_mapbox(
                    df,
                    lat=lat_col,
                    lon=lon_col,
                    hover_data=df.columns,
                    zoom=3,
                    title="Data Points Map"
                )
                fig_scatter.update_layout(mapbox_style="open-street-map")
                
                # 2. Heatmap
                fig_heat = go.Figure(go.Densitymapbox(
                    lat=df[lat_col],
                    lon=df[lon_col],
                    radius=10
                ))
                fig_heat.update_layout(
                    mapbox_style="open-street-map",
                    title="Density Heatmap"
                )
                
                # 3. Clustering analysis if enough points
                if len(df) > 10:
                    from sklearn.cluster import DBSCAN
                    from sklearn.preprocessing import StandardScaler
                    
                    # Prepare coordinates for clustering
                    coords = df[[lat_col, lon_col]].values
                    coords_normalized = StandardScaler().fit_transform(coords)
                    
                    # Perform clustering
                    db = DBSCAN(eps=0.3, min_samples=3).fit(coords_normalized)
                    df['cluster'] = db.labels_
                    
                    # Create clustered map
                    fig_clusters = px.scatter_mapbox(
                        df,
                        lat=lat_col,
                        lon=lon_col,
                        color='cluster',
                        hover_data=df.columns,
                        zoom=3,
                        title="Clustered Data Points"
                    )
                    fig_clusters.update_layout(mapbox_style="open-street-map")
                    
                    # Calculate cluster statistics
                    cluster_stats = df.groupby('cluster').agg({
                        lat_col: ['count', 'mean', 'std'],
                        lon_col: ['mean', 'std']
                    }).to_dict()
                    
                    results['cluster_analysis'] = cluster_stats
                
                # 4. Distance analysis
                if len(df) > 1:
                    from scipy.spatial.distance import pdist, squareform
                    
                    # Calculate distance matrix
                    coords = df[[lat_col, lon_col]].values
                    distances = pdist(coords)
                    distance_matrix = squareform(distances)
                    
                    # Create distance heatmap
                    fig_distance = go.Figure(data=go.Heatmap(
                        z=distance_matrix,
                        colorscale='Viridis'
                    ))
                    fig_distance.update_layout(
                        title='Distance Matrix Heatmap',
                        xaxis_title='Point Index',
                        yaxis_title='Point Index'
                    )
                    
                    results['distance_analysis'] = {
                        'mean_distance': np.mean(distances),
                        'max_distance': np.max(distances),
                        'min_distance': np.min(distances[distances > 0])
                    }
                
                # 5. Time analysis if datetime column exists
                datetime_cols = pd.to_datetime(
                    df.select_dtypes(include=['object']).apply(pd.to_datetime, errors='coerce')
                ).notna().any()
                
                if datetime_cols.any():
                    date_col = datetime_cols[datetime_cols].index[0]
                    df['date'] = pd.to_datetime(df[date_col])
                    
                    # Animated map
                    fig_time = px.scatter_mapbox(
                        df,
                        lat=lat_col,
                        lon=lon_col,
                        animation_frame=df['date'].dt.strftime('%Y-%m-%d'),
                        hover_data=df.columns,
                        zoom=3,
                        title="Time-based Animation"
                    )
                    fig_time.update_layout(mapbox_style="open-street-map")
                    
                    # Temporal statistics
                    temporal_stats = df.groupby(df['date'].dt.date).agg({
                        lat_col: ['count', 'mean', 'std'],
                        lon_col: ['mean', 'std']
                    }).to_dict()
                    
                    results['temporal_analysis'] = temporal_stats
                
                return {
                    'scatter_map': fig_scatter,
                    'heatmap': fig_heat,
                    'cluster_map': fig_clusters if len(df) > 10 else None,
                    'distance_heatmap': fig_distance if len(df) > 1 else None,
                    'time_animation': fig_time if datetime_cols.any() else None,
                    'analysis_results': results
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'plotly',
                'sklearn',
                'scipy'
            ],
            tags=[
                'geospatial',
                'mapping',
                'clustering',
                'interactive'
            ]
        ))

        # Network Visualization
        patterns.append(AnalysisPattern(
            category=AnalysisCategory.VISUALIZATION.value,
            subcategory="network",
            pattern_name="Network Graph Visualization",
            description="Create interactive network visualizations and graph analytics",
            required_columns={
                "source": {"min_count": 1},
                "target": {"min_count": 1}
            },
            example_prompt="Visualize my network data with interactive graphs",
            analysis_steps=[
                "Prepare network data",
                "Create network graph",
                "Calculate network metrics",
                "Visualize communities",
                "Add interactivity"
            ],
            code_template=r"""
                import pandas as pd
                import numpy as np
                import networkx as nx
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import community
                
                # Read data
                df = pd.read_csv(csv_path)
                
                # Identify source and target columns
                source_col = [col for col in df.columns if 'source' in col.lower()][0]
                target_col = [col for col in df.columns if 'target' in col.lower()][0]
                
                # Create network graph
                G = nx.from_pandas_edgelist(df, source_col, target_col)
                
                # Calculate layout
                pos = nx.spring_layout(G)
                
                # Basic network metrics
                metrics = {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'density': nx.density(G),
                    'avg_clustering': nx.average_clustering(G),
                    'avg_degree': sum(dict(G.degree()).values()) / len(G),
                    'components': list(nx.connected_components(G))
                }
                
                # Detect communities
                communities = community.best_partition(G)
                
                # Node centrality measures
                centrality_measures = {
                    'degree': nx.degree_centrality(G),
                    'betweenness': nx.betweenness_centrality(G),
                    'closeness': nx.closeness_centrality(G),
                    'eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
                }
                
                # Create interactive visualization
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=10,
                        colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                        )
                    )
                )

                # Add node attributes
                node_adjacencies = []
                node_text = []
                for node in G.nodes():
                    adjacencies = len(list(G.neighbors(node)))
                    node_adjacencies.append(adjacencies)
                    node_text.append(
                        f'Node: {node}<br>'
                        f'# Connections: {adjacencies}<br>'
                        f'Community: {communities[node]}<br>'
                        f'Degree Centrality: {centrality_measures["degree"][node]:.3f}<br>'
                        f'Betweenness: {centrality_measures["betweenness"][node]:.3f}'
                    )

                node_trace.marker.color = node_adjacencies
                node_trace.text = node_text

                # Create figure
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Network Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                # Create community visualization
                community_colors = [communities[node] for node in G.nodes()]
                community_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=community_colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Community')
                    ),
                    text=node_text,
                    hoverinfo='text'
                )

                fig_communities = go.Figure(
                    data=[edge_trace, community_trace],
                    layout=go.Layout(
                        title='Community Detection',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                # Create centrality comparison
                fig_centrality = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=list(centrality_measures.keys())
                )

                for i, (measure, values) in enumerate(centrality_measures.items()):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    ccentrality_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=list(values.values()),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=measure)
                        ),
                        text=[f'{node}: {values[node]:.3f}' for node in G.nodes()],
                        hoverinfo='text'
                    )
                    
                    fig_centrality.add_trace(
                        centrality_trace,
                        row=row,
                        col=col
                    )
                    
                    fig_centrality.update_xaxes(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        row=row,
                        col=col
                    )
                    fig_centrality.update_yaxes(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        row=row,
                        col=col
                    )

                fig_centrality.update_layout(
                    height=800,
                    title_text="Centrality Measures Comparison",
                    showlegend=False
                )

                # Degree distribution analysis
                degrees = [d for n, d in G.degree()]
                fig_degree_dist = go.Figure(data=[
                    go.Histogram(
                        x=degrees,
                        nbinsx=max(10, len(set(degrees))),
                        name='Degree Distribution'
                    )
                ])
                fig_degree_dist.update_layout(
                    title='Node Degree Distribution',
                    xaxis_title='Degree',
                    yaxis_title='Count'
                )

                # Calculate and visualize path lengths
                path_lengths = []
                for c in nx.connected_components(G):
                    subgraph = G.subgraph(c)
                    if len(subgraph) > 1:
                        lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                        path_lengths.extend(
                            length for paths in lengths.values() 
                            for length in paths.values()
                        )

                fig_path_dist = go.Figure(data=[
                    go.Histogram(
                        x=path_lengths,
                        nbinsx=max(10, len(set(path_lengths))),
                        name='Path Length Distribution'
                    )
                ])
                fig_path_dist.update_layout(
                    title='Shortest Path Length Distribution',
                    xaxis_title='Path Length',
                    yaxis_title='Count'
                )

                return {
                    'basic_network': fig,
                    'community_network': fig_communities,
                    'centrality_comparison': fig_centrality,
                    'degree_distribution': fig_degree_dist,
                    'path_length_distribution': fig_path_dist,
                    'metrics': metrics,
                    'centrality_measures': {
                        measure: dict(values)
                        for measure, values in centrality_measures.items()
                    },
                    'communities': dict(communities)
                }
            """,
            library_imports=[
                'pandas',
                'numpy',
                'networkx',
                'plotly',
                'community'
            ],
            tags=[
                'network',
                'graph',
                'community_detection',
                'centrality'
            ]
        ))

        return patterns

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize storage manager
    storage = StorageManager()
    
    # Initialize and generate knowledge base
    generator = BaseKnowledgeGenerator(storage)
    
    try:
        generator.generate_knowledge_base()
        logger.info(f"Successfully generated knowledge base in {generator.output_dir}")
    except Exception as e:
        logger.error(f"Error generating knowledge base: {str(e)}")
        raise