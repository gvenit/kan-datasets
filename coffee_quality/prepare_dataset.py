"""Coffee Quality Dataset Processing Module"""

import sys
import os
import json
import pandas as pd
import numpy as np
import urllib.request
from typing import Literal, Optional, Dict, List, Tuple

from sympy import N
from date_cleaner import date_parser, calculate_month_difference, months_to_int_map, int_to_month_map

# ============================================================================
# CONFIGURATION
# ============================================================================

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

DATASET_DIR = os.path.join(THIS_DIR, 'dataset/')
ARABICA_PATH = os.path.join(DATASET_DIR, 'arabica_data_cleaned.csv')
ROBUSTA_PATH = os.path.join(DATASET_DIR, 'robusta_data_cleaned.csv')
CLEANED_PATH = os.path.join(DATASET_DIR, 'coffee_quality_cleaned.csv')
LABELS_PATH = os.path.join(DATASET_DIR, 'labels.json')
STATS_PATH = os.path.join(DATASET_DIR, 'normalization_stats.json')

# Column definitions
DROP_COLS = [
    'Certification.Address', 'Certification.Contact', 'ICO.Number', 'Altitude', 
    'unit_of_measurement', 'altitude_low_meters', 'altitude_high_meters', 
    'Owner.1', 'Mill', 'Lot.Number', 'Farm.Name', 'Owner', 'Company', 
    'Region', 'Producer', 'Bag.Weight', 'Number.of.Bags', 'altitude_min_meters',
    'altitude_high_meters'
]

NUMERIC_COLS = [
    'Harvest_to_Grading_Months', 'Grading_to_Expiration_Months', 
    'altitude_mean_meters', 'Total.Cup.Points', 'Moisture', 
    'Category.One.Defects', 'Quakers', 'Category.Two.Defects'
]

QUALITY_SCORE_COLS = [
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance',
    'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points'
]

CATEGORY_COLS = [
    'Species', 'Country.of.Origin', 'In.Country.Partner', 'Variety', 
    'Processing.Method', 'Color', 'Certification.Body', 'Harvest_Month', 'Harvest_Year'
]

EXCLUDE_FROM_LABELS = NUMERIC_COLS.copy()

# URL constants
ARABICA_URL = 'https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv'
ROBUSTA_URL = 'https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/robusta_data_cleaned.csv'

# ============================================================================
# DATASET BUILDING
# ============================================================================

def _download_dataset(force: bool = False) -> None:
    """Download dataset files if they don't exist or if forced."""
    if force or not os.path.exists(ARABICA_PATH) or not os.path.exists(ROBUSTA_PATH):
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        # print(f'Downloading arabica data from {ARABICA_URL}...')
        urllib.request.urlretrieve(ARABICA_URL, ARABICA_PATH)
        
        # print(f'Downloading robusta data from {ROBUSTA_URL}...')
        urllib.request.urlretrieve(ROBUSTA_URL, ROBUSTA_PATH)
        # print('Download complete!')
    # else:
    #     print('Dataset files already exist. Skipping download.')


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Build and return the combined coffee quality dataset."""
    # print('Building coffee quality dataset...')
    
    _download_dataset(force)
    return get_dataset()


def _clean_column_names(df_robusta: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names between arabica and robusta datasets."""
    column_mapping = {
        'Fragrance...Aroma': 'Aroma',
        'Salt...Acid': 'Acidity',
        'Bitter...Sweet': 'Sweetness',
        'Mouthfeel': 'Body',
        'Uniform.Cup': 'Uniformity'
    }
    return df_robusta.rename(columns=column_mapping)


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning operations to a dataframe."""
    # Remove index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Clean object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = (
            df[col]
            .fillna('Unknown')
            .replace('Other', 'Unknown')
            .str.strip()
        )
    
    # Standardize color values
    if 'Color' in df.columns:
        df['Color'] = df['Color'].replace('Bluish-Green', 'Blue-Green')
    
    return df


def _fix_arabica_inconsistencies(df_arabica: pd.DataFrame) -> pd.DataFrame:
    """Fix known typos and inconsistencies in arabica dataset."""
    df_arabica['In.Country.Partner'] = df_arabica['In.Country.Partner'].replace(
        'Specialty Coffee Ass', 'Specialty Coffee Association'
    )
    df_arabica['Country.of.Origin'] = df_arabica['Country.of.Origin'].replace(
        'Cote d?Ivoire', "Côte d'Ivoire"
    )
    df_arabica['Country.of.Origin'] = df_arabica['Country.of.Origin'].replace(
        'Tanzania, United Republic Of', 'Tanzania'
    )
    return df_arabica


def get_dataset(combine: bool = True) -> pd.DataFrame:
    """Load and optionally combine arabica and robusta datasets."""
    # print('Loading and combining datasets...')
    
    df_arabica = pd.read_csv(ARABICA_PATH, encoding='utf-8')
    df_robusta = pd.read_csv(ROBUSTA_PATH, encoding='utf-8')
    
    # Add species identifier
    df_arabica['Species'] = 'Arabica'
    df_robusta['Species'] = 'Robusta'
    
    # Standardize column names
    df_robusta = _clean_column_names(df_robusta)
    
    # Apply basic cleaning
    df_arabica = _basic_cleaning(df_arabica)
    df_robusta = _basic_cleaning(df_robusta)
    
    # Fix arabica inconsistencies
    df_arabica = _fix_arabica_inconsistencies(df_arabica)
    
    if not combine:
        return df_arabica, df_robusta
    
    return pd.concat([df_arabica, df_robusta], ignore_index=True)

# ============================================================================
# DATA CLEANING
# ============================================================================

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and extract date information from date columns."""
    df = df.copy()
    
    # Parse Grading Date
    df['Grading.Date'] = pd.to_datetime(df['Grading.Date'], errors='coerce')
    df['Grading_Month'] = df['Grading.Date'].dt.month.fillna(0).astype(int)
    df['Grading_Year'] = df['Grading.Date'].dt.year
    
    # Parse Harvest Year
    harvest_info = df['Harvest.Year'].apply(
        lambda x: date_parser(x, reference_year=None)
    )
    df['Harvest_Month'] = harvest_info.apply(
        lambda x: months_to_int_map.get(x[0], 0)
    )
    df['Harvest_Year'] = harvest_info.apply(lambda x: x[1])
    
    # Parse Expiration
    df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
    df['Expiration_Month'] = df['Expiration'].dt.month.fillna(0).astype(int)
    df['Expiration_Year'] = df['Expiration'].dt.year
    
    return df


def _fill_missing_years(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing year values using cascading logic."""
    # Create masks for missing values
    missing_grading = df['Grading_Year'].isna()
    missing_harvest = df['Harvest_Year'].isna() | (df['Harvest_Year'] == 0)
    missing_expiration = df['Expiration_Year'].isna()
    
    # Cascading fill logic
    df.loc[missing_grading & ~missing_harvest, 'Grading_Year'] = \
        df.loc[missing_grading & ~missing_harvest, 'Harvest_Year']
    
    missing_grading = df['Grading_Year'].isna()
    df.loc[~missing_grading & missing_harvest, 'Harvest_Year'] = \
        df.loc[~missing_grading & missing_harvest, 'Grading_Year']
    
    missing_grading = df['Grading_Year'].isna()
    missing_expiration = df['Expiration_Year'].isna()
    df.loc[~missing_grading & missing_expiration, 'Expiration_Year'] = \
        df.loc[~missing_grading & missing_expiration, 'Grading_Year'] + 1
    
    missing_grading = df['Grading_Year'].isna()
    missing_harvest = df['Harvest_Year'].isna() | (df['Harvest_Year'] == 0)
    all_missing_mask = missing_grading & missing_harvest & ~missing_expiration
    df.loc[all_missing_mask, 'Grading_Year'] = \
        df.loc[all_missing_mask, 'Expiration_Year'] - 1
    df.loc[all_missing_mask, 'Harvest_Year'] = \
        df.loc[all_missing_mask, 'Expiration_Year'] - 1
    
    return df


def _calculate_month_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate month differences between key dates."""
    def _safe_month_diff(row, from_cols, to_cols):
        """Safe wrapper for month difference calculation."""
        from_date = [
            int_to_month_map.get(row[from_cols[0]], 'Unknown'),
            row[from_cols[1]]
        ]
        to_date = [
            int_to_month_map.get(row[to_cols[0]], 'Unknown'),
            row[to_cols[1]]
        ]
        return calculate_month_difference(from_date, to_date)
    
    df['Harvest_to_Grading_Months'] = df.apply(
        lambda row: _safe_month_diff(
            row, ['Harvest_Month', 'Harvest_Year'], ['Grading_Month', 'Grading_Year']
        ), axis=1
    )
    
    df['Grading_to_Expiration_Months'] = df.apply(
        lambda row: _safe_month_diff(
            row, ['Grading_Month', 'Grading_Year'], ['Expiration_Month', 'Expiration_Year']
        ), axis=1
    )
    
    return df


def clean_dataset(
    df: pd.DataFrame, 
    remove_cols: Optional[List[str]] = DROP_COLS,
    treat_quality_as_categorical: bool = True
) -> pd.DataFrame:
    """
    Clean the dataset by removing columns, parsing dates, and handling missing values.
    
    Args:
        df: Input DataFrame
        remove_cols: Columns to remove (uses DROP_COLS by default)
        treat_quality_as_categorical: Whether to treat quality scores as categorical
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    remove_cols = remove_cols or DROP_COLS
    
    # Remove specified columns
    cols_to_remove = [col for col in remove_cols if col in df.columns]
    df = df.drop(columns=cols_to_remove)
    
    # Parse dates
    df = _parse_dates(df)
    
    # Fill missing years
    df = _fill_missing_years(df)
    
    # Drop rows with all years missing
    all_years_missing = (
        df['Grading_Year'].isna() & 
        (df['Harvest_Year'].isna() | (df['Harvest_Year'] == 0)) & 
        df['Expiration_Year'].isna()
    )
    
    if all_years_missing.sum() > 0:
        # print(f'Dropped {all_years_missing.sum()} rows with all year columns missing')
        df = df[~all_years_missing].copy()
    
    # Convert year columns to integers
    year_cols = ['Grading_Year', 'Harvest_Year', 'Expiration_Year']
    df[year_cols] = df[year_cols].astype('Int64')
    
    # Remove outliers (all-zero quality scores)
    zero_quality_mask = (df[QUALITY_SCORE_COLS] == 0).all(axis=1)
    if zero_quality_mask.sum() > 0:
        # print(f'Dropped {zero_quality_mask.sum()} outlier rows with all-zero quality scores')
        df = df[~zero_quality_mask].copy()
    
    # Calculate month differences
    df = _calculate_month_differences(df)
    
    # Process quality scores
    if treat_quality_as_categorical:
        for col in QUALITY_SCORE_COLS:
            if col in df.columns:
                df[col] = df[col].round().astype('Int64')
    
    # Convert date columns to categorical
    df['Harvest_Month'] = df['Harvest_Month'].map(int_to_month_map).astype('category')
    df['Harvest_Year'] = df['Harvest_Year'].astype('category')
    
    # Drop redundant columns
    redundant_cols = [
        'Grading.Date', 'Expiration', 'Harvest.Year', 
        'Grading_Month', 'Grading_Year', 'Expiration_Month', 'Expiration_Year'
    ]
    df = df.drop(columns=[col for col in redundant_cols if col in df.columns])
    
    # Handle missing values
    # initial_rows = len(df)
    # nan_counts = df.isna().sum()
    # if not nan_counts[nan_counts > 0].empty:
    #     print('NaN counts in columns:')
    #     for col, count in nan_counts[nan_counts > 0].items():
    #         print(f'  {col}: {count}')
    df = df.dropna()
    # dropped_rows = initial_rows - len(df)
    # if dropped_rows > 0:
    #     print(f'Dropped {dropped_rows} rows with NaN values in numeric columns')
    
    return df

# ============================================================================
# LABEL ENCODING
# ============================================================================

def create_labels(
    df: pd.DataFrame, 
    label_enumeration: Literal['linear', 'exponential'] = 'linear', 
    force: bool = False,
    exclude_cols: Optional[List[str]] = None,
    treat_quality_as_categorical: bool = True
) -> Dict:
    """
    Create a dictionary mapping categorical column values to numerical labels.
    
    Args:
        df: Input DataFrame
        label_enumeration: Method for assigning label values
        force: Whether to force recreation of labels file
        exclude_cols: Columns to exclude from labeling
        treat_quality_as_categorical: Whether quality scores are treated as categorical
    
    Returns:
        Dictionary of label mappings
    """
    exclude_cols = exclude_cols or EXCLUDE_FROM_LABELS
    
    if force or not os.path.exists(LABELS_PATH):
        label_dict = {}
        
        # Determine which columns to process
        categorical_cols = CATEGORY_COLS.copy()
        if treat_quality_as_categorical:
            categorical_cols.extend(QUALITY_SCORE_COLS)
        
        for col in categorical_cols:
            if col not in df.columns or col in exclude_cols:
                continue
            
            label_dict[col] = {}
            unique_vals = sorted(df[col].astype(str).unique())
            
            for idx, val in enumerate(unique_vals, start=(1 if label_enumeration == 'linear' else 0)):
                if label_enumeration == 'exponential':
                    label_dict[col][val] = 2 ** idx
                else:  # linear
                    label_dict[col][val] = idx
        
        # Save labels to file
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(LABELS_PATH, 'w', encoding='utf-8') as f:
            json.dump(label_dict, f, indent=4, ensure_ascii=False)
    
    # Load labels from file
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_safe_column_name(label: str) -> str:
    """Create a safe column name by replacing special characters."""
    return label.replace(' ', '_').replace('.', '_').replace('-', '_')


def set_df_labels(
    df: pd.DataFrame, 
    label_dict: Optional[Dict] = None,
    treat_quality_as_categorical: bool = True
) -> pd.DataFrame:
    """Replace categorical values with numerical labels."""
    df = df.copy()
    
    if label_dict is None:
        label_dict = create_labels(df, treat_quality_as_categorical=treat_quality_as_categorical)
    
    for col, labels in label_dict.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda x: labels.get(x, labels.get('Unknown', 0)))
            )
    
    return df.sort_index(axis=1)


def expand_df_labels(
    df: pd.DataFrame, 
    label_dict: Optional[Dict] = None,
    treat_quality_as_categorical: bool = True
) -> pd.DataFrame:
    """Expand categorical columns into one-hot encoded columns."""
    df = df.copy()
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Convert categorical integer columns to strings
    categorical_int_cols = []
    if treat_quality_as_categorical:
        categorical_int_cols.extend(QUALITY_SCORE_COLS)
    categorical_int_cols.extend(['Harvest_Month', 'Harvest_Year'])
    
    for col in categorical_int_cols:
        if col in df.columns and col in numeric_cols:
            df[col] = df[col].astype(str)
            categorical_cols = categorical_cols.union([col])
    
    # Create label dictionary if not provided
    if label_dict is None:
        label_dict = {
            col: {str(v): None for v in df[col].dropna().unique()}
            for col in categorical_cols
        }
    
    new_columns = {}
    columns_to_drop = []
    
    # Create one-hot encoded columns
    for col, labels in label_dict.items():
        if col not in df.columns or col not in categorical_cols:
            continue
        
        columns_to_drop.append(col)
        
        # Create columns for each label
        for label in labels.keys():
            safe_label = _create_safe_column_name(label)
            new_columns[f'{col}_Is_{safe_label}'] = (df[col] == label).astype(int)
        
        # Create unknown indicator
        new_columns[f'{col}_Is_Unknown'] = (
            df[col].isin(['Unknown', 'nan', 'None', np.nan])
        ).astype(int)
    
    # Drop original categorical columns and add new ones
    df = df.drop(columns=columns_to_drop)
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    return df.sort_index(axis=1)

# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_dataset(
    df: pd.DataFrame, 
    method: Literal['minmax', 'standard', 'none'] = 'minmax', 
    reverse: bool = False
) -> pd.DataFrame:
    """
    Normalize numeric columns in the dataset.
    
    Args:
        df: Input DataFrame
        method: Normalization method ('minmax', 'standard', or 'none')
        reverse: Whether to reverse the normalization
    
    Returns:
        Normalized DataFrame
    """
    if method == 'none':
        return df.copy()
    
    df = df.copy()
    if reverse:
        if not os.path.exists(STATS_PATH):
            raise ValueError("Normalization stats not found. Cannot reverse normalization.")
        
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        
        if method == 'minmax':
            for col in df.columns:
                if col in stats and 'min' in stats[col] and 'max' in stats[col]:
                    min_val, max_val = stats[col]['min'], stats[col]['max']
                    df[col] = df[col] * (max_val - min_val) + min_val
        elif method == 'standard':
            for col in df.columns:
                if col in stats and 'mean' in stats[col] and 'std' in stats[col]:
                    mean_val, std_val = stats[col]['mean'], stats[col]['std']
                    df[col] = df[col] * std_val + mean_val
    else:
        stats = {}
        # Get numeric columns that actually exist in the DataFrame
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val, max_val = df[col].min(), df[col].max()
                stats[col] = {'min': float(min_val), 'max': float(max_val)}
                
                if max_val - min_val != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in numeric_cols:
                mean_val, std_val = df[col].mean(), df[col].std()
                stats[col] = {'mean': float(mean_val), 'std': float(std_val)}
                
                if std_val != 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        # Save normalization stats
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f, indent=2)
    
    return df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    force_download: bool = False,
    treat_quality_as_categorical: bool = False,
    save_cleaned: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete data processing pipeline.
    
    Returns:
        Tuple of (raw_data, cleaned_data, labeled_data, label_dict)
    """
    print('=== Coffee Quality Data Processing Pipeline ===\n')
    
    # 1. Build dataset
    print('1. Building dataset...')
    df_raw = build_dataset(force=force_download)
    
    # 2. Clean dataset (including NaN removal)
    print(f'\n2. Cleaning dataset (quality_as_categorical={treat_quality_as_categorical})...')
    df_cleaned = clean_dataset(df_raw, treat_quality_as_categorical=treat_quality_as_categorical)
    
    if save_cleaned:
        df_cleaned.to_csv(CLEANED_PATH, index=False, encoding='utf-8')
        print(f'\nSaved cleaned dataset to: {CLEANED_PATH}')
    
    # 3. Create labels
    print('\n3. Creating label mappings...')
    label_dict = create_labels(
        df_cleaned, 
        force=True, 
        treat_quality_as_categorical=treat_quality_as_categorical
    )
    
    # 4. Apply labels
    print('\n4. Applying label encoding...')
    df_labeled = set_df_labels(df_cleaned.copy(), label_dict, treat_quality_as_categorical)
    
    # 5. Expand labels (one-hot encoding)
    print('\n5. Expanding labels (one-hot encoding)...')
    df_expanded = expand_df_labels(df_cleaned.copy(), label_dict, treat_quality_as_categorical)
    
    # 6. Normalize
    print('\n6. Normalizing dataset...')
    df_normalized = normalize_dataset(df_expanded, method='minmax')
    
    print('\n=== Pipeline Complete ===')
    print(f'Raw dataset shape: {df_raw.shape}')
    print(f'Cleaned dataset shape: {df_cleaned.shape}')
    print(f'Labeled dataset shape: {df_labeled.shape}')
    print(f'Expanded dataset shape: {df_expanded.shape}')
    print(f'Normalized dataset shape: {df_normalized.shape}')
    
    # Show numeric column statistics
    print('\nNumeric column statistics:')
    for col in NUMERIC_COLS + (QUALITY_SCORE_COLS if treat_quality_as_categorical else []):
        print(f'  {col}: min={df_cleaned[col].min():.2f}, max={df_cleaned[col].max():.2f}')
    
    return df_raw, df_cleaned, df_normalized, label_dict


def get_prepared_dataset(treat_quality_as_categorical: bool = False) -> pd.DataFrame:
    """
    Get the fully prepared, expanded, and normalized dataset for training.
    
    This is a convenience function that handles the complete pipeline:
    - Load/build raw dataset
    - Clean the dataset
    - Create and apply labels
    - Expand categorical columns (one-hot encoding)
    - Normalize numeric columns
    
    Args:
        treat_quality_as_categorical: Whether to treat quality scores as categorical
        
    Returns:
        Fully prepared DataFrame ready for training
    """
    df_cleaned = clean_dataset(
        build_dataset(force=False),
        treat_quality_as_categorical=treat_quality_as_categorical
    )
    
    label_dict = create_labels(
        df_cleaned,
        force=False,
        treat_quality_as_categorical=treat_quality_as_categorical
    )
    
    df_expanded = expand_df_labels(
        df=df_cleaned,
        label_dict=label_dict,
        treat_quality_as_categorical=treat_quality_as_categorical
    )
    
    df_normalized = normalize_dataset(df_expanded, method='minmax')
    
    return df_normalized


if __name__ == '__main__':
    # Run the complete pipeline
    df_raw, df_cleaned, df_normalized, labels = run_pipeline(
        force_download=False,
        treat_quality_as_categorical=True,
        save_cleaned=True
    )