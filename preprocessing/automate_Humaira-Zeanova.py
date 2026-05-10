import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


def load_data(input_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        input_path (str): Path ke file CSV dataset mentah.

    Returns:
        pd.DataFrame: DataFrame hasil pembacaan CSV.
    """
    df = pd.read_csv(input_path)
    print(f"[load_data] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak relevan untuk pemodelan (kolom 'id').

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa kolom 'id'.
    """
    df = df.drop(columns=['id'])
    print(f"[drop_unnecessary_columns] Kolom 'id' dihapus. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values pada kolom 'bmi' dengan imputasi median.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    bmi_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_median)
    print(f"[handle_missing_values] Kolom 'bmi' diimputasi dengan median = {bmi_median:.2f}")
    print(f"  Total missing values tersisa: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa baris duplikat.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[remove_duplicates] Duplikat dihapus: {removed} baris. Shape: {df.shape}")
    return df


def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris noise, yaitu baris dengan nilai 'gender' = 'Other'
    karena hanya berjumlah 1 baris dan tidak representatif.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa baris noise.
    """
    before = len(df)
    df = df[df['gender'] != 'Other'].reset_index(drop=True)
    removed = before - len(df)
    print(f"[remove_noise] Baris gender='Other' dihapus: {removed} baris. Shape: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier pada kolom numerik menggunakan metode IQR Capping.
    Kolom yang ditangani: 'avg_glucose_level' dan 'bmi'.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah di-cap.
    """
    outlier_cols = ['avg_glucose_level', 'bmi']

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[handle_outliers] '{col}': {n_outlier} outlier di-cap ke [{lower:.2f}, {upper:.2f}]")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada fitur kategorikal:
    - Label Encoding: 'gender', 'ever_married', 'Residence_type' (biner)
    - One-Hot Encoding: 'work_type', 'smoking_status' (multi-class)

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan fitur kategorikal yang sudah di-encode.
    """
    le = LabelEncoder()

    binary_cols = ['gender', 'ever_married', 'Residence_type']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        print(f"[encode_features] Label Encoding '{col}' selesai.")

    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=False)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[encode_features] One-Hot Encoding 'work_type' & 'smoking_status' selesai. Shape: {df.shape}")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan standarisasi pada fitur numerik menggunakan StandardScaler.
    Kolom yang di-scale: 'age', 'avg_glucose_level', 'bmi'.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame dengan fitur numerik yang sudah di-standarisasi.
    """
    scaler = StandardScaler()
    scale_cols = ['age', 'avg_glucose_level', 'bmi']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[scale_features] StandardScaler diterapkan pada kolom: {scale_cols}")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Args:
        df (pd.DataFrame): DataFrame hasil preprocessing.
        output_path (str): Path tujuan penyimpanan file CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    df.to_csv(output_path, index=False)
    print(f"[save_data] Dataset preprocessing disimpan ke: {output_path}")
    print(f"  Shape tersimpan: {df.shape}")


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing secara otomatis.

    Tahapan:
        1. Load data
        2. Drop kolom tidak relevan ('id')
        3. Tangani missing values (imputasi median pada 'bmi')
        4. Hapus duplikat
        5. Hapus noise (gender='Other')
        6. Tangani outlier (IQR Capping)
        7. Encoding fitur kategorikal
        8. Standarisasi fitur numerik
        9. Simpan hasil

    Args:
        input_path (str): Path ke file CSV dataset mentah.
        output_path (str): Path tujuan penyimpanan dataset hasil preprocessing.

    Returns:
        pd.DataFrame: Dataset yang sudah siap dilatih.
    """
    print("=" * 55)
    print("   PIPELINE PREPROCESSING — STROKE PREDICTION DATASET")
    print("=" * 55)

    df = load_data(input_path)
    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_noise(df)
    df = handle_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    save_data(df, output_path)

    print("=" * 55)
    print(f"   PREPROCESSING SELESAI")
    print(f"   Shape akhir  : {df.shape}")
    print(f"   Total NaN    : {df.isnull().sum().sum()}")
    print(f"   Output       : {output_path}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    INPUT_PATH  = "healthcare-dataset-stroke-data.csv"   
    OUTPUT_PATH = "preprocessing/stroke_preprocessing.csv"  

    df_result = preprocess(INPUT_PATH, OUTPUT_PATH)
