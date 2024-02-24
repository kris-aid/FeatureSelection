import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def excel_to_dataframe(file_path, sheet_name=0, header_row=0):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        return df
    except Exception as e:
        print("An error occurred:", str(e))
        return None
def process_dataset(df):
    df['espectrograma etiqueta'] = range(1, len(df) + 1)
    processed_df = df.dropna()
    word_mapping = {'Long Period': 1, 'Volcano Tectonic': 0, 'Hybrid':1,'Tremor':0 }
    processed_df['Etiqueta'] = processed_df['Etiqueta'].replace(word_mapping)
    processed_df = min_max_norm(processed_df)
    return processed_df
def delete_outlayers(df):
    z_scores = stats.zscore(df)
    threshold = 4
    outlier_indices = (z_scores > threshold).any(axis=1)
    df_filtered = df[~outlier_indices]
    return df_filtered
def min_max_norm(df):
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    escaled_df = pd.DataFrame(df_normalized, columns=df.columns)
    return escaled_df
# Example usage:
file_path = "features completas.xlsx"
dataframe_from_excel = excel_to_dataframe(file_path)
if dataframe_from_excel is not None:
    print(dataframe_from_excel.head())  # Display the first few rows of the DataFrame
print(dataframe_from_excel.iloc[235])
processed_df=process_dataset(dataframe_from_excel)
print(processed_df.iloc[235])
print(processed_df)
filtered_df=delete_outlayers(processed_df)
print(min_max_norm(filtered_df))
