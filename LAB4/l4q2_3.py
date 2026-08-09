# AI tool used: ChatGPT for function creation
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
# Lists of attributes
ordinal = ['Education']
nominal = ['Marital_Status', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain','Response']


def data_clean(df):
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Fill missing numerical values with median
    numerical_columns = df.select_dtypes(include=["number"]).columns
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def ai_label(df, ordinal):
    # Label encoding for ordinal attributes
    label_encoder = LabelEncoder()
    for col in ordinal:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df


def ai_onehot(df, nominal):
    # One-hot encoding for nominal attributes
    valid_nominal = [col for col in nominal if col in df.columns]
    df = pd.get_dummies(
        df,
        columns=valid_nominal,
        dtype=int
    )
    return df

# Step 1: Clean the data
df = data_clean(df)
# Step 2: Label encode ordinal attributes
df = ai_label(df, ordinal)
# Step 3: One-hot encode nominal attributes
df = ai_onehot(df, nominal)
# Step 4: Print resultant dataset
print("\nResultant Dataset:")
print(df)