# AI Tool Used: ChatGPT
# Lab 6 - AI-generated data splitting function
# Test: NO CONDITION    750, CONDITION       600

from sklearn.model_selection import train_test_split
import pandas as pd

def ai_data_split(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=42,
        stratify=Y
    )
    return X_train, X_test, y_train, y_test

def main():
    p=pd.read_excel('thyroid_dataset.xlsx')
    X=p.drop(columns=['Condition'])
    Y=p['Condition']
    X_train, X_test, y_train, y_test = ai_data_split(X, Y)
    print(y_train.value_counts())
    print(y_test.value_counts())

if __name__ == "__main__":
    main()