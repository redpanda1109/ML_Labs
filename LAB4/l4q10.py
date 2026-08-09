# AI tool used: ChatGPT for function creation to plot the histogram
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')

def histogram(df):
    income = df["Income"].dropna()
    bins = [0, 25000, 50000, 75000, 100000, 125000, 150000, 200000]

    plt.figure(figsize=(10, 6))
    plt.hist(
        income,
        bins=bins,
        density=True,
        edgecolor="black"
    )
    plt.xlabel("Income")
    plt.ylabel("Density")
    plt.title("Density Distribution of Income")
    plt.xticks(bins, rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.show()


histogram(df)