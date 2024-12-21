import pandas as pd

# Вхідні дані
data = [
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "High", "Strong", "No"],
    ["Overcast", "High", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Rain", "Normal", "Weak", "Yes"],
    ["Rain", "Normal", "Strong", "No"],
    ["Overcast", "Normal", "Strong", "Yes"],
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Sunny", "Normal", "Strong", "Yes"],
    ["Overcast", "High", "Strong", "Yes"],
    ["Overcast", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Strong", "No"]
]

columns = ["Outlook", "Humidity", "Wind", "Play"]
df = pd.DataFrame(data, columns=columns)

# Частотна таблиця
frequency_table = df.groupby(["Outlook", "Play"]).size().unstack()
print("Частотна таблиця:")
print(frequency_table)

# Ймовірності
total_yes = len(df[df["Play"] == "Yes"])
total_no = len(df[df["Play"] == "No"])
total = len(df)

prob_yes = total_yes / total
prob_no = total_no / total

print(f"\nЙмовірність 'Yes': {prob_yes}")
print(f"Ймовірність 'No': {prob_no}")
