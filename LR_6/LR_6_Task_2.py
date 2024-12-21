# Умовні ймовірності для кожної ознаки
# Для 'Yes'
prob_outlook_overcast_yes = len(df[(df["Outlook"] == "Overcast") & (df["Play"] == "Yes")]) / total_yes
prob_humidity_high_yes = len(df[(df["Humidity"] == "Normal") & (df["Play"] == "Yes")]) / total_yes
prob_wind_strong_yes = len(df[(df["Wind"] == "Strong") & (df["Play"] == "Yes")]) / total_yes

# Для 'No'
prob_outlook_overcast_no = len(df[(df["Outlook"] == "Overcast") & (df["Play"] == "No")]) / total_no
prob_humidity_high_no = len(df[(df["Humidity"] == "High") & (df["Play"] == "No")]) / total_no
prob_wind_strong_no = len(df[(df["Wind"] == "Strong") & (df["Play"] == "No")]) / total_no

# Байєсівські ймовірності для кожного класу
prob_yes_given_conditions = prob_yes * prob_outlook_overcast_yes * prob_humidity_high_yes * prob_wind_strong_yes
prob_no_given_conditions = prob_no * prob_outlook_overcast_no * prob_humidity_high_no * prob_wind_strong_no

print(f"\nЙмовірність 'Yes': {prob_yes_given_conditions}")
print(f"Ймовірність 'No': {prob_no_given_conditions}")

# Порівняння ймовірностей
if prob_yes_given_conditions > prob_no_given_conditions:
    print("\nМатч відбудеться!")
else:
    print("\nМатч не відбудеться!")
