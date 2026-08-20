import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("minna_solar_data.csv", index_col=0, parse_dates=True)

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["ALLSKY_SFC_SW_DWN"], linewidth=0.7)
plt.title("Daily Solar Irradiance in Minna (2015–2024)")
plt.xlabel("Date")
plt.ylabel("ALLSKY_SFC_SW_DWN (kWh/m²/day)")
plt.grid(alpha=0.3)
plt.show()