import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt   
import seaborn as sns
from prophet import Prophet  
import os
def clear():
    os.system("cls"if os.name=="nt" else "clear")
clear()

a = pd.read_csv("E-commerce Dataset.csv",encoding="UTF-8")

# Data cleaning:-
c = a.drop("Time",axis=1,inplace=True)
b = ["Aging", "Sales", "Quantity", "Discount", "Shipping_Cost", "Order_Priority"]
a = a.dropna(subset=b)
a = a.drop_duplicates()
a["Order_Date"] = pd.to_datetime(a["Order_Date"], format="%Y-%m-%d")
a["Month"] = a["Order_Date"].dt.month
a["Year"] = a["Order_Date"].dt.year
a["Profit_Margin"] = (a["Profit"] / a["Sales"]) * 100
monthly_summary = a.groupby(["Year", "Month"]).agg({
    "Sales": "sum",
    "Profit": "sum",
    "Quantity":"sum"
}).reset_index()

b = pd.read_csv("Monthly_Sales_Profit.csv",encoding="UTF-8")

# Graphs of all monthly sales analysis:-
month = b["Month"].value_counts
sales = b["Sales"].value_counts
profit = b["Profit"].value_counts
sns.set_theme(style="whitegrid")  
palette = sns.color_palette("Set2")

plt.figure(figsize=(10,5))
sns.barplot(x=b["Month"], y=b["Sales"], palette="Blues_d")
plt.title("Sales per Month")
plt.savefig("sales.png")
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x=b["Month"], y=b["Quantity"], marker="o", color="green")
plt.title("Quantity per Month")
plt.savefig("Quantity.png")
plt.show()

plt.figure(figsize=(10,5))
sns.pointplot(x=b["Month"], y=b["Profit"], color="red")
plt.title("Profit per Month")
plt.savefig("Profit.png")
plt.show()

corr = b[["Sales","Profit","Quantity"]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Sales, Profit & Quantity", fontsize=14, weight='bold')
plt.savefig("Corelatinsales.png")
plt.show()

# Future predictions
d = pd.DataFrame({
    "Month":[1,2,3,4,5,6,7,8,9,10,11,12],
    "Sales":[379627,332495,435502,596990,824362,642501,809974,664245,738303,743137,877881,767147]
})

d['ds'] = pd.to_datetime("2018-" + d['Month'].astype(str) + "-01")
d['y'] = d['Sales']

m = Prophet()
m.fit(d[['ds','y']])

# Forecast next 6 months:-
future = m.make_future_dataframe(periods=6, freq='M')
forecast = m.predict(future)

# Plot forecast:-
fig = m.plot(forecast)
plt.title("Sales Forecast Next 6 Months", fontsize=14)
plt.savefig("forecast.png")
plt.show()

monthly_summary.to_csv("Monthly_Sales_Profit.csv", index=False)

