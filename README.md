# 🛒 E-commerce Data Analytics & Forecasting Project

## 📘 1. Project Overview

**Objective:**  
Analyze an e-commerce dataset to uncover insights, optimize profits, and forecast future trends.

**Tech Stack:**  
- 🐍 **Languages:** Python  
- 📚 **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Prophet  
- 💻 **Tools:** Jupyter Notebook / VS Code  

**Dataset:**  
Contains order-level data with:
> Sales, Profit, Quantity, Discounts, Product details, Customer info, and Timestamps.

---

## ⚙️ 2. Data Import & Initial Overview

```python
import pandas as pd

# Load dataset
df = pd.read_csv("E-commerce Dataset.csv", encoding="UTF-8")

# Basic overview
print(df.head())
print(df.info())
print(df.columns)
print(df.shape)
print(df.describe())
```

**🧠 Key Insight:** Initial exploration helps understand dataset structure, completeness, and numeric ranges.

---

## 🧹 3. Data Cleaning

```python
# Missing values
df.isnull().sum()

# Handle missing values & duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Data type correction
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df = df.convert_dtypes()

# Remove outliers (example)
df = df[df['Profit'] < df['Profit'].quantile(0.99)]

# Save cleaned data
df.to_csv("Cleaned_Data.csv", index=False)
```

**🧠 Key Insight:** Clean data ensures accurate KPI calculation and reliable forecasts.

---

## 🧩 4. Feature Engineering

```python
# Extract date-based features
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Weekday'] = df['Order_Date'].dt.day_name()

# Profit Margin
df['Profit_Margin'] = df['Profit'] / df['Sales']
```

**🧠 Key Insight:** Derived features enable deeper analysis and improve forecasting accuracy.

---

## 📊 5. KPI Calculation

```python
# Key Metrics
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
avg_margin = df['Profit_Margin'].mean()
unique_orders = df['Order_ID'].nunique()
unique_customers = df['Customer_ID'].nunique()

# Top Customers & Products
top_customers = df.groupby('Customer_ID')['Sales'].sum().sort_values(ascending=False).head(10)
top_products = df.groupby('Product_Name')['Sales'].sum().sort_values(ascending=False).head(10)
```

**🧠 Key Insight:** KPIs summarize overall business performance and profitability at a glance.

---

## 🔍 6. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Category-wise analysis
sns.barplot(x='Category', y='Sales', data=df)
plt.title("Category-wise Sales")

# Monthly Trends
monthly = df.groupby('Month')['Sales'].sum()
monthly.plot(kind='line', title='Monthly Sales Trend')

# Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

**🧠 Key Insight:** Visual analysis reveals top-performing categories, regions, and seasonal patterns.

---

## 📈 7. Forecasting Next 6 Months

```python
from prophet import Prophet

# Prepare data for Prophet
forecast_df = df.groupby('Order_Date')['Sales'].sum().reset_index()
forecast_df.columns = ['ds', 'y']

# Train model
model = Prophet()
model.fit(forecast_df)

# Forecast next 6 months
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("🕒 6-Month Sales Forecast")
plt.show()
```

**🧠 Key Insight:** Forecasting helps in inventory planning, marketing strategy, and budgeting.

---

## 💡 8. Recommendations

| Area | Action Plan |
|------|--------------|
| **Profit Improvement** | 🔸 Reduce discounts on low-margin products<br>🔸 Upsell high-margin categories |
| **Customer Strategy** | 🎯 Loyalty programs for top customers<br>🎯 Targeted promotions for low-profit segments |
| **Marketing & Logistics** | 🚚 Seasonal campaigns during peak months<br>🚚 Optimize shipping for low-profit regions |
| **Forecast Action** | 📦 Stock up for forecasted high-demand months<br>📉 Adjust budget for low-profit periods |

---

## 🏆 9. Project Highlights

✅ Cleaned, feature-engineered, and analyzed real-world e-commerce data  
✅ Delivered KPIs, visual insights, and performance metrics  
✅ Forecasted 6-month future sales & profit using Prophet  
✅ Provided actionable business recommendations  

---

## 📁 Folder Structure

```
📂 E-commerce-Analytics-Forecasting
 ┣ 📜 E-commerce Dataset.csv
 ┣ 📜 Cleaned_Data.csv
 ┣ 📓 E-commerce_Analysis.ipynb
 ┣ 📜 README.md
 ┗ 📁 /plots
     ┣ category_sales.png
     ┣ profit_trends.png
     ┣ forecast.png
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-analytics.git
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn prophet
   ```

3. **Run the notebook or script**
   ```bash
   jupyter notebook E-commerce_Analysis.ipynb
   ```

4. **View visualizations and forecast results 🎉**

---

**⭐ If you found this project helpful, don’t forget to star the repo!**
