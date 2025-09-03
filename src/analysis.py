
"""analysis.py
Demo pandas project for retail sales analytics.
Functions:
 - load data (sample CSV by default, can download full UCI dataset if requested)
 - cleaning and feature engineering
 - compute KPIs (revenue, orders, AOV, repeat rate, top products)
 - time-series aggregation and simple forecasting baseline
 - cohort analysis (monthly acquisition cohorts)
 - saves PNG visualizations into output/
Usage:
    python src/analysis.py --input ../data/sample_online_retail.csv
If you have internet and want the full dataset from UCI, uncomment download block and provide URL.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, sys

INPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_online_retail.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path=INPUT):
    df = pd.read_csv(path, parse_dates=['InvoiceDate'])
    return df

def clean_data(df):
    # drop rows without CustomerID
    df = df.dropna(subset=['CustomerID'])
    # convert types
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(float)
    # remove cancelled (negative) quantities for core analysis, but keep a flag
    df['IsReturn'] = df['Quantity'] < 0
    df = df[df['Quantity'] != 0]
    # create revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    # add date parts
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDay'] = df['InvoiceDate'].dt.date
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    return df

def kpis(df):
    # Revenue, Orders, AOV, Unique customers
    revenue = df['Revenue'].sum()
    orders = df['InvoiceNo'].nunique()
    customers = df['CustomerID'].nunique()
    aov = revenue / orders if orders else 0
    avg_items_per_order = df.groupby('InvoiceNo')['Quantity'].sum().mean()
    top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
    k = dict(revenue=revenue, orders=orders, customers=customers, aov=aov, avg_items_per_order=avg_items_per_order, top_products=top_products)
    return k


# noinspection PyBroadException
def time_series(df):
    ts = df.groupby('InvoiceMonth')['Revenue'].sum().rename('Revenue').reset_index()
    ts = ts.sort_values('InvoiceMonth')
    # simple rolling mean for baseline forecasting
    ts['Rolling_3M'] = ts['Revenue'].rolling(3, min_periods=1).mean()
    # simple linear trend forecast (OLS on time index)
    ts = ts.copy()
    ts['t'] = np.arange(len(ts))
    X = np.vstack([np.ones(len(ts)), ts['t']]).T
    y = ts['Revenue'].values
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ts['TrendForecast'] = coef[0] + coef[1]*ts['t']
    except Exception as e:
        ts['TrendForecast'] = np.nan
    return ts

def cohort_analysis(df):
    # cohort by first purchase month of customer
    df['FirstMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M').dt.to_timestamp()
    cohort = df.groupby(['FirstMonth','InvoiceMonth']).agg(customers=('CustomerID','nunique'), revenue=('Revenue','sum')).reset_index()
    # compute retention: customers per cohort month / cohort size
    cohort_pivot = cohort.pivot_table(index='FirstMonth', columns='InvoiceMonth', values='customers', fill_value=0)
    cohort_size = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_size, axis=0).round(3)
    return cohort_pivot, retention

def plots(ts, kpis_dict, df):
    # Revenue timeseries
    plt.figure(figsize=(8,4))
    plt.plot(ts['InvoiceMonth'], ts['Revenue'], marker='o')
    plt.plot(ts['InvoiceMonth'], ts['Rolling_3M'], linestyle='--', marker='x')
    plt.title('Monthly Revenue and 3-month rolling mean')
    plt.xlabel('Month'); plt.ylabel('Revenue')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'monthly_revenue.png')); plt.close()

    # Top products bar
    top = kpis_dict['top_products']
    plt.figure(figsize=(8,4))
    top.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title('Top products by revenue (sample)')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'top_products.png')); plt.close()

    # Cohort heatmap (customers) - simple visualization using imshow
    cohort_pivot, retention = cohort_analysis(df)
    plt.figure(figsize=(6,4))
    arr = retention.fillna(0).values
    plt.imshow(arr, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Retention rate')
    plt.xticks(range(arr.shape[1]), [str(x.date()) for x in retention.columns], rotation=45, fontsize=8)
    plt.yticks(range(arr.shape[0]), [str(x.date()) for x in retention.index], fontsize=8)
    plt.title('Cohort retention (sample)')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'cohort_retention.png')); plt.close()

def save_kpis(k):
    with open(os.path.join(OUTPUT_DIR,'kpis.txt'),'w',encoding='utf-8') as f:
        f.write(f"""KPIs (sample dataset)
Revenue: {k['revenue']:.2f}
Orders: {k['orders']}
Customers: {k['customers']}
AOV: {k['aov']:.2f}
Avg items per order: {k['avg_items_per_order']:.2f}

Top products (by revenue):
{k['top_products'].to_string()}
""")

def main():
    df = load_data()
    df = clean_data(df)
    k = kpis(df)
    ts = time_series(df)
    plots(ts, k, df)
    save_kpis(k)
    print('Analysis finished. Outputs in output/ folder.')

if __name__ == '__main__':
    main()
