import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Cấu hình đường dẫn
DATA_PATH = "data/online_retail_II.xlsx"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_and_clean_data(filepath: str) -> tuple[pd.DataFrame, dict]:
    print("1. Đang đọc dữ liệu từ file Excel...")
    df1 = pd.read_excel(filepath, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(filepath, sheet_name="Year 2010-2011")
    df_raw = pd.concat([df1, df2], ignore_index=True)
    
    stats = {
        "raw_rows": len(df_raw),
        "raw_invoices": df_raw["Invoice"].nunique(),
    }

    print("2. Đang làm sạch dữ liệu...")
    df = df_raw.dropna(subset=["Description"]).copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].str.strip()

    # Loại bỏ đơn hủy (bắt đầu bằng C) và số lượng/đơn giá <= 0
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Loại bỏ các mã phí dịch vụ không phải hàng hóa thực tế
    ignore_codes = {
        "POST", "D", "M", "BANK CHARGES", "AMAZONFEE", 
        "DOT", "CRUK", "ADJUST", "ADJUST2", "S", "B"
    }
    df = df[~df["StockCode"].isin(ignore_codes)]

    # Lọc riêng thị trường United Kingdom (chiếm >90% dữ liệu) để hành vi mua sắm đồng nhất
    df = df[df["Country"] == "United Kingdom"]

    # Lọc các hóa đơn có ít nhất 2 loại sản phẩm (đơn 1 món không tạo ra luật kết hợp)
    invoice_item_counts = df.groupby("Invoice")["Description"].nunique()
    valid_invoices = invoice_item_counts[invoice_item_counts >= 2].index
    df_clean = df[df["Invoice"].isin(valid_invoices)].copy()

    stats["clean_rows"] = len(df_clean)
    stats["clean_invoices"] = df_clean["Invoice"].nunique()
    stats["unique_items"] = df_clean["Description"].nunique()
    stats["avg_basket_size"] = round(invoice_item_counts[valid_invoices].mean(), 2)

    return df_clean, stats

def run_fpgrowth(df_clean: pd.DataFrame, min_support: float = 0.012, min_lift: float = 1.2):
    print("3. Đang gom nhóm giỏ hàng & mã hóa One-Hot (Sparse)...")
    transactions = df_clean.groupby("Invoice")["Description"].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    basket_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    print(f"4. Đang chạy thuật toán FP-Growth (min_support={min_support})...")
    frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    print(f"5. Đang sinh luật kết hợp (min_lift={min_lift})...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    
    # Chuyển frozenset thành list/chuỗi để lưu trữ và truy vấn dễ dàng
    rules["antecedents_list"] = rules["antecedents"].apply(lambda x: sorted(list(x)))
    rules["consequents_list"] = rules["consequents"].apply(lambda x: sorted(list(x)))
    rules["antecedents_str"] = rules["antecedents_list"].apply(lambda x: " + ".join(x))
    rules["consequents_str"] = rules["consequents_list"].apply(lambda x: " + ".join(x))

    # Chỉ giữ lại các cột cần thiết cho Web App
    cols_to_keep = [
        "antecedents_list", "consequents_list",
        "antecedents_str", "consequents_str",
        "antecedent support", "consequent support",
        "support", "confidence", "lift", "conviction"
    ]
    rules_export = rules[cols_to_keep].sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
    return rules_export, frequent_itemsets

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{DATA_PATH}'. Hãy copy file online_retail_II.xlsx vào thư mục data/.")

    df_clean, stats = load_and_clean_data(DATA_PATH)
    
    # Lưu thống kê EDA & Top sản phẩm phổ biến
    top_items = (
        df_clean.groupby("Description")
        .agg(
            invoice_count=("Invoice", "nunique"),
            total_qty=("Quantity", "sum"),
            avg_price=("Price", "mean")
        )
        .sort_values(by="invoice_count", ascending=False)
        .reset_index()
    )
    top_items["support"] = top_items["invoice_count"] / stats["clean_invoices"]
    top_items["avg_price"] = top_items["avg_price"].round(2)

    # Phân bổ kích thước giỏ hàng
    basket_dist = (
        df_clean.groupby("Invoice")["Description"]
        .nunique()
        .value_counts()
        .sort_index()
        .reset_index()
    )
    basket_dist.columns = ["basket_size", "invoice_count"]

    # Chạy FP-Growth với ngưỡng support thấp (0.01) để Web App có thể lọc linh hoạt bằng thanh trượt
    rules_df, _ = run_fpgrowth(df_clean, min_support=0.01, min_lift=1.0)

    # Xuất ra định dạng Parquet để Streamlit load cực nhanh
    rules_df.to_parquet(os.path.join(ARTIFACTS_DIR, "rules.parquet"), index=False)
    top_items.to_parquet(os.path.join(ARTIFACTS_DIR, "top_items.parquet"), index=False)
    basket_dist.to_parquet(os.path.join(ARTIFACTS_DIR, "basket_dist.parquet"), index=False)
    pd.DataFrame([stats]).to_parquet(os.path.join(ARTIFACTS_DIR, "stats.parquet"), index=False)

    print(f"Hoàn tất! Đã xuất {len(rules_df)} luật kết hợp vào thư mục '{ARTIFACTS_DIR}/'.")

if __name__ == "__main__":
    main()
