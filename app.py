import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hệ thống Gợi ý Sản phẩm FP-Growth",
    page_icon="🛒",
    layout="wide"
)

ARTIFACTS_DIR = "artifacts"

@st.cache_data
def load_artifacts():
    rules = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "rules.parquet"))
    top_items = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "top_items.parquet"))
    basket_dist = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "basket_dist.parquet"))
    stats = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "stats.parquet")).iloc[0].to_dict()
    return rules, top_items, basket_dist, stats

# Kiểm tra dữ liệu đã huấn luyện chưa
if not os.path.exists(os.path.join(ARTIFACTS_DIR, "rules.parquet")):
    st.error("⚠️ Chưa tìm thấy dữ liệu mô hình trong thư mục `artifacts/`. Vui lòng chạy `python train_model.py` trước!")
    st.stop()

rules_df, top_items_df, basket_dist_df, stats = load_artifacts()

# Khởi tạo session_state cho giỏ hàng mô phỏng
if "cart" not in st.session_state:
    st.session_state.cart = ["REGENCY CAKESTAND 3 TIER"]

def add_to_cart(item_name: str):
    if item_name not in st.session_state.cart:
        st.session_state.cart.append(item_name)

def remove_from_cart(item_name: str):
    if item_name in st.session_state.cart:
        st.session_state.cart.remove(item_name)

# --- SIDEBAR: BỘ LỌC THAM SỐ FP-GROWTH ---
st.sidebar.header("⚙️ Cấu hình Luật Kết Hợp")
st.sidebar.caption("Điều chỉnh ngưỡng lọc luật sinh ra từ thuật toán FP-Growth:")

min_sup = st.sidebar.slider(
    "Min Support (Độ phổ biến tối thiểu)",
    min_value=0.010, max_value=0.050, value=0.012, step=0.002, format="%.3f"
)
min_conf = st.sidebar.slider(
    "Min Confidence (Độ tin cậy tối thiểu)",
    min_value=0.10, max_value=0.90, value=0.35, step=0.05, format="%.2f"
)
min_lift = st.sidebar.slider(
    "Min Lift (Độ nâng tối thiểu)",
    min_value=1.0, max_value=20.0, value=2.0, step=0.5, format="%.1f"
)

# Lọc tập luật theo thanh trượt
filtered_rules = rules_df[
    (rules_df["support"] >= min_sup) &
    (rules_df["confidence"] >= min_conf) &
    (rules_df["lift"] >= min_lift)
].copy()

st.sidebar.divider()
st.sidebar.metric("Số luật thỏa mãn điều kiện", f"{len(filtered_rules):,}")

# --- TIÊU ĐỀ CHÍNH ---
st.title("🛒 Hệ thống Gợi ý Sản phẩm Bán lẻ (FP-Growth)")
st.markdown("Ứng dụng khai phá luật kết hợp trên tập dữ liệu **Online Retail II** để gợi ý sản phẩm mua kèm theo thời gian thực.")

tab1, tab2, tab3 = st.tabs([
    "🛍️ 1. Mô phỏng Giỏ hàng & Gợi ý",
    "📊 2. Khám phá Luật Kết hợp",
    "🧹 3. Thống kê Dữ liệu & Tiền xử lý"
])

# ==============================================================================
# TAB 1: MÔ PHỎNG GIỎ HÀNG & GỢI Ý SẢN PHẨM
# ==============================================================================
with tab1:
    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        st.subheader("🧺 Giỏ hàng hiện tại của bạn")
        all_products = top_items_df["Description"].tolist()
        
        selected_product = st.selectbox(
            "Tìm kiếm & chọn sản phẩm để thêm vào giỏ:",
            options=all_products,
            index=0
        )
        if st.button("➕ Thêm vào giỏ hàng", type="primary"):
            add_to_cart(selected_product)
            st.rerun()

        st.divider()
        if not st.session_state.cart:
            st.info("Giỏ hàng đang trống. Hãy chọn ít nhất 1 sản phẩm ở trên.")
        else:
            for idx, item in enumerate(st.session_state.cart):
                item_info = top_items_df[top_items_df["Description"] == item]
                price_str = f"£{item_info['avg_price'].values[0]:.2f}" if not item_info.empty else ""
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{idx+1}. {item}** `{price_str}`")
                if c2.button("Xóa", key=f"del_{item}"):
                    remove_from_cart(item)
                    st.rerun()

            if st.button("🗑️ Xóa toàn bộ giỏ hàng"):
                st.session_state.cart = []
                st.rerun()

    with col_right:
        st.subheader("✨ Thường được mua cùng (Gợi ý từ FP-Growth)")
        cart_set = set(st.session_state.cart)

        if not cart_set:
            st.caption("Thêm sản phẩm vào giỏ để nhận gợi ý cá nhân hóa.")
        else:
            # Tìm các luật mà vế trái (antecedents) là tập con của giỏ hàng hiện tại
            # và vế phải (consequents) có chứa ít nhất 1 món CHƯA có trong giỏ
            matched_rows = []
            for _, row in filtered_rules.iterrows():
                ant = set(row["antecedents_list"])
                cons = set(row["consequents_list"])
                if ant.issubset(cart_set):
                    new_items = cons - cart_set
                    if new_items:
                        for new_item in new_items:
                            matched_rows.append({
                                "item": new_item,
                                "based_on": row["antecedents_str"],
                                "confidence": row["confidence"],
                                "lift": row["lift"],
                                "support": row["support"]
                            })

            if matched_rows:
                rec_df = pd.DataFrame(matched_rows)
                # Nếu 1 món được gợi ý bởi nhiều luật, ưu tiên luật có Lift cao nhất
                rec_df = (
                    rec_df.sort_values(by=["lift", "confidence"], ascending=False)
                    .drop_duplicates(subset=["item"])
                    .head(8)
                )

                for _, rec in rec_df.iterrows():
                    with st.container(border=True):
                        r1, r2 = st.columns([3.5, 1])
                        price_info = top_items_df[top_items_df["Description"] == rec["item"]]
                        price_tag = f" • Giá TB: £{price_info['avg_price'].values[0]:.2f}" if not price_info.empty else ""
                        
                        r1.markdown(f"**{rec['item']}**{price_tag}")
                        r1.caption(
                            f"💡 Vì bạn đã chọn *{rec['based_on']}* "
                            f"(Độ tin cậy: **{rec['confidence']*100:.1f}%** | Độ nâng Lift: **{rec['lift']:.2f}x**)"
                        )
                        if r2.button("➕ Thêm", key=f"add_rec_{rec['item']}"):
                            add_to_cart(rec["item"])
                            st.rerun()
            else:
                st.warning(
                    "Không tìm thấy luật kết hợp nào khớp với giỏ hàng ở ngưỡng hiện tại. "
                    "Bạn có thể giảm **Min Confidence / Min Lift** ở thanh bên trái, hoặc tham khảo các sản phẩm bán chạy dưới đây:"
                )
                fallback_df = top_items_df[~top_items_df["Description"].isin(cart_set)].head(5)
                for _, fb in fallback_df.iterrows():
                    with st.container(border=True):
                        f1, f2 = st.columns([3.5, 1])
                        f1.markdown(f"**{fb['Description']}** • Giá TB: £{fb['avg_price']:.2f}")
                        f1.caption(f"🔥 Sản phẩm phổ biến (Xuất hiện trong {fb['invoice_count']:,} đơn hàng)")
                        if f2.button("➕ Thêm", key=f"fb_{fb['Description']}"):
                            add_to_cart(fb["Description"])
                            st.rerun()

# ==============================================================================
# TAB 2: KHÁM PHÁ LUẬT KẾT HỢP (DATA MINING VIEW)
# ==============================================================================
with tab2:
    st.subheader("📈 Trực quan hóa Tập luật Kết hợp")
    if filtered_rules.empty:
        st.warning("Không có luật nào thỏa mãn bộ lọc hiện tại. Hãy hạ thấp các ngưỡng ở Sidebar.")
    else:
        fig_scatter = px.scatter(
            filtered_rules.head(500),
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_data=["antecedents_str", "consequents_str", "lift"],
            color_continuous_scale="Viridis",
            title="Phân bố Support vs Confidence (Kích thước & Màu sắc biểu diễn Lift - Top 500 luật)",
            labels={
                "support": "Support (Độ phổ biến)",
                "confidence": "Confidence (Độ tin cậy)",
                "lift": "Lift (Độ nâng)"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📋 Bảng chi tiết các Luật Kết hợp")
        display_rules = filtered_rules[[
            "antecedents_str", "consequents_str", "support", "confidence", "lift"
        ]].rename(columns={
            "antecedents_str": "Nếu mua (Antecedents)",
            "consequents_str": "Thì sẽ mua (Consequents)",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift"
        })
        st.dataframe(
            display_rules.style.format({
                "Support": "{:.4f}",
                "Confidence": "{:.2%}",
                "Lift": "{:.2f}"
            }),
            use_container_width=True,
            height=400
        )

        csv_data = display_rules.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Tải xuống danh sách luật (CSV)",
            data=csv_data,
            file_name="fpgrowth_association_rules.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 3: THỐNG KÊ DỮ LIỆU & TIỀN XỬ LÝ (EDA)
# ==============================================================================
with tab3:
    st.subheader("🧹 Hiệu quả Làm sạch Dữ liệu")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng dữ liệu gốc", f"{int(stats['raw_rows']):,}")
    m2.metric("Số dòng sạch (UK, >=2 món)", f"{int(stats['clean_rows']):,}")
    m3.metric("Số hóa đơn hợp lệ", f"{int(stats['clean_invoices']):,}")
    m4.metric("Số mặt hàng duy nhất", f"{int(stats['unique_items']):,}")

    st.divider()
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        fig_top = px.bar(
            top_items_df.head(15).sort_values("invoice_count", ascending=True),
            x="invoice_count",
            y="Description",
            orientation="h",
            title="Top 15 Sản phẩm xuất hiện trong nhiều hóa đơn nhất",
            labels={"invoice_count": "Số lượng hóa đơn", "Description": "Tên sản phẩm"},
            color="invoice_count",
            color_continuous_scale="Blues"
        )
        fig_top.update_layout(showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with c_chart2:
        fig_basket = px.bar(
            basket_dist_df[basket_dist_df["basket_size"] <= 40],
            x="basket_size",
            y="invoice_count",
            title=f"Phân bổ kích thước giỏ hàng (Trung bình: {stats['avg_basket_size']} món/đơn)",
            labels={"basket_size": "Số loại sản phẩm trong 1 hóa đơn", "invoice_count": "Số lượng hóa đơn"}
        )
        st.plotly_chart(fig_basket, use_container_width=True)
