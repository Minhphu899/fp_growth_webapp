import os
import streamlit as st
import pandas as pd

# 1. Cấu hình trang
st.set_page_config(
    page_title="UK Retail Store - Gợi ý mua sắm thông minh",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS tùy chỉnh cho phong cách E-commerce
st.markdown("""
<style>
    /* Header Banner */
    .store-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .store-banner h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .store-banner p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    /* Nhãn giá */
    .price-tag {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e11d48;
    }
    .product-title {
        font-weight: 600;
        font-size: 0.95rem;
        min-height: 44px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-top: 2px;
        margin-bottom: 4px;
    }
    .total-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = "artifacts"


@st.cache_data
def load_artifacts():
    rules = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "rules.parquet"))
    top_items = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "top_items.parquet"))
    return rules, top_items


if not os.path.exists(os.path.join(ARTIFACTS_DIR, "rules.parquet")):
    st.error("⚠️ Chưa tìm thấy dữ liệu mô hình trong thư mục `artifacts/`. Vui lòng chạy `python train_model.py` trước!")
    st.stop()

rules_df, top_items_df = load_artifacts()

# Khởi tạo session_state cho giỏ hàng
if "cart" not in st.session_state:
    st.session_state.cart = ["REGENCY CAKESTAND 3 TIER"]


def add_to_cart(item_name: str):
    if item_name not in st.session_state.cart:
        st.session_state.cart.append(item_name)
        st.toast(f"Đã thêm **{item_name}** vào giỏ hàng!", icon="🛒")


def remove_from_cart(item_name: str):
    if item_name in st.session_state.cart:
        st.session_state.cart.remove(item_name)


def get_item_price(item_name: str) -> float:
    info = top_items_df[top_items_df["Description"] == item_name]
    return float(info["avg_price"].values[0]) if not info.empty else 0.0


# --- SIDEBAR: CẤU HÌNH THUẬT TOÁN ---
st.sidebar.header("⚙️ Cấu hình Thuật toán FP-Growth")
st.sidebar.caption("Điều chỉnh ngưỡng lọc luật kết hợp:")

min_sup = st.sidebar.slider(
    "Min Support (Độ phổ biến)",
    min_value=0.010, max_value=0.050, value=0.010, step=0.002, format="%.3f"
)
min_conf = st.sidebar.slider(
    "Min Confidence (Độ tin cậy)",
    min_value=0.10, max_value=0.90, value=0.60, step=0.05, format="%.2f"
)
min_lift = st.sidebar.slider(
    "Min Lift (Độ nâng)",
    min_value=1.0, max_value=20.0, value=1.2, step=0.1, format="%.1f"
)

filtered_rules = rules_df[
    (rules_df["support"] >= min_sup) &
    (rules_df["confidence"] >= min_conf) &
    (rules_df["lift"] >= min_lift)
]

# --- TOP BANNER CỬA HÀNG ---
st.markdown("""
<div class="store-banner">
    <h1>🛍️ BRITISH HOME & GIFT STORE</h1>
    <p>Hệ thống bán lẻ trực tuyến tích hợp gợi ý mua kèm thông minh theo thời gian thực (FP-Growth Engine)</p>
</div>
""", unsafe_allow_html=True)

# --- BỐ CỤC CHÍNH: GIAN HÀNG (TRÁI) & GIỎ HÀNG CHECKOUT (PHẢI) ---
main_col, cart_col = st.columns([2.2, 1], gap="large")

# ==============================================================================
# CỘT TRÁI: TÌM KIẾM, GỢI Ý CÁ NHÂN HÓA & DANH MỤC SẢN PHẨM
# ==============================================================================
with main_col:
    # 1. Thanh tìm kiếm nhanh sản phẩm
    search_col1, search_col2 = st.columns([4, 1], vertical_alignment="bottom")
    with search_col1:
        all_products = top_items_df["Description"].tolist()
        selected_product = st.selectbox(
            "🔍 Tìm kiếm nhanh trong danh mục sản phẩm:",
            options=all_products,
            index=0
        )
    with search_col2:
        if st.button("➕ Thêm nhanh", type="primary", use_container_width=True):
            add_to_cart(selected_product)
            st.rerun()

    st.markdown("---")

    # 2. KHU VỰC GỢI Ý CÁ NHÂN HÓA (Để trống hoàn toàn nếu không có gợi ý nào khớp)
    st.subheader("✨ Gợi ý mua kèm dành riêng cho giỏ hàng của bạn")
    cart_set = set(st.session_state.cart)
    matched_rows = []
    if cart_set:
        for _, row in filtered_rules.iterrows():
            ant = set(row["antecedents_list"])
            cons = set(row["consequents_list"])
            if ant.issubset(cart_set):
                new_items = cons - cart_set
                for new_item in new_items:
                    matched_rows.append({
                        "item": new_item,
                        "based_on": row["antecedents_str"],
                        "confidence": row["confidence"],
                        "lift": row["lift"],
                        "support": row["support"]
                    })

    if matched_rows:
        rec_df = (
            pd.DataFrame(matched_rows)
            .sort_values(by=["lift", "confidence"], ascending=False)
            .drop_duplicates(subset=["item"])
            .head(6)
        )

        rec_cols = st.columns(3)
        for idx, (_, rec) in enumerate(rec_df.iterrows()):
            with rec_cols[idx % 3]:
                with st.container(border=True):
                    price = get_item_price(rec["item"])
                    st.markdown(f'<div class="product-title" title="{rec["item"]}">{rec["item"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="price-tag">£{price:.2f}</div>', unsafe_allow_html=True)
                    st.caption(f"Mua kèm với: *{rec['based_on'][:28]}...*" if len(rec['based_on']) > 28 else f"Mua kèm với: *{rec['based_on']}*")
                    if st.button("🛒 Thêm vào giỏ", key=f"rec_{rec['item']}", use_container_width=True):
                        add_to_cart(rec["item"])
                        st.rerun()

    st.markdown("---")

    # 3. KHU VỰC SẢN PHẨM GỢI Ý (Lấy từ các luật kết hợp mạnh nhất)
    st.subheader("🛍️ Sản phẩm gợi ý")

    source_rules = filtered_rules if not filtered_rules.empty else rules_df
    strong_rules = source_rules.sort_values(by=["lift", "confidence"], ascending=False)

    strong_items = []
    seen_items = set(cart_set)
    for _, row in strong_rules.iterrows():
        for item in list(row["antecedents_list"]) + list(row["consequents_list"]):
            if item not in seen_items:
                strong_items.append(item)
                seen_items.add(item)
            if len(strong_items) >= 6:
                break
        if len(strong_items) >= 6:
            break

    # Bổ sung nếu số sản phẩm từ luật ít hơn 6
    if len(strong_items) < 6:
        for desc in top_items_df["Description"].tolist():
            if desc not in seen_items:
                strong_items.append(desc)
                seen_items.add(desc)
            if len(strong_items) >= 6:
                break

    cat_cols = st.columns(3)
    for idx, item_name in enumerate(strong_items):
        with cat_cols[idx % 3]:
            with st.container(border=True):
                price = get_item_price(item_name)
                st.markdown(f'<div class="product-title" title="{item_name}">{item_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-tag">£{price:.2f}</div>', unsafe_allow_html=True)
                st.caption("Sẵn hàng • Giao tiêu chuẩn UK")
                if st.button("➕ Chọn mua", key=f"cat_{item_name}", use_container_width=True):
                    add_to_cart(item_name)
                    st.rerun()

# ==============================================================================
# CỘT PHẢI: GIỎ HÀNG & THANH TOÁN
# ==============================================================================
with cart_col:
    with st.container(border=True):
        st.subheader(f"🛒 Giỏ hàng ({len(st.session_state.cart)})")

        if not st.session_state.cart:
            st.info("Giỏ hàng của bạn đang trống. Hãy chọn sản phẩm bên trái để bắt đầu mua sắm!")
        else:
            total_amount = 0.0
            for idx, item in enumerate(st.session_state.cart):
                price = get_item_price(item)
                total_amount += price

                item_c1, item_c2 = st.columns([4, 1])
                with item_c1:
                    st.markdown(f"**{item}**")
                    st.caption(f"Đơn giá: £{price:.2f} × 1")
                with item_c2:
                    if st.button("✕", key=f"del_{item}", help="Xóa khỏi giỏ"):
                        remove_from_cart(item)
                        st.rerun()
                st.divider()

            # Bảng tổng kết đơn hàng
            shipping_fee = 0.0 if total_amount >= 20.0 else 3.50
            final_total = total_amount + shipping_fee

            st.markdown(f"""
            <div class="total-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span>Tạm tính ({len(st.session_state.cart)} món):</span>
                    <strong>£{total_amount:.2f}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.9rem; color:#475569;">
                    <span>Phí vận chuyển (Miễn phí từ £20):</span>
                    <span>{"Miễn phí" if shipping_fee == 0 else f"£{shipping_fee:.2f}"}</span>
                </div>
                <hr style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:1.15rem; color:#e11d48;">
                    <strong>Tổng cộng:</strong>
                    <strong>£{final_total:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            if st.button("💳 Tiến hành đặt hàng", type="primary", use_container_width=True):
                st.balloons()
                st.success("🎉 Đặt hàng thành công! Cảm ơn bạn đã mua sắm.")

            if st.button("🗑️ Làm trống giỏ hàng", use_container_width=True):
                st.session_state.cart = []
                st.rerun()