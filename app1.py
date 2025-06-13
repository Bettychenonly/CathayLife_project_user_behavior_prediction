import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import base64
from PIL import Image
import plotly.express as px

# ✨ SessionState 初始化
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "all_columns" not in st.session_state:
    st.session_state.all_columns = []

# 目標行為選項
action_group_options = [
    "Line分享轉傳", "線上繳費", "預約顧問與商品諮詢", "資料填寫與確認",
    "試算: 網投－壽險－試算", "試算: 網投－意外傷害險－試算", "試算: 網投－健康險－試算",
    "試算: 網投－旅平險－試算", "試算: 網投－投資型年金險－試算", "試算: 網投－年金險－試算",
    "試算: 壽險－試算", "試算: 意外傷害險－試算", "試算: 健康醫療險－實支實付－試算",
    "試算: 健康醫療險－重大疾病－試算", "試算: 健康醫療險－長期照顧－試算", "試算: 健康醫療險－住院手術－試算",
    "試算: 其他試算", "試算: 生涯推薦61歲以上－試算", "試算: 生涯推薦45-60－試算",
    "試算: 生涯推薦35-44－試算", "試算: 生涯推薦19-34－試算", "試算: 生涯推薦0-18－試算",
    "填寫預約資料", "商品資訊頁－還本與年金型保險", "商品資訊頁－網投－壽險",
    "商品資訊頁－網投－意外傷害險", "商品資訊頁－網投－健康險", "商品資訊頁－網投－旅平險",
    "商品資訊頁－網投－投資型年金險", "商品資訊頁－網投－年金險", "商品資訊頁－壽險",
    "商品資訊頁－意外傷害險", "商品資訊頁－健康醫療險", "商品資訊頁－旅行平安險",
    "商品資訊頁－投資型保險", "商品資訊頁－主題商品", "訂製保險組合－投資規劃試算結果",
    "訂製保險組合－人身規劃試算結果", "訂製保險組合", "挑選預約顧問",
    "保險視圖、保單明細、資產總覽、保險明細", "保存與分享試算結果", "保存與分享自由配、訂製組合結果",
    "其他", "投保資格確認", "找服務（尋求服務與客服）", "我的保險試算結果", "完成O2O",
    "完成網路投保", "自由配－配置我的基金試算結果", "自由配－套餐", "自由配－保障規劃試算結果",
    "自由配－保障規劃", "自由配－投資規劃", "自由配", "好康優惠", "立即投保",
    "生涯推薦－商品資訊頁", "方案確認", "手機驗證碼"
]

# 變成 base64 的圖片顯示
def get_base64_image(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# 修改為相對路徑
image_path = "image.png"
img_base64 = get_base64_image(image_path)

st.set_page_config(page_title="國泰人壽 - 用戶行為預測工具", layout="centered", initial_sidebar_state="collapsed")

# 配色 + UI CSS 設計
st.markdown("""
<style>
body { background-color: #e9f7f5; }
.stApp { background-color: #e9f7f5; color: #333333; }

input, select, textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
    border-radius: 6px !important;
}

section[data-testid="stDateInput"] {
    background-color: #ffffff !important;
    border-radius: 6px;
    padding: 4px;
}

button[kind="primary"] {
    background-color: #4fc08d !important;
    color: white !important;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
}

[data-testid="stNotification"] {
    background-color: #d2f3ea !important;
    color: #000000 !important;
    border: 1px solid #99d6cc;
}

/* ✅ st.form 整體背景 */
div[data-testid="stForm"] {
    background-color: #f8fffc !important;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #cceee4;
}

/* ✅ st.expander 背景與內框 */
div[data-testid="stExpander"] > div:first-child {
    background-color: #f8fffc !important;
    border-radius: 12px;
    border: 1px solid #cceee4;
    padding: 16px;
}

/* ✅ st.expander 標題放大 */
div[data-testid="stExpander"] > summary {
    font-size: 1.15rem !important;
    font-weight: 600;
    color: #2a6154;
}

label[for="column_selector"] {
    font-size: 1.1rem !important;
    font-weight: 600;
    color: #1f3f3e;
}
</style>
""", unsafe_allow_html=True)

# 圖示 + 標題
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="{img_base64}" alt="logo" width="40" style="margin-right: 10px;">
        <h3 style="margin: 0;">國泰人壽 - 多元訪客進站行為預測</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("#### 用戶行為預測工具 – 分析大量用戶軌跡")

# 目標行為選項
action_group_options = [
    "Line分享轉傳", "線上繳費", "預約顧問與商品諮詢", "資料填寫與確認",
    "試算: 網投－壽險－試算", "試算: 網投－意外傷害險－試算", "試算: 網投－健康險－試算",
    "試算: 網投－旅平險－試算", "試算: 網投－投資型年金險－試算", "試算: 網投－年金險－試算",
    "試算: 壽險－試算", "試算: 意外傷害險－試算", "試算: 健康醫療險－實支實付－試算",
    "試算: 健康醫療險－重大疾病－試算", "試算: 健康醫療險－長期照顧－試算",
    "試算: 健康醫療險－住院手術－試算", "試算: 其他試算", "試算: 生涯推薦61歲以上－試算",
    "試算: 生涯推薦45-60－試算", "試算: 生涯推薦35-44－試算", "試算: 生涯推薦19-34－試算",
    "試算: 生涯推薦0-18－試算", "填寫預約資料", "商品資訊頁－還本與年金型保險",
    "商品資訊頁－網投－壽險", "商品資訊頁－網投－意外傷害險", "商品資訊頁－網投－健康險",
    "商品資訊頁－網投－旅平險", "商品資訊頁－網投－投資型年金險", "商品資訊頁－網投－年金險",
    "商品資訊頁－壽險", "商品資訊頁－意外傷害險", "商品資訊頁－健康醫療險",
    "商品資訊頁－旅行平安險", "商品資訊頁－投資型保險", "商品資訊頁－主題商品",
    "訂製保險組合－投資規劃試算結果", "訂製保險組合－人身規劃試算結果", "訂製保險組合",
    "挑選預約顧問", "保險視圖、保單明細、資產總覽、保險明細", "保存與分享試算結果",
    "保存與分享自由配、訂製組合結果", "其他", "投保資格確認", "找服務（尋求服務與客服）",
    "我的保險試算結果", "完成O2O", "完成網路投保", "自由配－配置我的基金試算結果",
    "自由配－套餐", "自由配－保障規劃試算結果", "自由配－保障規劃", "自由配－投資規劃",
    "自由配", "好康優惠", "立即投保", "生涯推薦－商品資訊頁", "方案確認", "手機驗證碼"
]

# 表單
with st.form("predict_form"):
    st.markdown("#### 預測時間設定")
    
    start_time = st.date_input("開始時間（UTC）", value=datetime.utcnow(), help="分析資料起始日")
    end_time = st.date_input("結束時間（UTC）", value=datetime.utcnow(), help="分析資料結束日")
    
    st.markdown("#### 模型參數與目標行為")
    threshold = st.number_input("模型關值設定", 0.0, 1.0, 0.75, help="預測分數高於此值觀察為有效")

    with st.expander(" 選擇目標行為（可多選）", expanded=True):
        target_action = st.multiselect(
            "請選擇預測的下一步行為：",
            options=action_group_options,
            default=[],
            key="target_action_selector",
            help="可根據常見 user 行為選擇模型欲預測的目標行為（可複選）"
        )

    st.markdown("#### 結果版本命名")
    result_name = st.text_input("分析版本名稱", value=f"預測分析_{datetime.utcnow().strftime('%Y%m%d_%H%M')}", help="方便分析結果儲存與追蹤")
    result_notes = st.text_area(" 備註說明（選填）", placeholder="可輸入分析特性、用戶分稱、策略目標", help="幫助記錄分析或於後續檢討")

    st.markdown("#### 輸出 CSV 格式設定")
    separator = st.selectbox(" 欄位分隔符號", [",", ";", "\t"], format_func=lambda x: f"'{x}' 分隔", help="指定 CSV 各欄之間分隔符")
    include_header = st.checkbox(" 包含欄位標題", value=True, help="輸出時是否包含首行標題")
    encoding = st.selectbox(" 編碼格式", ["utf-8", "utf-8-sig", "big5"], help="選擇您系統支援的編碼，如 Windows 用 big5")

    submit_btn = st.form_submit_button("執行預測分析")

# 執行預測
if submit_btn:
    if not target_action:
        st.error("請輸入目標行為")
    elif end_time < start_time:
        st.warning(" 結束時間不能早於開始時間")
    else:
        with st.spinner(" 預測中，請稍候..."):
            actions = ["view", "click", "add_to_cart", "purchase", "search", "scroll"]
            groups = ["product", "checkout", "homepage", "support", "promotion"]
            result_list = []
            for i in range(1, 31):
                row = {"user_pseudo_id": f"user_{i}"}
                for step in range(-9, 1):
                    row[f" 第n{step}步 action"] = random.choice(actions)
                    row[f" 第n{step}步 action_group"] = random.choice(groups)
                row["last_platform"] = random.choice(["web", "android", "ios"])
                row["last_event_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                top5 = random.sample(groups, 5)
                for idx, group in enumerate(top5):
                    row[f"Top{idx+1}_next_action_group"] = group
                    row[f"Top{idx+1}_confidence"] = round(random.uniform(0.1, 0.99), 4)
                row["Online_conversion_prob"] = round(random.uniform(0, 1), 4)
                row["O2O_reservation_prob"] = round(random.uniform(0, 1), 4)
                row["Marketing_Strategy"] = random.choice([" 推播優惠券", " 提供預約", " 建議展示內容", " 等待更多資料"])
                result_list.append(row)

            df_result = pd.DataFrame(result_list)
            st.session_state.prediction_data = df_result
            st.session_state.all_columns = df_result.columns.tolist()

# 顯示預測結果 & 下載區
if st.session_state.prediction_data is not None:
    df = st.session_state.prediction_data

    st.success(f"分析完成！版本名稱: {result_name}")
    if result_notes:
        st.info(f"備註: {result_notes}")

    st.dataframe(df, use_container_width=True)

    # 選擇欄位 (expanded=True 避免自動收合)
    with st.expander(" 選擇輸出的欄位（可搜尋 / 多選，下拉顯示）", expanded=True):
        selected_columns = st.multiselect(
            "請選擇要輸出的欄位：",
            options=st.session_state.all_columns,
            default=[],
            key="column_selector",
            help="輸入關鍵字搜尋欄位，自由多選"
        )

    if not selected_columns:
        st.warning("請至少選擇一個欄位以進行下載")
    else:
        df_filtered = df[selected_columns]
        csv = df_filtered.to_csv(index=False, sep=separator, header=include_header, encoding=encoding).encode(encoding)
        st.download_button(
            "下載分析結果（CSV）",
            data=csv,
            file_name=f"{result_name}.csv",
            mime="text/csv"
        )

# 顯示預測結果 & 下載區後面加入以下內容：
if st.session_state.prediction_data is not None:
    df = st.session_state.prediction_data

    # Top1 行為分佈圖
    st.markdown("#### 預測行為分佈圖、Top1")
    chart_df = df["Top1_next_action_group"].value_counts().reset_index()
    chart_df.columns = ["action_group", "count"]

    fig = px.bar(
        chart_df,
        x="action_group",
        y="count",
        title="Top1 行為分佈",
        color_discrete_sequence=["#4fc08d"]
    )
    fig.update_layout(xaxis_title="Action Group", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # 信心分數分佈圖
    st.markdown("#### Top1 信心分數分佈（Histogram）")
    fig_conf = px.histogram(
        df,
        x="Top1_confidence",
        nbins=20,
        title="Top1 信心分數分佈",
        color_discrete_sequence=["#4fc08d"]
    )
    fig_conf.update_layout(xaxis_title="Confidence", yaxis_title="User Count")
    st.plotly_chart(fig_conf, use_container_width=True)

    # 信心分數 vs 轉換機率
    st.markdown("#### Top1 信心分數 vs 線上轉換機率")
    fig_scatter = px.scatter(
        df,
        x="Top1_confidence",
        y="Online_conversion_prob",
        title="信心分數 vs 線上轉換機率",
        color_discrete_sequence=["#4fc08d"]
    )
    fig_scatter.update_layout(xaxis_title="Top1 信心分數", yaxis_title="轉換機率")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 行銷策略圓餅圖
    st.markdown("#### 模型建議的行銷策略分佈(這邊之後會改成edm跟彈窗的東西)")
    strategy_df = df["Marketing_Strategy"].value_counts().reset_index()
    strategy_df.columns = ["strategy", "count"]
    fig_pie = px.pie(
        strategy_df,
        names="strategy",
        values="count",
        title="建議行銷策略比例",
        color_discrete_sequence=["#a8e6cf", "#4fc08d", "#3a7763", "#b2f1d5"]
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# streamlit run app1.py
