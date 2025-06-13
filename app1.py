import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import plotly.express as px

# 初始化 session state
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "all_columns" not in st.session_state:
    st.session_state.all_columns = []

# 模型與編碼器載入
SEQ_LEN = 10
cat_features = ['action_group', 'source', 'medium', 'platform']
num_features = ['staytime', 'has_shared', 'revisit_count']

@st.cache_resource
def load_model_and_encoders():
    model = load_model("lstm_multi_output_model_v2.h5")
    encoders = {col: load(f'encoder_{col}.pkl') for col in cat_features}
    scalers = {col: load(f'scaler_feature_{col}.pkl') for col in num_features}
    return model, encoders, scalers

model, encoders, scalers = load_model_and_encoders()

def safe_label_transform(encoder, value, default=-1):
    try:
        return encoder.transform([value])[0]
    except:
        return default

def safe_scale(scaler, value, default=0.0):
    try:
        return scaler.transform(np.array([[value]])).flatten()[0]
    except:
        return default

# 行銷策略推薦邏輯（根據 Top1 行為預測結果）
def recommend_strategy(action_group):
    if "試算" in action_group:
        return "推播優惠券"
    elif "商品資訊頁" in action_group or "訂製保險組合" in action_group:
        return "建議展示內容"
    elif "預約" in action_group:
        return "提供預約"
    else:
        return "等待更多資料"

def predict_from_uploaded_csv(df):
    df = df.sort_values(by=["user_pseudo_id", "event_time"])
    results = []

    for user_id, group in df.groupby("user_pseudo_id"):
        last_steps = group.tail(SEQ_LEN)
        pad_len = SEQ_LEN - len(last_steps)

        cat_seqs = []
        for col in cat_features:
            seq = [safe_label_transform(encoders[col], v) for v in last_steps[col]]
            cat_seqs.append(([0] * pad_len) + seq)

        num_seqs = []
        for col in num_features:
            seq = [safe_scale(scalers[col], v) for v in last_steps[col]]
            num_seqs.append(([0.0] * pad_len) + seq)

        X_cat = [np.array(seq).reshape(1, SEQ_LEN) for seq in cat_seqs]
        X_num = np.array(num_seqs).T.reshape(1, SEQ_LEN, len(num_features))

        y_pred_action_group, y_pred_online, y_pred_o2o = model.predict((*X_cat, X_num), verbose=0)

        top5_indices = y_pred_action_group[0].argsort()[-5:][::-1]
        top5_confidences = y_pred_action_group[0][top5_indices]

        inv_action = {i: v for i, v in enumerate(encoders['action_group'].classes_)}
        top5_actions = [inv_action[idx] for idx in top5_indices]

        strategy = recommend_strategy(top5_actions[0])

        result = {
            "user_pseudo_id": user_id,
            "Top1_next_action_group": top5_actions[0],
            "Top1_confidence": round(top5_confidences[0], 4),
            "Online_conversion_prob": round(y_pred_online[0][0], 4),
            "O2O_reservation_prob": round(y_pred_o2o[0][0], 4),
            "Marketing_Strategy": strategy
        }
        for i in range(5):
            result[f"Top{i+1}_next_action_group"] = top5_actions[i]
            result[f"Top{i+1}_confidence"] = round(top5_confidences[i], 4)

        results.append(result)

    return pd.DataFrame(results)

# === UI 介面 ===
st.set_page_config(page_title="國泰人壽 - 用戶行為預測工具", layout="centered", initial_sidebar_state="collapsed")
st.title("📊 國泰人壽 – 多元訪客行為預測工具")
st.markdown("請上傳包含用戶行為歷程資料的 CSV 進行模型預測")

uploaded_file = st.file_uploader("📥 上傳 CSV（需含欄位：user_pseudo_id, event_time, action_group, source, medium, platform, staytime, has_shared, revisit_count）", type=["csv"])

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功讀取 {len(user_df)} 筆資料，開始模型預測...")
        st.dataframe(user_df.head(10), use_container_width=True)

        df_result = predict_from_uploaded_csv(user_df)
        st.session_state.prediction_data = df_result
        st.session_state.all_columns = df_result.columns.tolist()
        st.success("🎉 預測完成！")

    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")

# === 預測結果展示與下載 ===
if st.session_state.prediction_data is not None:
    df = st.session_state.prediction_data
    st.markdown("### 📋 模型預測結果")
    st.dataframe(df, use_container_width=True)

    with st.expander("🔧 選擇欄位下載結果", expanded=True):
        selected_columns = st.multiselect(
            "請選擇要輸出的欄位：",
            options=st.session_state.all_columns,
            default=st.session_state.all_columns
        )

    if selected_columns:
        csv = df[selected_columns].to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載結果 CSV", data=csv, file_name="prediction_result.csv", mime="text/csv")
    else:
        st.warning("請選擇至少一個欄位")

    # 視覺化圖表
    st.markdown("### 📊 統計圖表")

    chart_df = df["Top1_next_action_group"].value_counts().reset_index()
    chart_df.columns = ["action_group", "count"]
    fig1 = px.bar(chart_df, x="action_group", y="count", title="Top1 行為分佈")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="Top1_confidence", nbins=20, title="Top1 信心分數分佈")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="Top1_confidence", y="Online_conversion_prob", title="信心分數 vs 線上轉換機率")
    st.plotly_chart(fig3, use_container_width=True)

    strategy_df = df["Marketing_Strategy"].value_counts().reset_index()
    strategy_df.columns = ["strategy", "count"]
    fig4 = px.pie(strategy_df, names="strategy", values="count", title="建議行銷策略比例")
    st.plotly_chart(fig4, use_container_width=True)


# streamlit run app1.py
