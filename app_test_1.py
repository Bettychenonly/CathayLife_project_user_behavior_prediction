import streamlit as st

# 🚨 重要：st.set_page_config 必須是第一個 Streamlit 命令
st.set_page_config(page_title="國泰人壽 - 用戶行為預測工具", layout="centered", initial_sidebar_state="collapsed")

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# 嘗試載入 TensorFlow 和相關套件
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from joblib import load
    TF_AVAILABLE = True
except ImportError as e:
    st.error(f"載入依賴套件時發生錯誤: {e}")
    TF_AVAILABLE = False

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
   if not TF_AVAILABLE:
       st.error("TensorFlow 未正確安裝，無法載入模型")
       return None, None, None
   
   # 診斷檔案狀態
   import os
   model_file = "lstm_multi_output_model_v2.h5"
   
   if os.path.exists(model_file):
       file_size = os.path.getsize(model_file)
       st.info(f"模型檔案存在，大小: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
   
   try:
       # 先載入模型
       st.info("正在載入 LSTM 模型...")
       model = load_model("lstm_multi_output_model_v2.h5")
       st.success("✅ LSTM 模型載入成功")
       
       # 逐一載入 encoders
       encoders = {}
       for col in cat_features:
           encoder_file = f'encoder_{col}.pkl'
           try:
               if os.path.exists(encoder_file):
                   file_size = os.path.getsize(encoder_file)
                   st.info(f"載入 {encoder_file} (大小: {file_size} bytes)")
                   
                   if file_size < 500:  # 檔案太小可能有問題
                       st.warning(f"⚠️ {encoder_file} 檔案大小異常: {file_size} bytes")
                       
                       # 檢查是否為 LFS 指針檔案
                       try:
                           with open(encoder_file, 'r', encoding='utf-8') as f:
                               content = f.read()
                               st.error(f"檔案內容（LFS 指針）: {content}")
                               return None, None, None
                       except:
                           with open(encoder_file, 'rb') as f:
                               content = f.read()
                               st.error(f"檔案內容（二進制）: {content}")
                               return None, None, None
                   
                   encoder = load(encoder_file)
                   encoders[col] = encoder
                   st.success(f"✅ {col} encoder 載入成功，類別數: {len(encoder.classes_)}")
               else:
                   st.error(f"❌ 找不到 {encoder_file}")
                   return None, None, None
           except Exception as e:
               st.error(f"❌ 載入 {encoder_file} 時發生錯誤: {e}")
               return None, None, None
       
       # 逐一載入 scalers
       scalers = {}
       for col in num_features:
           scaler_file = f'scaler_feature_{col}.pkl'
           try:
               if os.path.exists(scaler_file):
                   file_size = os.path.getsize(scaler_file)
                   st.info(f"載入 {scaler_file} (大小: {file_size} bytes)")
                   
                   if file_size < 500:  # 檔案太小可能有問題
                       st.warning(f"⚠️ {scaler_file} 檔案大小異常: {file_size} bytes")
                       
                       # 檢查是否為 LFS 指針檔案
                       try:
                           with open(scaler_file, 'r', encoding='utf-8') as f:
                               content = f.read()
                               st.error(f"檔案內容（LFS 指針）: {content}")
                               return None, None, None
                       except:
                           with open(scaler_file, 'rb') as f:
                               content = f.read()
                               st.error(f"檔案內容（二進制）: {content}")
                               return None, None, None
                   
                   scaler = load(scaler_file)
                   scalers[col] = scaler
                   st.success(f"✅ {col} scaler 載入成功")
               else:
                   st.error(f"❌ 找不到 {scaler_file}")
                   return None, None, None
           except Exception as e:
               st.error(f"❌ 載入 {scaler_file} 時發生錯誤: {e}")
               return None, None, None
               
       return model, encoders, scalers
       
   except Exception as e:
       st.error(f"載入過程中發生錯誤: {e}")
       return None, None, None

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

def recommend_strategy(action_group):
    if "試算" in action_group:
        return "推播優惠券"
    elif "商品資訊頁" in action_group or "訂製保險組合" in action_group:
        return "建議展示內容"
    elif "預約" in action_group:
        return "提供預約"
    else:
        return "等待更多資料"



def safe_label_transform(encoder, value, default=0):  # 改為 default=0
    """
    安全地進行 label encoding，避免未知值導致錯誤
    """
    try:
        # 確保值不是 NaN 或 None
        if pd.isna(value) or value is None:
            return default
        
        # 轉換為字符串（因為訓練時使用了 astype(str)）
        str_value = str(value)
        
        # 檢查是否在已知類別中
        if str_value in encoder.classes_:
            return encoder.transform([str_value])[0]
        else:
            # 未知類別返回 0（填充值）
            print(f"警告：未知類別 '{str_value}'，使用默認值 {default}")
            return default
    except Exception as e:
        print(f"Label encoding 錯誤: {e}，使用默認值 {default}")
        return default

def safe_scale(scaler, value, default=0.0):
    """
    安全地進行數值標準化
    """
    try:
        # 確保值不是 NaN 或 None
        if pd.isna(value) or value is None:
            return default
        
        # 轉換為浮點數
        float_value = float(value)
        
        # 進行標準化
        scaled_value = scaler.transform([[float_value]])[0][0]
        
        # 檢查結果是否有效
        if pd.isna(scaled_value):
            return default
        
        return scaled_value
    except Exception as e:
        print(f"Scaling 錯誤: {e}，使用默認值 {default}")
        return default

def create_unknown_category_mapping(encoders):
    """
    為每個 encoder 創建未知類別的映射索引
    """
    mapping = {}
    for col, encoder in encoders.items():
        # 使用類別數量作為未知類別的索引（確保不超出 embedding 範圍）
        unknown_index = min(len(encoder.classes_), encoder.transform(encoder.classes_).max())
        mapping[col] = unknown_index
    return mapping

def safe_label_transform(encoder, value, default=0):
    """
    安全地進行 label encoding，確保不會產生負值
    """
    try:
        # 處理 NaN, None, 空字符串
        if pd.isna(value) or value is None or value == '' or value == 'nan':
            return default
        
        # 轉換為字符串（保持與訓練時一致）
        str_value = str(value).strip()
        
        # 如果是空字符串，返回默認值
        if str_value == '' or str_value == 'nan':
            return default
        
        # 檢查是否在已知類別中
        if str_value in encoder.classes_:
            result = encoder.transform([str_value])[0]
            # 確保結果不是負數
            if result < 0:
                print(f"警告：編碼結果為負數 {result}，使用默認值 {default}")
                return default
            return result
        else:
            # 未知類別，返回默認值
            print(f"警告：未知類別 '{str_value}' (原值: {value})，使用默認值 {default}")
            return default
            
    except Exception as e:
        print(f"Label encoding 錯誤 (值: {value}): {e}，使用默認值 {default}")
        return default

def safe_scale(scaler, value, default=0.0):
    """
    安全地進行數值標準化
    """
    try:
        # 處理 NaN, None
        if pd.isna(value) or value is None:
            return default
        
        # 轉換為浮點數
        float_value = float(value)
        
        # 檢查是否為有限數
        if not np.isfinite(float_value):
            return default
        
        # 進行標準化
        scaled_value = scaler.transform([[float_value]])[0][0]
        
        # 檢查結果是否有效
        if not np.isfinite(scaled_value):
            return default
        
        return scaled_value
        
    except Exception as e:
        print(f"Scaling 錯誤 (值: {value}): {e}，使用默認值 {default}")
        return default

def validate_and_fix_sequence(seq, seq_name):
    """
    驗證並修復序列中的無效值
    """
    seq = np.array(seq)
    
    # 檢查並修復負值
    negative_mask = seq < 0
    if np.any(negative_mask):
        print(f"警告：{seq_name} 中發現 {np.sum(negative_mask)} 個負值，已修復為 0")
        seq[negative_mask] = 0
    
    # 檢查並修復 NaN 值
    nan_mask = ~np.isfinite(seq)
    if np.any(nan_mask):
        print(f"警告：{seq_name} 中發現 {np.sum(nan_mask)} 個無效值，已修復為 0")
        seq[nan_mask] = 0
    
    return seq

def predict_from_uploaded_csv(df):
    if model is None or encoders is None or scalers is None:
        st.error("模型未正確載入，無法進行預測")
        return pd.DataFrame()
    
    # 檢查必要欄位
    required_columns = ['user_pseudo_id', 'event_time'] + cat_features + num_features
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"缺少必要欄位: {missing_columns}")
        return pd.DataFrame()
    
    # 資料預處理
    df = df.copy()
    df = df.sort_values(by=["user_pseudo_id", "event_time"])
    
    # 列印編碼器信息用於除錯
    print("\n=== 編碼器信息 ===")
    for col, encoder in encoders.items():
        print(f"{col}: {len(encoder.classes_)} 個類別, 範圍 [0, {len(encoder.classes_)-1}]")
        print(f"  前5個類別: {list(encoder.classes_[:5])}")
    
    results = []
    
    for user_id, group in df.groupby("user_pseudo_id"):
        try:
            print(f"\n處理用戶: {user_id}")
            last_steps = group.tail(SEQ_LEN)
            pad_len = SEQ_LEN - len(last_steps)
            print(f"  序列長度: {len(last_steps)}, 需要填充: {pad_len}")

            # 處理類別特徵 (4個輸入)
            cat_inputs = []
            for i, col in enumerate(cat_features):
                print(f"  處理類別特徵: {col}")
                
                # 檢查原始數據
                raw_values = last_steps[col].tolist()
                print(f"    原始值: {raw_values}")
                
                # 進行編碼
                seq = []
                for v in raw_values:
                    encoded_value = safe_label_transform(encoders[col], v, default=0)
                    seq.append(encoded_value)
                
                print(f"    編碼後: {seq}")
                
                # 填充序列
                padded_seq = [0] * pad_len + seq
                print(f"    填充後: {padded_seq}")
                
                # 驗證並修復序列
                final_seq = validate_and_fix_sequence(padded_seq, f"{col}_sequence")
                
                # 確保數據類型和形狀
                final_seq = np.array(final_seq, dtype=np.int32).reshape(1, SEQ_LEN)
                
                # 最終檢查
                if final_seq.min() < 0:
                    st.error(f"最終序列仍包含負值: {final_seq}")
                    return pd.DataFrame()
                
                if final_seq.max() >= len(encoders[col].classes_):
                    st.error(f"序列值超出範圍: max={final_seq.max()}, 允許範圍=[0, {len(encoders[col].classes_)-1}]")
                    return pd.DataFrame()
                
                cat_inputs.append(final_seq)
                print(f"    最終形狀: {final_seq.shape}, 範圍: [{final_seq.min()}, {final_seq.max()}]")

            # 處理數值特徵 (3個輸入)
            num_inputs = []
            for col in num_features:
                print(f"  處理數值特徵: {col}")
                
                # 檢查原始數據
                raw_values = last_steps[col].tolist()
                print(f"    原始值: {raw_values}")
                
                # 進行標準化
                seq = []
                for v in raw_values:
                    scaled_value = safe_scale(scalers[col], v, default=0.0)
                    seq.append(scaled_value)
                
                print(f"    標準化後: {seq}")
                
                # 填充序列
                padded_seq = [0.0] * pad_len + seq
                
                # 驗證並修復序列
                final_seq = validate_and_fix_sequence(padded_seq, f"{col}_sequence")
                
                # 確保數據類型和形狀
                final_seq = np.array(final_seq, dtype=np.float32).reshape(1, SEQ_LEN)
                num_inputs.append(final_seq)
                print(f"    最終形狀: {final_seq.shape}")

            # 組合所有輸入
            all_inputs = cat_inputs + num_inputs
            
            print(f"  總輸入數量: {len(all_inputs)} (期望: 7)")
            
            # 最終驗證
            for i, inp in enumerate(all_inputs):
                input_name = cat_features[i] if i < len(cat_features) else num_features[i - len(cat_features)]
                print(f"    輸入 {i+1} ({input_name}): 形狀={inp.shape}, 類型={inp.dtype}")
                
                if i < len(cat_features):  # 類別特徵
                    if inp.min() < 0:
                        st.error(f"類別輸入 {input_name} 包含負值: {inp.min()}")
                        return pd.DataFrame()
                    
                    max_allowed = len(encoders[cat_features[i]].classes_) - 1
                    if inp.max() > max_allowed:
                        st.error(f"類別輸入 {input_name} 值超出範圍: {inp.max()} > {max_allowed}")
                        return pd.DataFrame()

            # 進行預測
            print("  開始預測...")
            predictions = model.predict(all_inputs, verbose=0)
            y_pred_action_group, y_pred_online, y_pred_o2o = predictions
            print("  預測完成")

            # 處理預測結果
            top5_indices = y_pred_action_group[0].argsort()[-5:][::-1]
            top5_confidences = y_pred_action_group[0][top5_indices]

            inv_action = {i: v for i, v in enumerate(encoders['action_group'].classes_)}
            top5_actions = [inv_action.get(idx, "未知類別") for idx in top5_indices]

            strategy = recommend_strategy(top5_actions[0])

            result = {
                "user_pseudo_id": user_id,
                "Top1_next_action_group": top5_actions[0],
                "Top1_confidence": round(float(top5_confidences[0]), 4),
                "Online_conversion_prob": round(float(y_pred_online[0][0]), 4),
                "O2O_reservation_prob": round(float(y_pred_o2o[0][0]), 4),
                "Marketing_Strategy": strategy
            }
            
            # 添加 Top 5 結果
            for i in range(5):
                result[f"Top{i+1}_next_action_group"] = top5_actions[i]
                result[f"Top{i+1}_confidence"] = round(float(top5_confidences[i]), 4)

            results.append(result)
            print(f"  用戶 {user_id} 處理完成")
            
        except Exception as e:
            st.error(f"處理用戶 {user_id} 時發生錯誤: {e}")
            print(f"錯誤詳情: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n總共處理了 {len(results)} 個用戶")
    return pd.DataFrame(results)

# UI 介面
st.title("📊 國泰人壽 – 多元訪客行為預測工具")

# 載入模型並顯示狀態
with st.spinner("正在載入模型..."):
    if TF_AVAILABLE:
        model, encoders, scalers = load_model_and_encoders()
        if model is not None:
            st.success("✅ 模型載入成功")
        else:
            st.error("❌ 模型載入失敗")
            st.stop()
    else:
        st.error("⚠️ TensorFlow 不可用")
        st.stop()

st.markdown("請上傳包含用戶行為歷程資料的 CSV 進行模型預測")

# 檔案上傳器
uploaded_file = st.file_uploader(
    "📥 上傳 CSV（需含欄位：user_pseudo_id, event_time, action_group, source, medium, platform, staytime, has_shared, revisit_count）", 
    type=["csv"]
)

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"✅ 成功讀取 {len(user_df)} 筆資料，開始模型預測...")
        st.dataframe(user_df.head(10), use_container_width=True)

        with st.spinner("正在進行預測..."):
            df_result = predict_from_uploaded_csv(user_df)
            
        if not df_result.empty:
            st.session_state.prediction_data = df_result
            st.session_state.all_columns = df_result.columns.tolist()
            st.success("🎉 預測完成！")
        else:
            st.error("預測失敗，請檢查資料格式")

    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")

# 預測結果展示與下載
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

    if len(df) > 0:
        # 圖表 1: Top1 行為分佈
        chart_df = df["Top1_next_action_group"].value_counts().reset_index()
        chart_df.columns = ["action_group", "count"]
        fig1 = px.bar(chart_df, x="action_group", y="count", title="Top1 行為分佈")
        st.plotly_chart(fig1, use_container_width=True)

        # 圖表 2: 信心分數分佈
        fig2 = px.histogram(df, x="Top1_confidence", nbins=20, title="Top1 信心分數分佈")
        st.plotly_chart(fig2, use_container_width=True)

        # 圖表 3: 散點圖
        fig3 = px.scatter(df, x="Top1_confidence", y="Online_conversion_prob", title="信心分數 vs 線上轉換機率")
        st.plotly_chart(fig3, use_container_width=True)

        # 圖表 4: 行銷策略比例
        strategy_df = df["Marketing_Strategy"].value_counts().reset_index()
        strategy_df.columns = ["strategy", "count"]
        fig4 = px.pie(strategy_df, names="strategy", values="count", title="建議行銷策略比例")
        st.plotly_chart(fig4, use_container_width=True)
