import streamlit as st

# 🚨 重要：st.set_page_config 必須是第一個 Streamlit 命令
st.set_page_config(page_title="國泰人壽 - 用戶行為預測工具", layout="wide", initial_sidebar_state="collapsed")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

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

# 1. 定義等級映射函式
def get_level(action_group: str) -> float:
    # 檢查輸入是否為有效值
    if action_group is None or pd.isna(action_group) or action_group == '':
        return -1
        
    # 轉換為字符串以確保一致性
    action_group = str(action_group).strip()
    
    # 等級 0
    if action_group in {"其他", "保險視圖、保單明細、資產總覽、保險明細"}:
        return 0
    # 等級 1
    if action_group == "找服務（尋求服務與客服）":
        return 1
    # 等級 2
    if "商品資訊頁" in action_group or action_group == "好康優惠":
        return 2
    # 等級 3
    combo3 = {
        "自由配－投資規劃", "自由配", "訂製保險組合", "自由配－套餐"
    }
    if action_group in combo3 or action_group.startswith("試算"):
        return 3
    # 等級 4
    result4 = {
        "自由配－保障規劃試算結果",
        "訂製保險組合－人身規劃試算結果",
        "我的保險試算結果",
        "訂製保險組合－投資規劃試算結果",
        "自由配－配置我的基金試算結果"
    }
    if action_group in result4:
        return 4
    # 等級 5
    if action_group in {
        "保存與分享試算結果",
        "保存與分享自由配、訂製組合結果",
        "Line分享轉傳"
    }:
        return 5
    # 等級 6
    if action_group in {
        "挑選預約顧問",
        "預約顧問與商品諮詢",
        "立即投保"
    }:
        return 6
    # 等級 7.x
    if action_group in {"方案確認", "填寫預約資料"}:
        return 7.1
    if action_group in {"資料填寫與確認", "投保資格確認", "手機驗證碼"}:
        return 7.2
    if action_group == "線上繳費":
        return 7.3
    # 等級 8
    if action_group in {"完成網路投保", "完成O2O"}:
        return 8
    # 未知
    return -1

# 2. 推薦策略函式（使用 next_action_group 參數）
def recommend_strategy(
        next_action_group: str,
        history: list[str],
        last_event_time: datetime
    ) -> str | None:
    # 檢查 next_action_group 是否為 None 或空值
    if next_action_group is None or pd.isna(next_action_group) or next_action_group == '':
        return None
        
    # 處理時區問題：統一轉換為 naive datetime
    now = datetime.now()
    if hasattr(last_event_time, 'tz_localize'):
        # 如果是 pandas Timestamp 且有時區信息，轉換為 naive
        if last_event_time.tz is not None:
            last_event_time = last_event_time.tz_localize(None)
    elif hasattr(last_event_time, 'tzinfo'):
        # 如果是 datetime 且有時區信息，轉換為 naive
        if last_event_time.tzinfo is not None:
            last_event_time = last_event_time.replace(tzinfo=None)

    # 等級 0
    if next_action_group in {"其他", "保險視圖、保單明細、資產總覽、保險明細"}:
        return None

    # 等級 1：找服務
    if next_action_group == "找服務（尋求服務與客服）":
        s = "箭頭提醒「找服務」介面在畫面上方選單"
        if history[-10:].count("找服務（尋求服務與客服）") > 3:
            s += "；讓右下角的阿發的「Hi 需要幫忙嗎」秀出來"
        return s

    # 等級 2：商品資訊頁
    if "商品資訊頁" in next_action_group:
        cnt = sum(1 for ag in history[-10:] if ag and "商品資訊頁" in ag)
        if cnt >= 3:
            return "顯示入口推動試算"
        return None

    # 等級 2：好康優惠
    if next_action_group == "好康優惠":
        if now - last_event_time > timedelta(minutes=3):
            return "彈窗顯示好康優惠相關資訊"
        return None

    # 等級 3：挑選組合
    combo3 = {"自由配－投資規劃", "自由配", "訂製保險組合", "自由配－套餐"}
    if next_action_group in combo3:
        prompt = (
            "不知道如何開始嗎？簡單三個問題，讓系統提供最適合你的商品！"
            if next_action_group == "訂製保險組合"
            else "一鍵搭配個人化的商品組合，只要 2 分鐘！"
        )
        return f"彈窗：「{prompt}」"

    # 等級 3：試算
    if next_action_group.startswith("試算"):
        return "彈窗：「一鍵帶你完成試算，只要 2 步驟，取得商品費用估算」"

    # 等級 4：試算結果
    result4 = {
        "自由配－保障規劃試算結果",
        "訂製保險組合－人身規劃試算結果",
        "我的保險試算結果",
        "訂製保險組合－投資規劃試算結果",
        "自由配－配置我的基金試算結果"
    }
    if next_action_group in result4:
        return "提供進度提醒：就快完成了！試算結果即將產生"

    # 等級 5：保存／分享結果
    if next_action_group in {
        "保存與分享試算結果",
        "保存與分享自由配、訂製組合結果"
    }:
        return "諮詢按鈕提示：「對試算結果有疑問嗎？預約免費專人解讀！」"
    if next_action_group == "Line分享轉傳":
        return "彈窗鼓勵推薦、推播推薦獎勵機制或分享回饋活動"

    # 等級 6：挑顧問 / 諮詢需求
    if next_action_group in {"挑選預約顧問", "預約顧問與商品諮詢"}:
        return "彈窗推薦顧問：「這位專家最擅長XX險，3 分鐘內確認預約」"
    if next_action_group == "立即投保":
        return "立即投保按鈕或橫幅CTA：「立即投保享優惠！」"

    # 等級 7.1：方案確認 / 填寫預約資料
    if next_action_group in {"方案確認", "填寫預約資料"}:
        if now - last_event_time <= timedelta(minutes=5):
            return (
                "進度提醒：「再三步即可完成投保」"
                if next_action_group == "方案確認"
                else "進度提醒：「再兩步即可完成預約」"
            )

    # 等級 7.2：資料填寫與確認 / 投保資格確認 / 手機驗證碼
    if next_action_group in {"資料填寫與確認", "投保資格確認", "手機驗證碼"}:
        if now - last_event_time > timedelta(minutes=30):
            return "機器人引導：「上次還沒填完？點此一鍵回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return (
                "進度提醒：「還差最後一步就OK，即將完成預約」"
                if next_action_group == "手機驗證碼"
                else "進度提醒：「還差最後兩步就OK，即將完成投保」"
            )

    # 等級 7.3：線上繳費
    if next_action_group == "線上繳費":
        if now - last_event_time > timedelta(minutes=30):
            return "機器人引導：「上次還沒填完？點此一鍵回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return "進度提醒：「還差最後一步就OK，即將完成投保」"

    # 等級 8：完成網投 / 完成O2O
    if next_action_group == "完成網路投保":
        if now - last_event_time > timedelta(minutes=30):
            return (
                "寄發EDM提醒即將完成投保，並附上連結「還差最後一步就OK，"
                "點我回到流程」"
            )
        return "感謝頁推薦：「感謝您！邀請好友完成網投送雙方 NT$300元」"
    if next_action_group == "完成O2O":
        if now - last_event_time > timedelta(minutes=30):
            return (
                "寄發EDM提醒即將完成預約，並附上連結「還差最後一步就OK，"
                "點我回到流程」"
            )

    return None

# 3. 根據 Top1~Top5 選出 next_action_group
def pick_next_action_group(row) -> str:
    candidates = []
    for i in range(1, 6):
        ag = row[f'Top{i}_next_action_group']
        conf = row[f'Top{i}_confidence']
        
        # 檢查 action_group 是否為有效值
        if ag is None or pd.isna(ag) or ag == '':
            lvl = -1  # 給無效值最低等級
        else:
            lvl = get_level(ag)
            
        candidates.append((ag, conf, lvl))
    
    # 先比等級，再比機率，過濾掉無效值
    valid_candidates = [(ag, conf, lvl) for ag, conf, lvl in candidates if lvl >= 0]
    
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: (x[2], x[1]))
        return best[0]
    else:
        # 如果沒有有效候選，返回第一個非空值或默認值
        for ag, conf, lvl in candidates:
            if ag is not None and not pd.isna(ag) and ag != '':
                return ag
        return "其他"  # 默認值

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
    required_columns = ['user_pseudo_id', 'event_time', 'platform', 'action', 'action_group'] \
                       + cat_features + num_features
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"缺少必要欄位: {missing_columns}")
        return pd.DataFrame()
    
    # 資料預處理
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 統一處理時區：如果有時區信息，轉換為 naive datetime
    if df['event_time'].dt.tz is not None:
        df['event_time'] = df['event_time'].dt.tz_localize(None)
    
    df = df.sort_values(by=["user_pseudo_id", "event_time"])
    
    results = []
    
    for user_id, group in df.groupby("user_pseudo_id"):
        try:
            # 取最後 SEQ_LEN 步
            last_steps = group.tail(SEQ_LEN)
            pad_len = SEQ_LEN - len(last_steps)
            
            # --- 擷取 raw features ---
            last = last_steps.iloc[-1]
            raw_last_event_time   = last['event_time']
            raw_last_platform     = last['platform']
            raw_last_action       = last['action']
            raw_last_action_group = last['action_group']
            
            # 取前 2~9 步的 raw action & action_group
            prev_records = {}
            for i in range(2, 10):
                if len(last_steps) >= i:
                    rec = last_steps.iloc[-i]
                    prev_records[f"-{i}_action"]        = rec['action']
                    prev_records[f"-{i}_action_group"]  = rec['action_group']
                else:
                    prev_records[f"-{i}_action"]        = None
                    prev_records[f"-{i}_action_group"]  = None
            
            # --- 類別特徵編碼 & 填充 ---
            cat_inputs = []
            for col in cat_features:
                raw_vals = last_steps[col].tolist()
                encoded = [ safe_label_transform(encoders[col], v, default=0)
                            for v in raw_vals ]
                padded = [0]*pad_len + encoded
                seq = validate_and_fix_sequence(padded, f"{col}_sequence")
                cat_inputs.append(np.array(seq, dtype=np.int32).reshape(1, SEQ_LEN))
            
            # --- 數值特徵標準化 & 填充 ---
            num_inputs = []
            for col in num_features:
                raw_vals = last_steps[col].tolist()
                scaled = [ safe_scale(scalers[col], v, default=0.0)
                           for v in raw_vals ]
                padded = [0.0]*pad_len + scaled
                seq = validate_and_fix_sequence(padded, f"{col}_sequence")
                num_inputs.append(np.array(seq, dtype=np.float32).reshape(1, SEQ_LEN))
            
            all_inputs = cat_inputs + num_inputs

            # 進行預測並取 Top5
            y_pred_action_group, y_pred_online, y_pred_o2o = model.predict(all_inputs, verbose=0)
            top5_idx   = y_pred_action_group[0].argsort()[-5:][::-1]
            top5_confs = y_pred_action_group[0][top5_idx]
            inv_ag     = {i: v for i, v in enumerate(encoders['action_group'].classes_)}
            top5_actions = [inv_ag.get(idx, "未知") for idx in top5_idx]
            
            # 正確地 build 中介 dict，注意 for 迴圈後要有冒號
            temp = {}
            for i in range(5):
                temp[f"Top{i+1}_next_action_group"] = top5_actions[i]
                temp[f"Top{i+1}_confidence"]       = round(float(top5_confs[i]), 4)
                
            # 再把這個 dict 傳給 pick_next_action_group
            next_ag = pick_next_action_group(temp)
            
            # 計算行銷策略
            strategy = recommend_strategy(
                next_ag,
                [ prev_records.get(f"-{i}_action_group") for i in range(1, 10) ],
                raw_last_event_time
            )
            
            # --- 組裝結果 ---
            result = {
                "user_pseudo_id":      user_id,
                "last_event_time":     raw_last_event_time,
                "last_platform":       raw_last_platform,
                "last_action":         raw_last_action,
                "last_action_group":   raw_last_action_group,
                **prev_records,
                "Online_conversion_prob":  round(float(y_pred_online[0][0]), 4),
                "O2O_reservation_prob":    round(float(y_pred_o2o[0][0]), 4),
                "Marketing_Strategy":      strategy
            }
            
            # 添加 Top1~Top5
            for i in range(5):
                result[f"Top{i+1}_next_action_group"] = top5_actions[i]
                result[f"Top{i+1}_confidence"]       = round(float(top5_confs[i]), 4)
            
            results.append(result)
        
        except Exception as e:
            st.error(f"處理用戶 {user_id} 時發生錯誤: {e}")
            continue
    
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
        # 檢查是否是新上傳的文件
        current_file_name = uploaded_file.name
        if (st.session_state.uploaded_file_name != current_file_name or 
            st.session_state.prediction_data is None):
            
            # 新文件或未處理過，開始預測
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ 成功讀取 {len(user_df)} 筆資料")
            
            # === 輸入篩選區域 ===
            with st.expander("📅 資料期間篩選", expanded=True):
                col1, col2 = st.columns(2)
                
                # 獲取資料的時間範圍
                user_df['event_time'] = pd.to_datetime(user_df['event_time'])
                min_date = user_df['event_time'].min().date()
                max_date = user_df['event_time'].max().date()
                
                with col1:
                    start_date = st.date_input(
                        "起始日期", 
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                with col2:
                    end_date = st.date_input(
                        "截止日期", 
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                # 套用日期篩選
                if start_date <= end_date:
                    filtered_df = user_df[
                        (user_df['event_time'].dt.date >= start_date) & 
                        (user_df['event_time'].dt.date <= end_date)
                    ]
                    st.info(f"篩選後資料: {len(filtered_df)} 筆 ({start_date} ~ {end_date})")
                else:
                    st.error("起始日期不能大於截止日期")
                    filtered_df = user_df
            
            st.dataframe(filtered_df.head(10), use_container_width=True)

            with st.spinner("正在進行預測..."):
                df_result = predict_from_uploaded_csv(filtered_df)
                
            if not df_result.empty:
                st.session_state.prediction_data = df_result
                st.session_state.all_columns = df_result.columns.tolist()
                st.session_state.uploaded_file_name = current_file_name
                st.session_state.processed_data = filtered_df
                st.success("🎉 預測完成！")
            else:
                st.error("預測失敗，請檢查資料格式")
        else:
            # 相同文件且已處理過，直接顯示之前的結果
            st.success(f"✅ 文件已處理完成 ({len(st.session_state.processed_data)} 筆資料)")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
            st.success("🎉 使用快取的預測結果")

    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")

# 預測結果展示與下載
if st.session_state.prediction_data is not None:
    df = st.session_state.prediction_data
    
    # === 輸出篩選區域 ===
    st.markdown("### 🎯 結果篩選")
    
    with st.expander("🔍 進階篩選選項", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # 1️⃣ 最近n步內踩過特定行為的用戶
            st.markdown("**1️⃣ 歷史行為篩選**")
            history_steps = st.selectbox("最近幾步內", options=list(range(2, 10)), index=6)  # 預設8步
            
            # 收集所有可能的 action_group
            history_columns = [f"-{i}_action_group" for i in range(2, 10)]
            all_history_actions = set()
            for col in history_columns:
                if col in df.columns:
                    all_history_actions.update(df[col].dropna().unique())
            all_history_actions = sorted([x for x in all_history_actions if pd.notna(x)])
            
            selected_history_actions = st.multiselect(
                "曾經執行的行為",
                options=all_history_actions,
                help="選擇用戶在歷史中曾經執行過的行為"
            )
        
        with col2:
            # 2️⃣ 下一步Top N預測篩選
            st.markdown("**2️⃣ 預測行為篩選**")
            top_n = st.selectbox("檢查Top幾的預測", options=[1, 2, 3, 4, 5], index=0)
            
            # 收集所有可能的預測 action_group
            prediction_columns = [f"Top{i}_next_action_group" for i in range(1, 6)]
            all_prediction_actions = set()
            for col in prediction_columns:
                if col in df.columns:
                    all_prediction_actions.update(df[col].dropna().unique())
            all_prediction_actions = sorted([x for x in all_prediction_actions if pd.notna(x)])
            
            selected_prediction_actions = st.multiselect(
                "預測的下一步行為",
                options=all_prediction_actions,
                help="選擇預測的下一步行為"
            )
        
        # 3️⃣ Top1 機率門檻
        st.markdown("**3️⃣ 預測信心門檻**")
        min_confidence = st.number_input(
            "Top1 最低機率",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="設定 Top1 預測的最低信心分數（0-1之間）"
        )
        
        if not (0.0 <= min_confidence <= 1.0):
            st.error("⚠️ 機率值必須在 0.0 到 1.0 之間，請重新輸入")
            min_confidence = 0.0
    
    # 套用篩選邏輯
    filtered_df = df.copy()
    filter_conditions = []  # 記錄篩選條件
    
    # 1️⃣ 歷史行為篩選
    if selected_history_actions:
        history_mask = pd.Series([False] * len(filtered_df))
        for _, row in filtered_df.iterrows():
            for step in range(2, history_steps + 2):
                col_name = f"-{step}_action_group"
                if col_name in row and row[col_name] in selected_history_actions:
                    history_mask[row.name] = True
                    break
        filtered_df = filtered_df[history_mask]
        
        # 記錄篩選條件
        actions_text = "、".join(selected_history_actions)
        filter_conditions.append(f"最近{history_steps}步內曾踩過：{actions_text}")
    
    # 2️⃣ 預測行為篩選
    if selected_prediction_actions:
        prediction_mask = pd.Series([False] * len(filtered_df))
        for _, row in filtered_df.iterrows():
            for n in range(1, top_n + 1):
                col_name = f"Top{n}_next_action_group"
                if col_name in row and row[col_name] in selected_prediction_actions:
                    prediction_mask[row.name] = True
                    break
        filtered_df = filtered_df[prediction_mask]
        
        # 記錄篩選條件
        actions_text = "、".join(selected_prediction_actions)
        filter_conditions.append(f"下一步的Top{top_n}可能為：{actions_text}")
    
    # 3️⃣ Top1 機率門檻篩選
    if min_confidence > 0.0:
        filtered_df = filtered_df[filtered_df['Top1_confidence'] >= min_confidence]
        filter_conditions.append(f"Top1的最低機率需大於等於：{min_confidence}")
    
    # 📋 篩選條件總結
    st.markdown("### 📋 篩選條件總結")
    
    if filter_conditions:
        # 獲取日期篩選資訊
        if 'start_date' in locals() and 'end_date' in locals():
            date_range = f"在 {start_date} ~ {end_date} 之間"
        else:
            date_range = "在所選日期範圍內"
        
        # 構建條件描述
        conditions_text = f"""
        **您的篩選條件為：**
        
        {date_range}，
        """
        
        for i, condition in enumerate(filter_conditions):
            if i == 0:
                conditions_text += f"{condition}"
            else:
                conditions_text += f"\n且 {condition}"
        
        conditions_text += f"""
        
        **當前條件下，全數符合的用戶數量共有：{len(filtered_df)} 人**
        """
        
        st.markdown(conditions_text)
        
        # 如果有篩選但無結果
        if len(filtered_df) == 0:
            st.warning("⚠️ 沒有符合所有篩選條件的用戶")
    else:
        # 沒有設定任何篩選條件
        st.info("🔍 目前未設定任何篩選條件，顯示所有預測結果")
        st.markdown(f"**用戶總數：{len(filtered_df)} 人**")
    
    # 顯示篩選結果
    if len(filtered_df) > 0:
        with st.expander("📊 查看篩選結果", expanded=False):
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("⚠️ 沒有符合篩選條件的資料")

    # 4️⃣ 下載設定
    if len(filtered_df) > 0:
        with st.expander("📥 下載設定", expanded=True):
            st.markdown("**選擇輸出欄位**")
            
            # 提供快速選項
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 全選"):
                    st.session_state.selected_columns = st.session_state.all_columns
            with col2:
                if st.button("📊 核心欄位"):
                    core_columns = [
                        'user_pseudo_id', 'last_event_time', 'last_action_group',
                        'Top1_next_action_group', 'Top1_confidence', 
                        'Online_conversion_prob', 'O2O_reservation_prob', 'Marketing_Strategy'
                    ]
                    st.session_state.selected_columns = [col for col in core_columns if col in st.session_state.all_columns]
            with col3:
                if st.button("🎯 預測欄位"):
                    prediction_cols = [col for col in st.session_state.all_columns if 'Top' in col or 'prob' in col]
                    st.session_state.selected_columns = prediction_cols

            # 欄位多選器
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = st.session_state.all_columns
            
            selected_columns = st.multiselect(
                "選擇要輸出的欄位",
                options=st.session_state.all_columns,
                default=st.session_state.selected_columns,
                key="column_selector"
            )

            if selected_columns:
                csv = filtered_df[selected_columns].to_csv(index=False).encode("utf-8-sig")
                
                # 根據篩選條件生成檔名
                if filter_conditions:
                    filename = f"prediction_result_filtered_{len(filtered_df)}users.csv"
                else:
                    filename = f"prediction_result_all_{len(filtered_df)}users.csv"
                
                st.download_button(
                    "📥 下載結果 CSV", 
                    data=csv, 
                    file_name=filename, 
                    mime="text/csv",
                    help=f"下載 {len(filtered_df)} 位用戶的篩選結果"
                )
            else:
                st.warning("請選擇至少一個欄位")
    else:
        st.error("📥 無法下載：沒有符合條件的資料")

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
