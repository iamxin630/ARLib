import os
import pandas as pd
import gzip
import urllib.request
import ssl
import numpy as np

# ================= 參數設定區 =================

# 1. [設定領域]
SOURCE_CATEGORY = "Book"       # 來源領域 (用來過濾用戶)
TARGET_CATEGORY = "Electronic"  # 目標領域 (我們要產出的資料)

# 2. [檔案路徑]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
# 輸出路徑建議包含兩個領域名稱，方便辨識
OUTPUT_DIR = os.path.join(BASE_DIR, f"data/clean/{SOURCE_CATEGORY}_{TARGET_CATEGORY}/")

# 3. [過濾設定]
MIN_INTERACTIONS = 2    # Target Domain 中，互動數要求 (對齊另一專案: Target >= 2)
ITEM_MIN_INTERACTIONS = 7 # 每個物品至少互動次數 (對齊另一專案: Item >= 7)
RATING_THRESHOLD = 0    # 評分過濾 (0 代表保留所有)

# ============================================

# Amazon 5-core 資料集對照表 (維持不變)
ROOT_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
CATEGORY_FILE_NAMES = {
    "Book": "reviews_Books_5.json.gz",
    "Electronic": "reviews_Electronics_5.json.gz",
    "Movie": "reviews_Movies_and_TV_5.json.gz",
    "CD": "reviews_CDs_and_Vinyl_5.json.gz",
    "Clothing": "reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
    "Kitchen": "reviews_Home_and_Kitchen_5.json.gz",
    "Kindle": "reviews_Kindle_Store_5.json.gz",
    "Sports": "reviews_Sports_and_Outdoors_5.json.gz",
    "Phone": "reviews_Cell_Phones_and_Accessories_5.json.gz",
    "Health": "reviews_Health_and_Personal_Care_5.json.gz",
    "Toy": "reviews_Toys_and_Games_5.json.gz",
    "Game": "reviews_Video_Games_5.json.gz",
    "Tool": "reviews_Tools_and_Home_Improvement_5.json.gz",
    "Beauty": "reviews_Beauty_5.json.gz",
    "App": "reviews_Apps_for_Android_5.json.gz",
    "Office": "reviews_Office_Products_5.json.gz",
    "Pet": "reviews_Pet_Supplies_5.json.gz",
    "Automotive": "reviews_Automotive_5.json.gz",
    "Grocery": "reviews_Grocery_and_Gourmet_Food_5.json.gz",
    "Patio": "reviews_Patio_Lawn_and_Garden_5.json.gz",
    "Baby": "reviews_Baby_5.json.gz",
    "Music": "reviews_Digital_Music_5.json.gz",
    "Instrument": "reviews_Musical_Instruments_5.json.gz",
    "Video": "reviews_Amazon_Instant_Video_5.json.gz",
}

def download_dataset(category, save_dir):
    """自動下載指定的 Dataset"""
    if category not in CATEGORY_FILE_NAMES:
        raise ValueError(f"錯誤的類別: {category}。請檢查拼字是否在列表內。")

    filename = CATEGORY_FILE_NAMES[category]
    url = ROOT_URL + filename
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return save_path

    print(f"Downloading {category} from {url}...")
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        urllib.request.urlretrieve(url, save_path)
        print("Download finished!")
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise e

    return save_path

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def process_cross_domain():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # === Step 1: 下載並載入資料 ===
    print(f"=== Step 1: Loading Data ===")
    
    # 1.1 載入 Source Domain
    print(f"   Loading Source: {SOURCE_CATEGORY} ...")
    source_path = download_dataset(SOURCE_CATEGORY, RAW_DIR)
    df_source = get_df(source_path)[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    df_source.columns = ["user", "item", "rating", "timestamp"]
    df_source["is_target"] = 0
    
    # 1.2 載入 Target Domain (我們要主要處理的資料)
    print(f"   Loading Target: {TARGET_CATEGORY} ...")
    target_path = download_dataset(TARGET_CATEGORY, RAW_DIR)
    df_target = get_df(target_path)[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    df_target.columns = ["user", "item", "rating", "timestamp"]
    df_target["is_target"] = 1

    print(f"   Original Source records: {len(df_source)}")
    print(f"   Original Target records: {len(df_target)}")

    # === Step 2: 找出跨域交疊用戶 ===
    print(f"=== Step 2: Finding Common Users ===")
    source_users = set(df_source["user"].unique())
    target_users = set(df_target["user"].unique())
    common_users = source_users.intersection(target_users)
    print(f"   Common Users count: {len(common_users)}")

    df = pd.concat([df_source, df_target])
    df = df[df["user"].isin(common_users)]
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # === Step 3: 防止時間洩漏 (Time Leakage Prevention) ===
    print(f"=== Step 3: Filtering Time Leakage ===")
    target_df_pre = df[df["is_target"] == 1]
    training_time = dict()
    
    # 找出各用戶在 Target Domain 的倒數第二筆時間 (-2)
    for user, group in target_df_pre.groupby("user"):
        if len(group) >= 2:
            training_time[user] = group["timestamp"].values[-2]
        else:
            training_time[user] = -1

    remove_index = []
    source_df_pre = df[df["is_target"] == 0]
    for user, group in source_df_pre.groupby("user"):
        if user in training_time and training_time[user] != -1:
            # 移除在 Source Domain 中，時間 >= Target Domain 訓練終點的資料
            indexs = group[group["timestamp"] >= training_time[user]].index
            remove_index.extend(indexs)
    
    df.drop(remove_index, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # === Step 4: 遞迴過濾 (Iterative Filtering) ===
    print(f"=== Step 4: Iterative Filtering (Item >= {ITEM_MIN_INTERACTIONS}, Target User >= {MIN_INTERACTIONS}, Source User >= 1) ===")
    iteration = 0
    while True:
        iteration += 1
        before_count = len(df)
        
        # 1. 物品過濾 (使用 ITEM_MIN_INTERACTIONS 設定)
        s_item_counts = df[df["is_target"] == 0]["item"].value_counts()
        t_item_counts = df[df["is_target"] == 1]["item"].value_counts()
        keep_s_items = s_item_counts[s_item_counts >= ITEM_MIN_INTERACTIONS].index
        keep_t_items = t_item_counts[t_item_counts >= ITEM_MIN_INTERACTIONS].index
        
        df = df[
            ((df["is_target"] == 0) & (df["item"].isin(keep_s_items))) |
            ((df["is_target"] == 1) & (df["item"].isin(keep_t_items)))
        ]
        
        # 2. 用戶過濾 (使用 MIN_INTERACTIONS 設定)
        u_counts_t = df[df["is_target"] == 1]["user"].value_counts()
        u_counts_s = df[df["is_target"] == 0]["user"].value_counts()
        valid_u = u_counts_t[u_counts_t >= MIN_INTERACTIONS].index.intersection(u_counts_s[u_counts_s >= 1].index)
        
        df = df[df["user"].isin(valid_u)]
        
        after_count = len(df)
        print(f"   Filter Iteration {iteration}: {before_count} -> {after_count}")
        
        if before_count == after_count:
            break

    if len(df) == 0:
        raise ValueError("Empty dataset after filtering.")

    # 最終導出：只需要 Target Domain 資料
    print(f"=== Step 5: Exporting Target Domain Only ===")
    df_export = df[df["is_target"] == 1].copy()

    # === Step 6: 排序與 ID 重編 ===
    print("=== Step 6: Sorting and Remapping IDs ===")
    
    df_export.sort_values(by=["user", "timestamp"], inplace=True)
    df_export.reset_index(drop=True, inplace=True)

    # User ID Mapping
    user_ids = df_export["user"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    df_export["user"] = df_export["user"].map(user2idx)
    
    # Item ID Mapping
    item_ids = df_export["item"].unique()
    item2idx = {item: i for i, item in enumerate(item_ids)} 
    df_export["item"] = df_export["item"].map(item2idx)

    print(f"   Total Users: {len(user_ids)}")
    print(f"   Total Items: {len(item_ids)}")

    # 儲存 Mapping 對照表
    with open(os.path.join(OUTPUT_DIR, "item_list.txt"), "w") as f:
        f.write("org_id remap_id\n")
        for org_id, remap_id in item2idx.items():
            f.write(f"{org_id} {remap_id}\n")
    
    with open(os.path.join(OUTPUT_DIR, "user_list.txt"), "w") as f:
        f.write("org_id remap_id\n")
        for org_id, remap_id in user2idx.items():
            f.write(f"{org_id} {remap_id}\n")

    # === Step 7: 資料分割 (Time-Aware Split) ===
    print("=== Step 7: Splitting Data (Train/Val/Test) ===")
    
    train_data = []
    val_data = []
    test_data = []

    grouped = df_export.groupby("user")
    
    for user_id, group in grouped:
        interactions = group[['user', 'item', 'rating']].values.tolist()
        
        # 最後一筆是 Test
        test_data.append(interactions[-1])
        # 倒數第二筆是 Val
        val_data.append(interactions[-2])
        # 其餘是 Train
        train_data.extend(interactions[:-2])

    # === Step 8: 寫入檔案 ===
    print(f"=== Step 8: Writing to {OUTPUT_DIR} ===")
    
    def write_txt(filename, data_list):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w') as f:
            for line in data_list:
                f.write(f"{int(line[0])} {int(line[1])} {int(line[2])}\n")
    
    write_txt("train.txt", train_data)
    write_txt("val.txt", val_data)
    write_txt("test.txt", test_data)

    print("Done!")
    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")
    print(f"Data Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_cross_domain()