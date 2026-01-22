import os
import pandas as pd
import gzip
import urllib.request
import ssl
import numpy as np

# ================= 參數設定區 =================

# 1. [設定領域]
SOURCE_CATEGORY = "CD"       # 來源領域 (用來過濾用戶)
TARGET_CATEGORY = "Kitchen"  # 目標領域 (我們要產出的資料)

# 2. [檔案路徑]
RAW_DIR = "/mnt/sda1/sherry/BiGNAS/SGL-BiGNAS-xin/ARLib/data/raw"
# 輸出路徑建議包含兩個領域名稱，方便辨識
OUTPUT_DIR = f"data/clean/{SOURCE_CATEGORY}_{TARGET_CATEGORY}/"

# 3. [過濾設定]
MIN_INTERACTIONS = 3    # Target Domain 中，互動數少於此值的用戶會被剔除 (為了切分 Train/Val/Test)
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # === Step 1: 下載並載入資料 ===
    print(f"=== Step 1: Loading Data ===")
    
    # 1.1 載入 Source Domain
    print(f"   Loading Source: {SOURCE_CATEGORY} ...")
    source_path = download_dataset(SOURCE_CATEGORY, RAW_DIR)
    df_source = get_df(source_path)[['reviewerID']] # Source 只需要 User ID 來做交集
    df_source.columns = ["user"]
    
    # 1.2 載入 Target Domain (我們要主要處理的資料)
    print(f"   Loading Target: {TARGET_CATEGORY} ...")
    target_path = download_dataset(TARGET_CATEGORY, RAW_DIR)
    df_target = get_df(target_path)[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    df_target.columns = ["user", "item", "rating", "timestamp"]

    print(f"   Original Source users: {len(df_source['user'].unique())}")
    print(f"   Original Target records: {len(df_target)}")

    # === Step 2: 找出交疊用戶 (Cross-Domain Constraint) ===
    print(f"=== Step 2: Finding Overlapping Users ===")
    
    source_users = set(df_source["user"].unique())
    target_users = set(df_target["user"].unique())
    
    common_users = source_users.intersection(target_users)
    print(f"   Common Users count: {len(common_users)}")

    # # 只保留 Target Domain 中，屬於交疊用戶的資料
    df_target = df_target[df_target["user"].isin(common_users)]
    print(f"   Target records after overlap filtering: {len(df_target)}")

    # === Step 3: 基本過濾 (Rating & Min Interactions) ===
    print(f"=== Step 3: Filtering Target Data ===")

    # 3.1 評分過濾
    if RATING_THRESHOLD > 0:
        df_target = df_target[df_target["rating"] > RATING_THRESHOLD]

    # 3.2 互動數過濾 (確保每個用戶在 Target Domain 至少有 MIN_INTERACTIONS 筆資料)
    # 注意：雖然是用戶交疊了，但如果他在 Target Domain 只有 1 筆資料，還是無法切分成 Train/Val/Test
    user_counts = df_target["user"].value_counts()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    df_target = df_target[df_target["user"].isin(valid_users)]
    
    print(f"   Records after Min-Interaction filter (>= {MIN_INTERACTIONS}): {len(df_target)}")
    print(f"   Final Valid Users: {len(valid_users)}")

    # === Step 4: 排序與 ID 重編 ===
    print("=== Step 4: Sorting and Remapping IDs ===")
    
    # 依時間排序 (這是 Time-aware split 的關鍵)
    df_target.sort_values(by=["user", "timestamp"], inplace=True)
    df_target.reset_index(drop=True, inplace=True)

    # User ID Mapping
    # 注意：這裡重新編號後，ID 0 ~ N-1 都是交疊且符合條件的用戶
    user_ids = df_target["user"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    df_target["user"] = df_target["user"].map(user2idx)
    
    # Item ID Mapping
    item_ids = df_target["item"].unique()
    item2idx = {item: i for i, item in enumerate(item_ids)} 
    df_target["item"] = df_target["item"].map(item2idx)

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

    # === Step 5: 資料分割 (Time-Aware Split) ===
    print("=== Step 5: Splitting Data (Train/Val/Test) ===")
    
    train_data = []
    val_data = []
    test_data = []

    # 這裡使用 Groupby 加速處理
    # 邏輯：最後一筆 -> Test, 倒數第二筆 -> Val, 其餘 -> Train
    
    grouped = df_target.groupby("user")
    
    for user_id, group in grouped:
        # 轉成 list 方便操作: [user, item, rating]
        interactions = group[['user', 'item', 'rating']].values.tolist()
        
        # 即使前面過濾過，保險起見再檢查一次長度
        if len(interactions) < 3:
            # 如果少於 3 筆，全部放進 Train (或是您可以選擇丟棄)
            train_data.extend(interactions)
            continue
            
        # 最後一筆是 Test
        test_data.append(interactions[-1])
        
        # 倒數第二筆是 Val
        val_data.append(interactions[-2])
        
        # 剩下的 (0 到 -3) 是 Train
        train_data.extend(interactions[:-2])

    # === Step 6: 驗證 Test Set 唯一性 ===
    print("=== Step 6: Validating Test Set Uniqueness ===")
    
    # 建立用戶歷史集合 (User, Item)
    train_val_history = set()
    for row in train_data:
        train_val_history.add((int(row[0]), int(row[1])))
    for row in val_data:
        train_val_history.add((int(row[0]), int(row[1])))
    
    # 檢查 Test Set 是否有重複出現於歷史中
    has_overlap = False
    for row in test_data:
        if (int(row[0]), int(row[1])) in train_val_history:
            has_overlap = True
            break
    
    if has_overlap:
        print("Test items overlap with Train/Val: no")
    else:
        print("Test items overlap with Train/Val: yes (Clean Split)")

    # === Step 7: 寫入檔案 ===
    print(f"=== Step 7: Writing to {OUTPUT_DIR} ===")
    
    def write_txt(filename, data_list):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w') as f:
            for line in data_list:
                # 確保轉成 int 寫入，去除小數點
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