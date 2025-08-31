import pandas as pd

def load_estat(path, year):
    # 「表章項目 コード」行を探す
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "表章項目 コード" in line:
                header_row = i
                break
    
    df = pd.read_csv(path, encoding="utf-8", skiprows=header_row)
    
    # 列名統一
    df = df.rename(columns={"地域 コード": "市区町村コード", "地域": "市区町村名"})
    
    # 必要列だけ残す
    df = df[["市区町村コード", "市区町村名", "総数", "空き家"]].copy()
    df["year"] = year
    
    # 数字カンマ削除 & 数値化
    for c in ["総数","空き家"]:
        df[c] = df[c].astype(str).str.replace(",","").replace({"-":None}).astype(float)
    
    # 空き家率を追加
    df["空き家率"] = df["空き家"] / df["総数"] * 100
    
    # コードはゼロ埋め5桁文字列に統一
    df["市区町村コード"] = df["市区町村コード"].astype(str).str.zfill(5)
    
    return df

# 実行例
df18 = load_estat("data/raw/FEH_00200522_2018_2508.csv", 2018)
df23 = load_estat("data/raw/FEH_00200522_2023_2508.csv", 2023)

# 結合
df = pd.concat([df18, df23], ignore_index=True)

print(df.head())