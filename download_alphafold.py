# -*- coding:utf-8 -*-
import pandas as pd
import requests
import os
from tqdm import tqdm  # 进度条工具

def download_alphafold_pdb(uniprot_id, save_dir='/root/workspace/HiSIF-DTA-main_mam/pdb_restore', version=4):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 构造下载URL
    base_url = "https://alphafold.ebi.ac.uk/files"
    pdb_url = f"{base_url}/AF-{uniprot_id}-F1-model_v{version}.pdb"

    # 设置请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        response = requests.get(pdb_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # 保存文件
        file_path = os.path.join(save_dir, f"AF_{uniprot_id}_v{version}.pdb")
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"\nError downloading {uniprot_id}: {str(e)}")
        return False

def batch_download(csv_path, id_column='uniprot_entry'):
    # 读取数据
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in CSV file")

    # 清洗数据
    uniprot_ids = df[id_column].str.strip().dropna().unique()
    total = len(uniprot_ids)
    success = 0

    # 带进度条的下载
    with tqdm(total=total, desc="Downloading PDB") as pbar:
        for uid in uniprot_ids:
            if download_alphafold_pdb(uid):
                success += 1
            pbar.update(1)

    # 打印总结报告
    print(f"\nDownload completed:")
    print(f"- Total IDs: {total}")
    print(f"- Successfully downloaded: {success}")
    print(f"- Failed downloads: {total - success}")
    print(f"Saved directory: {os.path.abspath('alphafold_pdb')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AlphaFold2 PDB Downloader')
    parser.add_argument('input_csv', type=str, help='Path to input CSV file')
    parser.add_argument('--id_column', type=str, default='uniprot_entry',
                       help='Column name containing Uniprot IDs (default: uniprot_entry)')
    
    args = parser.parse_args()
    
    try:
        batch_download(args.input_csv, args.id_column)
    except Exception as e:
        print(f"Fatal error: {str(e)}")