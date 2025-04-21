import os
import gzip
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

def merge_embeddings(input_faa: str, embeddings_dir: str, output_file: str):
    """
    合并所有ESMC embedding文件到一个CSV.gz文件中

    参数:
        input_faa: 原始FASTA文件路径
        embeddings_dir: 包含.esmcemb.csv.gz文件的目录
        output_file: 合并后的输出文件路径(.csv.gz)
    """
    # 读取原始FASTA获取所有序列ID
    seq_ids = [record.id for record in SeqIO.parse(input_faa, "fasta")]

    # 收集所有embedding文件路径
    embedding_files = []
    for seq_id in seq_ids:
        file_path = Path(embeddings_dir) / f"{seq_id}.esmcemb.csv.gz"
        if file_path.exists():
            embedding_files.append((seq_id, file_path))

    print(f"Found {len(embedding_files)} embedding files to merge")

    # 初始化DataFrame列表
    dfs = []

    # 使用进度条读取所有文件
    for seq_id, file_path in tqdm(embedding_files, desc="Merging embeddings"):
        try:
            # 读取单个embedding文件
            df = pd.read_csv(file_path, header=None)
            # 添加序列ID作为行名
            df.index = [seq_id]
            dfs.append(df)
        except Exception as e:
            print(f"\nError reading {file_path}: {str(e)}")

    # 合并所有DataFrame
    if dfs:
        merged_df = pd.concat(dfs)
        print(f"Merged shape: {merged_df.shape}")

        # 保存合并后的文件
        merged_df.to_csv(output_file, compression='gzip')
        print(f"Saved merged embeddings to {output_file}")
    else:
        print("No valid embedding files found to merge")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge individual ESMC embedding files into one CSV.gz file')
    parser.add_argument('--input', required=True, help='Original input FASTA file (.faa)')
    parser.add_argument('--embeddings_dir', required=True, help='Directory containing .esmcemb.csv.gz files')
    parser.add_argument('--output', required=True, help='Output merged CSV.gz file path')

    args = parser.parse_args()

    merge_embeddings(args.input, args.embeddings_dir, args.output)