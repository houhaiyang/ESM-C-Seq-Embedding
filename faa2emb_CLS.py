#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esmc_model_tokenizers
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm


MODEL_PATH = "/home/share/huadjyin/home/houhaiyang/HF_HOME/transformers/EvolutionaryScale/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"


def load_esmc_model(model_path: str, device: str = "cuda"):
    """
    加载ESMC模型
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ESMC(
        d_model=1152, n_heads=18, n_layers=36,
        tokenizer=get_esmc_model_tokenizers()
    ).eval().to(device)

    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    if device.type != "cpu":
        model = model.to(torch.bfloat16)

    return model, device


def get_esmc_embedding(model, device, sequence: str) -> np.ndarray:
    """
    获取蛋白质序列的ESMC embedding (只取CLS标记)
    """
    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        input_ids = model._tokenize([protein.sequence])
        input_ids = input_ids.to(device)
        output = model(input_ids)
        embeddings = output.embeddings

        # 只取第一个位置(CLS标记)的embedding
        cls_embedding = embeddings[:, 0, :]  # 形状变为 [1, embedding_dim]

    return cls_embedding.float().cpu().numpy()[0]  # 取第一个样本的CLS embedding


def process_fasta(input_faa: str, outdir: str, model, device):
    """
    处理FASTA文件，为每条序列生成embedding文件
    显示进度条，不打印序列信息
    """
    # 确保输出目录存在
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 读取FASTA文件并获取记录总数用于进度条
    records = list(SeqIO.parse(input_faa, "fasta"))
    total_sequences = len(records)

    # 使用tqdm创建进度条
    for record in tqdm(records, desc="Processing sequences", unit="seq"):
        seq_id = record.id
        sequence = str(record.seq)

        # 生成输出文件名
        output_file = os.path.join(outdir, f"{seq_id}.esmcemb.csv.gz")

        # 如果文件已存在则跳过
        if os.path.exists(output_file):
            continue  # 不再打印跳过信息以保持进度条整洁

        try:
            # 获取embedding
            embedding = get_esmc_embedding(model, device, sequence)

            # 保存为CSV.GZ（一行数据）
            df = pd.DataFrame(embedding.reshape(1, -1))  # 确保形状为 (1, embedding_dim)
            df.to_csv(output_file, index=False, header=False, compression='gzip')

        except Exception as e:
            print(f"\nError processing {seq_id}: {str(e)}")  # 错误信息换行显示，避免干扰进度条


def main():
    parser = argparse.ArgumentParser(description='Generate ESMC embeddings for protein sequences in a FASTA file.')
    parser.add_argument('--input', required=True, help='Input FASTA file (.faa) containing protein sequences')
    parser.add_argument('--outdir', required=True, help='Output directory for embedding files')
    parser.add_argument('--device', default='cuda', help='Device to use for computation (cuda or cpu)')

    args = parser.parse_args()

    # 加载模型
    print("Loading ESMC model...")
    model, device = load_esmc_model(MODEL_PATH, args.device)
    print(f"Model loaded on {device}")

    # 处理FASTA文件
    print(f"Processing sequences from {args.input}...")
    process_fasta(args.input, args.outdir, model, device)
    print("Processing completed.")


if __name__ == "__main__":
    main()
