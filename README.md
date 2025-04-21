# 使用ESM-C 600M生成seq embeddings

2025-04-21

ESM-C项目：https://github.com/SPYfighting/esm-C
在ESM-C模型中，每个氨基酸残基（即序列中的每个字符）都会生成一个对应的embedding向量。
因此，输出的embedding行数应该有关于输入蛋白质序列的长度（氨基酸数量），而不是像ESM2固定的1行。
对于长度为L的蛋白质序列，ESM-C的输出形状为（L+2）×d，包含CLS和EOS标记。

### 前提：
- 已经下载 esmc-600m-2024-12 的权重，并在我的脚本中更改为实际路径：
```python
model_path = "/home/share/.../EvolutionaryScale/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"
```
下载可参考：https://blog.csdn.net/qq_43611382/article/details/144453821

- 已经配置好ESM-C所需conda环境，并加载指定CUDA cudnn Gcc nccl openblas cmake等。

- 只提取CLS作为序列的embedding。


## ESM-C Seq-Embedding Example Usage

测试序列文件为 test_proteins.faa。

- 先执行 faa2emb_CLS.py，为每条序列生成 csv.gz 文件：

```bash
cd /home/.../houhaiyang/method/ESM-C
source ~/bashrc/ESM-C-py310-torch241-cu118.bashrc
python faa2emb_CLS.py --input test_proteins.faa --outdir embeddings/test/
```
- 再执行 merge_emb.py，合并 Embeddings：
```bash
python merge_emb.py --input test_proteins.faa  --embeddings_dir ./embeddings/test/ --output ./embeddings/test_proteins_merged_embeddings.csv.gz
```

最终结果为：./embeddings/test_proteins_merged_embeddings.csv.gz 
