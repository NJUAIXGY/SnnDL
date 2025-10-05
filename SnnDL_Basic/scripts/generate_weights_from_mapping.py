#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成基于映射结果的 per-PE 权重文件（权重驱动路由模式）。

- 输入：
  * 映射文件 mapping.json（来自 neuron_mapping_framework::exportArtifacts），形如：
      {"assignments": [{"neuron_id": <uint>, "pe_id": <uint>, "core_id": <uint>}, ...]}
  * 现有 per-PE 权重目录（按模板命名：classification_weights_pe_{pe}.bin）
    每个文件是 row-major 的 float32 二进制，形状为 [rows_per_pe, total_neurons]

- 输出：
  * 新的 per-PE 权重目录（默认写入 weights_mapped/，不会覆盖原文件）
    文件命名与模板一致，内容为重排后的权重矩阵，使“目标神经元→PE”符合映射 assignment，
    同时对列维（源神经元）做同一重排，从而保持连边 (old_src→old_dst) 在新索引下映射为
    (new_src→new_dst)。

注意：
- 本脚本不修改 SST/SnnDL 代码。它仅离线重排权重行/列，以便 SnnPESubComponent 的
  权重驱动路由在仿真时自动构建“源全局ID→目的全局ID”的扇出表。
- 为避免破坏现有数据，默认输出到 weights_mapped/。如需直接覆盖原 weights/，显式传入 -o 指定。
"""

import argparse
import json
import os
from pathlib import Path
from array import array

# 简单的二进制 float32 读/写工具（避免依赖 numpy）

def read_f32_rowmajor(path: Path, rows: int, cols: int):
    with open(path, 'rb') as f:
        data = array('f')
        data.frombytes(f.read())
        if len(data) != rows * cols:
            raise ValueError(f"{path} size mismatch: got {len(data)} floats, expect {rows*cols}")
    # data 是一维 row-major，切成二维列表
    out = []
    off = 0
    for _ in range(rows):
        out.append(data[off:off+cols].tolist())
        off += cols
    return out


def write_f32_rowmajor(path: Path, rows: int, cols: int, rows_list):
    # rows_list: list[list[float]]，形状 [rows, cols]
    buf = array('f')
    for r in range(rows):
        row = rows_list[r]
        if len(row) != cols:
            raise ValueError(f"Row {r} length {len(row)} != cols {cols}")
        buf.extend(row)
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def discover_pe_files(weights_dir: Path, pattern_prefix: str = "classification_weights_pe_", suffix: str = ".bin"):
    files = []
    for p in sorted(weights_dir.glob(f"{pattern_prefix}*{suffix}")):
        name = p.name
        try:
            mid = name[len(pattern_prefix): -len(suffix)]
            pe = int(mid)
        except Exception:
            continue
        files.append((pe, p))
    files.sort(key=lambda x: x[0])
    return files


def load_mapping(mapping_path: Path):
    m = json.loads(mapping_path.read_text())
    assigns = m.get("assignments", [])
    mapping = {}
    for a in assigns:
        nid = int(a["neuron_id"])  # old global id
        pe = int(a["pe_id"])       # target pe
        mapping[nid] = pe
    return mapping


def build_new_index(mapping: dict, total_neurons: int, rows_per_pe: int, total_pes: int):
    """
    生成两个数组：
    - row_new_to_old[i] = old_row_index    （新行 i 使用旧矩阵哪一行）
    - col_new_to_old[j] = old_col_index    （新列 j 使用旧矩阵哪一列）

    做法：
    - 按映射将所有 old_id -> target_pe 分桶；每个 PE 内按 old_id 升序分配本地行 [0..rows_per_pe)
    - new_global_id = pe*rows_per_pe + local_row
    - 该映射既用于“行”（目标神经元重排），也用于“列”（源神经元重排），从而保持连接关系
    """
    # 1) 先按PE分桶
    per_pe_ids = {pe: [] for pe in range(total_pes)}
    for old_id in range(total_neurons):
        tgt_pe = mapping.get(old_id, old_id // rows_per_pe)  # 缺失时回退到原分配
        if tgt_pe < 0 or tgt_pe >= total_pes:
            raise ValueError(f"mapping for old_id={old_id} -> invalid pe {tgt_pe}")
        per_pe_ids[tgt_pe].append(old_id)

    # 2) 容量校验
    for pe, ids in per_pe_ids.items():
        if len(ids) > rows_per_pe:
            raise ValueError(f"PE{pe} assigned {len(ids)} neurons > rows_per_pe={rows_per_pe}")

    # 3) 构造 old->new / new->old 映射
    old_to_new = {}
    new_to_old = {}
    for pe in range(total_pes):
        ids = sorted(per_pe_ids[pe])
        for local_row, old_id in enumerate(ids):
            new_id = pe * rows_per_pe + local_row
            old_to_new[old_id] = new_id
            new_to_old[new_id] = old_id
    # 4) 补齐未填满的行（保留为占位行，来源旧行：按照原布局顺延；也可以置零，这里保持稳定映射）
    #    若该PE分配未满，则将“原本属于该PE但未被映射选中的 old_id”按原顺序填充剩余行
    for pe in range(total_pes):
        base = pe * rows_per_pe
        assigned = [nid for nid in range(total_neurons) if old_to_new.get(nid, -1) // rows_per_pe == pe]
        used_slots = len(assigned)
        if used_slots < rows_per_pe:
            # 从原块中补齐
            block_ids = list(range(base, base + rows_per_pe))
            for old_id in block_ids:
                if old_id in old_to_new:
                    continue
                if used_slots >= rows_per_pe:
                    break
                new_id = base + used_slots
                old_to_new[old_id] = new_id
                new_to_old[new_id] = old_id
                used_slots += 1

    # 5) 生成 new->old 索引数组
    row_new_to_old = [None] * total_neurons
    col_new_to_old = [None] * total_neurons
    for new_id in range(total_neurons):
        old_id = new_to_old.get(new_id)
        if old_id is None:
            # 若仍未映射，回退为自身
            old_id = new_id
        row_new_to_old[new_id] = old_id
        col_new_to_old[new_id] = old_id

    return row_new_to_old, col_new_to_old, old_to_new, new_to_old


def main():
    ap = argparse.ArgumentParser(description="Generate per-PE weights from mapping.json (row/col reorder)")
    ap.add_argument("-m", "--mapping", required=True, help="Path to mapping.json (exportArtifacts)")
    ap.add_argument("-i", "--input-weights-dir", default=str(Path(__file__).resolve().parent.parent / "weights"),
                    help="Input directory containing classification_weights_pe_{pe}.bin")
    ap.add_argument("-o", "--output-weights-dir", default=str(Path(__file__).resolve().parent.parent / "weights_mapped"),
                    help="Output directory to write remapped weights")
    ap.add_argument("--rows-per-pe", type=int, default=16, help="Rows per PE (NEURONS_PER_PE), default 16")
    ap.add_argument("--pattern-prefix", default="classification_weights_pe_", help="File prefix")
    ap.add_argument("--pattern-suffix", default=".bin", help="File suffix")
    args = ap.parse_args()

    mapping_path = Path(args.mapping).resolve()
    in_dir = Path(args.input_weights_dir).resolve()
    out_dir = Path(args.output_weights_dir).resolve()

    if not mapping_path.exists():
        raise SystemExit(f"mapping file not found: {mapping_path}")
    if not in_dir.exists():
        raise SystemExit(f"input weights dir not found: {in_dir}")

    files = discover_pe_files(in_dir, args.pattern_prefix, args.pattern_suffix)
    if not files:
        raise SystemExit(f"no per-PE files match {args.pattern_prefix}*{args.pattern_suffix} in {in_dir}")

    total_pes = len(files)
    rows_per_pe = args.rows_per_pe

    # 读取第一份文件以推断列数
    first_path = files[0][1]
    fsz = first_path.stat().st_size
    if fsz % 4 != 0:
        raise SystemExit(f"{first_path} size {fsz} not multiple of 4")
    floats = fsz // 4
    if floats % rows_per_pe != 0:
        raise SystemExit(f"{first_path} has {floats} floats not divisible by rows_per_pe={rows_per_pe}")
    cols_total = floats // rows_per_pe

    total_neurons = total_pes * rows_per_pe
    if cols_total != total_neurons:
        raise SystemExit(f"matrix width {cols_total} != total_neurons {total_neurons} (expected cols == N)")

    print(f"[INFO] PEs={total_pes}, rows_per_pe={rows_per_pe}, total_neurons={total_neurons}, cols={cols_total}")

    # 读取旧矩阵 W_old（按旧的 block-全局ID 顺序）
    # 预分配二维列表 W_old[rows][cols]
    W_old = [[0.0 for _ in range(cols_total)] for _ in range(total_neurons)]
    for pe, p in files:
        rows = read_f32_rowmajor(p, rows_per_pe, cols_total)
        base = pe * rows_per_pe
        for r in range(rows_per_pe):
            W_old[base + r] = rows[r]

    # 加载映射
    mapping = load_mapping(mapping_path)
    if not mapping:
        print("[WARN] mapping.assignments is empty; fallback to identity mapping (no change)")

    # 构造新旧索引关系
    row_new_to_old, col_new_to_old, old_to_new, new_to_old = build_new_index(
        mapping, total_neurons, rows_per_pe, total_pes
    )

    # 创建输出目录（避免覆盖原目录）
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写出新 per-PE 文件
    for pe in range(total_pes):
        base = pe * rows_per_pe
        # 构造该PE的新行（每行应用列重排）
        new_rows = []
        for lr in range(rows_per_pe):
            new_row_id = base + lr
            old_row_id = row_new_to_old[new_row_id]
            old_row = W_old[old_row_id]
            # 应用列重排：新列顺序 j 使用 old_col = col_new_to_old[j]
            new_row = [old_row[col_new_to_old[j]] for j in range(cols_total)]
            new_rows.append(new_row)
        out_path = out_dir / f"{args.pattern_prefix}{pe}{args.pattern_suffix}"
        write_f32_rowmajor(out_path, rows_per_pe, cols_total, new_rows)
        print(f"[WRITE] {out_path} ({rows_per_pe}x{cols_total})")

    # 简要报告：每个PE的分配数量
    per_pe_count = {pe: 0 for pe in range(total_pes)}
    for old_id, new_id in old_to_new.items():
        pe = new_id // rows_per_pe
        per_pe_count[pe] += 1
    print("[REPORT] assigned neurons per PE:")
    for pe in range(total_pes):
        print(f"  PE{pe}: {per_pe_count[pe]} / {rows_per_pe}")

    print("[DONE] Remapped weights generated. Update your weights_dir if needed, then run SST.")


if __name__ == "__main__":
    main()
