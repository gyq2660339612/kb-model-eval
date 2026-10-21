"""命令行入口:加载数据集,跑召回评估(+可选重排评估),输出标准结果目录。

示例:
    python -m src.run_evaluation --backend mock --rerank-backend mock
    python -m src.run_evaluation --backend bm25
    python -m src.run_evaluation --backend openai --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-Embedding-8B --rerank-backend openai \
        --rerank-base-url http://127.0.0.1:8001 --rerank-model Qwen/Qwen3-Reranker-8B \
        --output-dir results/qwen3-8b-r1 --save-candidates

结果目录结构(--output-dir,对应总框架 §13):
    <output-dir>/run_config.json        运行配置(模型、参数、数据 hash、环境)
    <output-dir>/environment.txt        Python/numpy/平台版本
    <output-dir>/report.md              人类可读报告
    <output-dir>/report.json            机器可读完整结果
    <output-dir>/per_query_results.jsonl  逐查询指标与 top 结果(供失败分析与 bootstrap 对比)
    <output-dir>/benchmark_summary.csv  指标汇总表
    <output-dir>/candidates.jsonl       (可选)冻结的重排候选集
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends import DEFAULT_QWEN3_INSTRUCTION, build_embedding_backend, build_reranker_backend
from src.data_loader import load_dataset
from src.eval_rerank import dump_candidates, load_candidates, run_rerank_eval
from src.eval_retrieval import run_retrieval_eval

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_k_values(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


def _load_config(path: str) -> dict:
    full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        return {}
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_hashes(data_dir: str) -> dict:
    out = {}
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv", "qrels.jsonl"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            out[name] = _sha256(p)
    return out


def _check_acceptance(report: dict, config: dict) -> list[dict]:
    """硬门槛:整体阈值 + 分 category 门槛(对应总框架 §8.2)"""
    rows = []
    final_stage = "rerank" if "rerank" in report else "retrieval"
    final_metrics = (
        report["rerank"]["after"]["metrics"] if final_stage == "rerank"
        else report["retrieval"]["metrics"]["metrics"]
    )
    for key, target in config.get("acceptance_thresholds", {}).items():
        if key in final_metrics:
            val = final_metrics[key]
            rows.append({"scope": "overall", "metric": key, "value": round(val, 4),
                         "target": target, "pass": val >= target})
    by_cat = (
        report["rerank"]["by_category_after"] if final_stage == "rerank"
        else report["retrieval"]["by_category"]
    )
    for category, gates in config.get("category_gates", {}).items():
        cat_metrics = (by_cat.get(category) or {}).get("metrics", {})
        for key, target in gates.items():
            if key in cat_metrics:
                val = cat_metrics[key]
                rows.append({"scope": f"category:{category}", "metric": key,
                             "value": round(val, 4), "target": target, "pass": val >= target})
            else:
                rows.append({"scope": f"category:{category}", "metric": key,
                             "value": None, "target": target, "pass": False})
    return rows


def _composite_score(report: dict, config: dict) -> dict | None:
    """综合评分:Σ 权重 × 指标 × 100(对应总框架 §8.1,系统性能/成本项需人工补录)"""
    spec = config.get("composite_score")
    if not spec:
        return None
    sources = {
        "retrieval": report["retrieval"]["metrics"]["metrics"],
        "rerank_before": report.get("rerank", {}).get("before", {}).get("metrics", {}),
        "rerank_after": report.get("rerank", {}).get("after", {}).get("metrics", {}),
    }
    components = []
    total = 0.0
    weight_sum = 0.0
    for item in spec:
        val = sources.get(item["source"], {}).get(item["metric"])
        if val is None:
            return None
        contrib = item["weight"] * val * 100
        components.append({**item, "metric_value": round(val, 4),
                           "contribution": round(contrib, 2)})
        total += contrib
        weight_sum += item["weight"]
    if abs(weight_sum - 1.0) > 0.01:
        return None
    return {"value": round(total, 2), "components": components,
            "note": "仅含可自动计算的质量指标;性能/成本/稳健项按总框架 §8.1 人工补录"}


def _render_markdown(report: dict) -> str:
    lines = ["# 模型评估报告", ""]
    cfg = report["config"]
    lines += [
        f"- 时间:{report['timestamp']}",
        f"- 数据集:{cfg['data_dir']}",
        f"- Embedding 后端:{cfg['embedding_backend']}",
        f"- Reranker 后端:{cfg.get('reranker_backend') or '未启用'}",
        f"- 指标 k 值:{cfg['k_values']}",
        "",
        "## 召回(Retrieval)指标",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
    ]
    ret = report["retrieval"]
    for key, val in ret["metrics"]["metrics"].items():
        lines.append(f"| {key} | {val:.4f} |")
    if ret["metrics"].get("num_unanswerable"):
        lines.append(f"| (无答案查询,单独计数) | {ret['metrics']['num_unanswerable']} 条 |")
    perf = ret["performance"]
    lines += [
        "",
        "## 召回性能",
        "",
        f"- 文档数:{perf['num_documents']},查询数:{perf['num_queries']}"
        + (f",向量维度:{perf['vector_dim']}" if perf.get("vector_dim") else ""),
    ]
    if perf.get("doc_throughput_per_sec") is not None:
        lines.append(f"- 文档向量化:{perf['doc_embed_seconds']}s({perf['doc_throughput_per_sec']} docs/s)")
    if perf.get("index_seconds") is not None:
        lines.append(f"- BM25 建索引:{perf['index_seconds']}s({perf.get('note', '')})")
    lines.append(f"- 查询平均延迟:{perf['query_avg_latency_ms']} ms(批处理口径)")

    if ret.get("by_category"):
        lines += ["", "## 分层指标(按查询 category)", "",
                  "| 类别 | 查询数 | ndcg@10 | recall@10 | hit_rate@1 | mrr@10 |", "|---|---|---|---|---|---|"]
        for cat, m in ret["by_category"].items():
            mm = m["metrics"]
            def _g(k: str) -> str:
                return f"{mm[k]:.4f}" if k in mm else "—"
            lines.append(f"| {cat} | {m['num_queries']} | {_g('ndcg@10')} | {_g('recall@10')} "
                         f"| {_g('hit_rate@1')} | {_g('mrr@10')} |")

    if "rerank" in report:
        rr = report["rerank"]
        lines += ["", "## 重排序(Rerank)对比", "",
                  f"(候选集:{'冻结 candidates.jsonl' if rr['performance']['frozen_candidates'] else '召回实时生成'},"
                  f" top-{rr['performance']['rerank_top_k']})", "",
                  "| 指标 | 重排前 | 重排后 | Δ | 变化率 |", "|---|---|---|---|---|"]
        for key in rr["after"]["metrics"]:
            b = rr["before"]["metrics"][key]
            a = rr["after"]["metrics"][key]
            d = rr["delta"][key]
            dp = rr["delta_pct"][key]
            lines.append(f"| {key} | {b:.4f} | {a:.4f} | {d:+.4f} | "
                         f"{f'{dp:+.2f}%' if dp is not None else '—'} |")
        rp = rr["performance"]
        lines += ["", f"- 重排延迟:平均 {rp['avg_latency_ms']} ms/query,P95 {rp['p95_latency_ms']} ms", ""]

    if report.get("composite_score"):
        cs = report["composite_score"]
        lines += ["", f"## 综合评分:{cs['value']} / 100", "",
                  "| 维度 | 权重 | 指标 | 指标值 | 贡献 |", "|---|---|---|---|---|"]
        for c in cs["components"]:
            lines.append(f"| {c['name']} | {c['weight']} | {c['metric']}({c['source']}) "
                         f"| {c['metric_value']} | {c['contribution']} |")
        lines += ["", f"> {cs['note']}", ""]

    if report.get("acceptance"):
        lines += ["", "## 验收门槛对照", "", "| 范围 | 指标 | 实测 | 门槛 | 结论 |", "|---|---|---|---|---|"]
        for row in report["acceptance"]:
            val = f"{row['value']}" if row["value"] is not None else "无数据"
            lines.append(f"| {row['scope']} | {row['metric']} | {val} | {row['target']} | "
                         f"{'PASS' if row['pass'] else 'FAIL'} |")
        lines.append("")
    return "\n".join(lines)


def _write_output_dir(output_dir: str, report: dict, md: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(report["config"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    import numpy as np
    with open(os.path.join(output_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(f"python: {sys.version}\nplatform: {platform.platform()}\n"
                f"numpy: {np.__version__}\n")

    final_stage = report.get("rerank") or report["retrieval"]
    k_values = report["config"]["k_values"]
    with open(os.path.join(output_dir, "per_query_results.jsonl"), "w", encoding="utf-8") as f:
        for qid, metrics in final_stage["per_query"].items():
            row = {"query_id": qid,
                   "labels": report["config"].get("labels", {}).get(qid, {}),
                   "metrics": metrics,
                   "top_results": final_stage["results"][qid][: max(k_values)]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "benchmark_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "metric", "value"])
        for key, val in report["retrieval"]["metrics"]["metrics"].items():
            writer.writerow(["retrieval", key, f"{val:.6f}"])
        if "rerank" in report:
            for key, val in report["rerank"]["before"]["metrics"].items():
                writer.writerow(["rerank_before", key, f"{val:.6f}"])
            for key, val in report["rerank"]["after"]["metrics"].items():
                writer.writerow(["rerank_after", key, f"{val:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库 Embedding/Reranker 模型评估")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "sample"))
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--top-k", type=int, default=100, help="召回深度")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--backend", default="mock", choices=["mock", "openai", "st", "bm25"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--instruction", default=None,
                        help="query 侧指令(Qwen3 模板);空字符串表示不加指令")
    parser.add_argument("--dimensions", type=int, default=None, help="MRL 向量截断维度")
    parser.add_argument("--device", default=None)

    parser.add_argument("--rerank-backend", default="none",
                        choices=["none", "mock", "openai", "st"])
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-base-url", default=None)
    parser.add_argument("--rerank-top-k", type=int, default=20, help="重排深度")
    parser.add_argument("--candidates", default=None,
                        help="冻结候选集 candidates.jsonl(提供后重排不再实时取召回结果)")
    parser.add_argument("--save-candidates", action="store_true",
                        help="把召回 top-(rerank-top-k) 冻结为 candidates.jsonl")

    parser.add_argument("--config", default=os.path.join("configs", "eval_config.json"))
    parser.add_argument("--output", default=None, help="报告输出前缀(生成 .json 与 .md)")
    parser.add_argument("--output-dir", default=None,
                        help="标准结果目录(run_config/report/per_query/summary 全套)")
    args = parser.parse_args()

    instruction = DEFAULT_QWEN3_INSTRUCTION if args.instruction is None else (args.instruction or None)

    corpus, queries, qrels, labels = load_dataset(args.data_dir)
    k_values = _parse_k_values(args.k_values)
    config = _load_config(args.config)

    if args.backend == "bm25":
        from src.bm25 import run_bm25_eval
        print(f"[1/2] BM25 基线评估(文档 {len(corpus)},查询 {len(queries)})")
        retrieval = run_bm25_eval(corpus, queries, qrels, k_values=k_values,
                                  top_k=args.top_k, labels=labels)
        backend_name = "bm25-okapi(pure-python)"
    else:
        emb_backend = build_embedding_backend(
            args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key,
            instruction=instruction, dimensions=args.dimensions, device=args.device,
            batch_size=args.batch_size,
        )
        backend_name = emb_backend.name
        print(f"[1/2] 召回评估:{backend_name}(文档 {len(corpus)},查询 {len(queries)})")
        retrieval = run_retrieval_eval(
            emb_backend, corpus, queries, qrels,
            k_values=k_values, batch_size=args.batch_size, top_k=args.top_k, labels=labels,
        )

    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "data_dir": args.data_dir,
            "data_hashes": _data_hashes(args.data_dir),
            "embedding_backend": backend_name,
            "reranker_backend": None,
            "k_values": k_values,
            "top_k": args.top_k,
            "instruction": instruction,
            "dimensions": args.dimensions,
            "labels": labels,
        },
        "retrieval": retrieval,
    }

    if args.save_candidates:
        cand_path = (os.path.join(args.output_dir, "candidates.jsonl")
                     if args.output_dir else "candidates.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(cand_path)), exist_ok=True)
        dump_candidates(retrieval["results"], args.rerank_top_k, cand_path)
        report["config"]["candidates_file"] = cand_path
        print(f"候选集已冻结:{cand_path}")

    if args.rerank_backend != "none":
        rr_backend = build_reranker_backend(
            args.rerank_backend, model=args.rerank_model,
            base_url=args.rerank_base_url, api_key=args.api_key, device=args.device,
        )
        report["config"]["reranker_backend"] = rr_backend.name
        frozen = load_candidates(args.candidates) if args.candidates else None
        print(f"[2/2] 重排评估:{rr_backend.name}(top-{args.rerank_top_k}"
              f"{' ,冻结候选集' if frozen else ''})")
        report["rerank"] = run_rerank_eval(
            rr_backend, corpus, queries, qrels, retrieval["results"],
            k_values=k_values, rerank_top_k=args.rerank_top_k,
            candidates=frozen, labels=labels,
        )

    if config:
        acc = _check_acceptance(report, config)
        if acc:
            report["acceptance"] = acc
        comp = _composite_score(report, config)
        if comp:
            report["composite_score"] = comp

    md = _render_markdown(report)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(args.output + ".md", "w", encoding="utf-8") as f:
            f.write(md)
        print(f"报告已写入:{args.output}.json / {args.output}.md")
    if args.output_dir:
        _write_output_dir(args.output_dir, report, md)
        print(f"结果目录已写入:{args.output_dir}")
    print("\n" + md)


if __name__ == "__main__":
    main()
