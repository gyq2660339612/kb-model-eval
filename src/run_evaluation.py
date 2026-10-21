"""命令行入口:加载 BEIR 格式数据集,跑召回评估(+可选重排评估),输出 JSON + Markdown 报告。

示例:
    python -m src.run_evaluation --backend mock --rerank-backend mock
    python -m src.run_evaluation --backend openai --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-Embedding-8B --rerank-backend openai \
        --rerank-base-url http://127.0.0.1:8001 --rerank-model Qwen/Qwen3-Reranker-8B
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends import build_embedding_backend, build_reranker_backend
from src.data_loader import load_dataset
from src.eval_rerank import run_rerank_eval
from src.eval_retrieval import run_retrieval_eval

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_k_values(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


def _load_thresholds(path: str) -> dict:
    full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        return {}
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f).get("acceptance_thresholds", {})


def _check_thresholds(metrics: dict, thresholds: dict) -> list[dict]:
    rows = []
    for key, target in thresholds.items():
        if key in metrics:
            val = metrics[key]
            rows.append({"metric": key, "value": round(val, 4), "target": target,
                         "pass": val >= target})
    return rows


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
    for key, val in report["retrieval"]["metrics"]["metrics"].items():
        lines.append(f"| {key} | {val:.4f} |")
    perf = report["retrieval"]["performance"]
    lines += [
        "",
        "## 召回性能",
        "",
        f"- 文档数:{perf['num_documents']},查询数:{perf['num_queries']},向量维度:{perf['vector_dim']}",
        f"- 文档向量化:{perf['doc_embed_seconds']}s({perf['doc_throughput_per_sec']} docs/s)",
        f"- 查询平均延迟:{perf['query_avg_latency_ms']} ms(批处理口径)",
        "",
    ]
    if "rerank" in report:
        rr = report["rerank"]
        lines += [
            "## 重排序(Rerank)对比",
            "",
            "| 指标 | 重排前 | 重排后 | 变化 |",
            "|---|---|---|---|",
        ]
        for key in rr["after"]["metrics"]:
            b = rr["before"]["metrics"][key]
            a = rr["after"]["metrics"][key]
            d = rr["delta_pct"][key]
            dtxt = f"{d:+.2f}%" if d is not None else "—"
            lines.append(f"| {key} | {b:.4f} | {a:.4f} | {dtxt} |")
        rp = rr["performance"]
        lines += [
            "",
            f"- 重排深度 top-{rp['rerank_top_k']};平均延迟 {rp['avg_latency_ms']} ms/query,"
            f"P95 {rp['p95_latency_ms']} ms",
            "",
        ]
    if report.get("acceptance"):
        lines += ["## 验收门槛对照", "", "| 指标 | 实测 | 门槛 | 结论 |", "|---|---|---|---|"]
        for row in report["acceptance"]:
            lines.append(
                f"| {row['metric']} | {row['value']} | {row['target']} | "
                f"{'PASS' if row['pass'] else 'FAIL'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库 Embedding/Reranker 模型评估")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "sample"))
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--top-k", type=int, default=100, help="召回深度")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--backend", default="mock", choices=["mock", "openai", "st"])
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

    parser.add_argument("--config", default=os.path.join("configs", "eval_config.json"))
    parser.add_argument("--output", default=None, help="报告输出前缀(生成 .json 与 .md)")
    args = parser.parse_args()

    from src.backends import DEFAULT_QWEN3_INSTRUCTION
    instruction = DEFAULT_QWEN3_INSTRUCTION if args.instruction is None else (args.instruction or None)

    corpus, queries, qrels = load_dataset(args.data_dir)
    k_values = _parse_k_values(args.k_values)

    emb_backend = build_embedding_backend(
        args.backend, model=args.model, base_url=args.base_url, api_key=args.api_key,
        instruction=instruction, dimensions=args.dimensions, device=args.device,
        batch_size=args.batch_size,
    )
    print(f"[1/2] 召回评估:{emb_backend.name}(文档 {len(corpus)},查询 {len(queries)})")
    retrieval = run_retrieval_eval(
        emb_backend, corpus, queries, qrels,
        k_values=k_values, batch_size=args.batch_size, top_k=args.top_k,
    )

    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "data_dir": args.data_dir,
            "embedding_backend": emb_backend.name,
            "reranker_backend": None,
            "k_values": k_values,
            "top_k": args.top_k,
        },
        "retrieval": retrieval,
    }

    if args.rerank_backend != "none":
        rr_backend = build_reranker_backend(
            args.rerank_backend, model=args.rerank_model,
            base_url=args.rerank_base_url, api_key=args.api_key, device=args.device,
        )
        report["config"]["reranker_backend"] = rr_backend.name
        print(f"[2/2] 重排评估:{rr_backend.name}(top-{args.rerank_top_k})")
        report["rerank"] = run_rerank_eval(
            rr_backend, corpus, queries, qrels, retrieval["results"],
            k_values=k_values, rerank_top_k=args.rerank_top_k,
        )
        final_metrics = report["rerank"]["after"]["metrics"]
    else:
        final_metrics = retrieval["metrics"]["metrics"]

    thresholds = _load_thresholds(args.config)
    if thresholds:
        report["acceptance"] = _check_thresholds(final_metrics, thresholds)

    md = _render_markdown(report)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(args.output + ".md", "w", encoding="utf-8") as f:
            f.write(md)
        print(f"报告已写入:{args.output}.json / {args.output}.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
