import json
import math
import statistics
from pathlib import Path

METRICS = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Relevancy",
    "Contextual Recall",
    "Contextual Precision",
]


def mean_and_2se(values: list[float]) -> tuple[float, float]:
    """Return (mean, 2 * standard_error) for a list of numeric scores."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if n < 2:
        return mean, 0.0
    se = statistics.stdev(values) / math.sqrt(n)
    return mean, 2.0 * se


def load_local_outputs() -> list[dict]:
    path = "pipeline_outputs_local.json" if Path("pipeline_outputs_local.json").exists() else "pipeline_outputs.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_pipeline(rows: list[dict]) -> dict:
    gen_times = [r["generation_time_ms"] for r in rows if r.get("generation_time_ms") is not None]
    tokens = [r["tokens_generated"] for r in rows if r.get("tokens_generated") is not None]
    empty_or_failed = sum(
        1
        for r in rows
        if not (r.get("actual_answer") or "").strip() or r.get("finish_reason") != "stop"
    )
    return {
        "n": len(rows),
        "median_gen_ms": statistics.median(gen_times) if gen_times else 0.0,
        "max_gen_ms": max(gen_times) if gen_times else 0.0,
        "avg_tokens": statistics.mean(tokens) if tokens else 0.0,
        "empty_or_failed": empty_or_failed,
    }


def main() -> None:
    local_eval = {r["id"]: r for r in json.loads(Path("eval_results_local.json").read_text(encoding="utf-8"))}
    cloud_eval = {r["id"]: r for r in json.loads(Path("eval_results_cloud.json").read_text(encoding="utf-8"))}
    common_ids = sorted(set(local_eval) & set(cloud_eval))

    print("| Metric | n | Local (mean ± 2×SE) | Cloud (mean ± 2×SE) | Cloud − Local (mean ± 2×SE) | Verdict |")
    print("|---|---:|---:|---:|---:|---|")

    for metric in METRICS:
        local_scores = []
        cloud_scores = []
        diffs = []
        for qid in common_ids:
            ls = local_eval[qid]["metrics"].get(metric, {}).get("score")
            cs = cloud_eval[qid]["metrics"].get(metric, {}).get("score")
            if ls is not None and cs is not None:
                local_scores.append(ls)
                cloud_scores.append(cs)
                diffs.append(cs - ls)

        n = len(diffs)
        l_mean, l_2se = mean_and_2se(local_scores)
        c_mean, c_2se = mean_and_2se(cloud_scores)
        d_mean, d_2se = mean_and_2se(diffs)

        if d_mean - d_2se <= 0.0 <= d_mean + d_2se:
            verdict = "no clear difference"
        elif d_mean > 0:
            verdict = "cloud higher"
        else:
            verdict = "local higher"

        print(
            f"| {metric} | {n} | {l_mean:.3f} ± {l_2se:.3f} | "
            f"{c_mean:.3f} ± {c_2se:.3f} | {d_mean:+.3f} ± {d_2se:.3f} | {verdict} |"
        )

    local_pipe = summarize_pipeline(load_local_outputs())
    cloud_pipe = summarize_pipeline(json.loads(Path("pipeline_outputs_cloud.json").read_text(encoding="utf-8")))

    print("\n**Runtime & Token Summary:**\n")
    print("| Run | Median Gen Time | Slowest Gen Time | Avg Completion Tokens | Empty / Failed |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| Local (`qwen3.5:4b-q4_K_M`) | {local_pipe['median_gen_ms']:.0f} ms | "
        f"{local_pipe['max_gen_ms']:.0f} ms | {local_pipe['avg_tokens']:.1f} | {local_pipe['empty_or_failed']} |"
    )
    print(
        f"| Cloud (`gpt-5.6-luna`) | {cloud_pipe['median_gen_ms']:.0f} ms | "
        f"{cloud_pipe['max_gen_ms']:.0f} ms | {cloud_pipe['avg_tokens']:.1f} | {cloud_pipe['empty_or_failed']} |"
    )


if __name__ == "__main__":
    main()
