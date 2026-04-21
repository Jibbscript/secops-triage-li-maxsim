from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alert_triage.encoders.base import EncodedTokens
from alert_triage.evals.tier1_phase1 import _load_corpus, _load_queries
from alert_triage.evals.tier1_phase2 import _build_binary_docs, _build_context, _pad_to_binary_dim
from alert_triage.evals.tier1_phase3 import _load_text_map
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.binary_rerank import BinaryThenFP16RerankRetriever
from alert_triage.retrievers.hamming_udf import HammingUDFRetriever
from alert_triage.storage.in_memory import InMemoryTokenVectorStore
from alert_triage.triage import (
    EvidenceHit,
    InvestigationStepToolRuntime,
    JudgeRuntime,
    JSONTransport,
    MCPTerminalToolRuntime,
    ReplayJudge,
    ReplayTriageEngine,
    RuleBasedJudge,
    RuleBasedTriageEngine,
    TerminalToolRuntime,
    TriageRuntime,
    build_provider_runtimes,
    load_reasoning_targets,
    load_replay_runtime_fixture,
    summarize_audit,
    write_trace_jsonl,
)
from alert_triage.triage.reasoning import build_reasoning_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-4 tier-2 reasoning harness.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-trace", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rerank-depth", type=int, default=2)
    parser.add_argument("--runtime", choices=("local", "replay", "openai", "anthropic"), default="local")
    parser.add_argument("--runtime-fixture", type=Path)
    parser.add_argument("--llm-model", default=RuleBasedTriageEngine.default_model_id)
    parser.add_argument("--judge-model", default=RuleBasedJudge.default_model_id)
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-base-url")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--terminal-runtime", choices=("local", "mcp"), default="local")
    parser.add_argument("--mcp-server-command")
    parser.add_argument("--mcp-server-arg", action="append", default=[])
    parser.add_argument("--mcp-tool-name", default="propose_investigation_step")
    parser.add_argument("--mcp-startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--mcp-call-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def _build_binary_rerank_retriever(
    corpus_ids: list[str],
    docs: list[EncodedTokens],
    *,
    rerank_depth: int,
    threads: int,
) -> tuple[BinaryThenFP16RerankRetriever, object]:
    fp16_docs = [doc.require_fp16().astype(np.float32, copy=False) for doc in docs]
    calibrator, binary_docs = _build_binary_docs(fp16_docs)
    ctx, rowid_to_alert_id = _build_context(corpus_ids, binary_docs, threads=threads)
    binary = HammingUDFRetriever(ctx=ctx, rowid_to_alert_id=rowid_to_alert_id.get)

    vector_store = InMemoryTokenVectorStore()
    vector_store.upsert(
        corpus_ids,
        [
            EncodedTokens(fp16=doc_fp16, binary=doc_bin)
            for doc_fp16, doc_bin in zip(fp16_docs, binary_docs, strict=True)
        ],
    )
    retriever = BinaryThenFP16RerankRetriever(
        candidate_retriever=binary,
        vector_store=vector_store,
        prefilter_top_n=rerank_depth,
    )
    return retriever, calibrator


def _build_reasoning_runtimes(
    *,
    runtime: str,
    llm_model: str,
    judge_model: str,
    runtime_fixture: Path | None,
    terminal_runtime: str = "local",
    mcp_server_command: str | None = None,
    mcp_server_args: tuple[str, ...] = (),
    mcp_tool_name: str = "propose_investigation_step",
    mcp_startup_timeout_seconds: float = 5.0,
    mcp_call_timeout_seconds: float = 10.0,
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_base_url: str | None = None,
    timeout_seconds: float = 30.0,
    transport: JSONTransport | None = None,
) -> tuple[TriageRuntime, JudgeRuntime, TerminalToolRuntime]:
    terminal_tool = _build_terminal_tool_runtime(
        terminal_runtime=terminal_runtime,
        mcp_server_command=mcp_server_command,
        mcp_server_args=mcp_server_args,
        mcp_tool_name=mcp_tool_name,
        mcp_startup_timeout_seconds=mcp_startup_timeout_seconds,
        mcp_call_timeout_seconds=mcp_call_timeout_seconds,
    )
    if runtime == "local":
        return (
            RuleBasedTriageEngine(model_id=llm_model),
            RuleBasedJudge(model_id=judge_model),
            terminal_tool,
        )
    if runtime == "replay":
        if runtime_fixture is None:
            raise ValueError("runtime_fixture is required when runtime='replay'")
        fixture = load_replay_runtime_fixture(runtime_fixture)
        return (
            ReplayTriageEngine(fixture=fixture, model_id=llm_model),
            ReplayJudge(fixture=fixture, model_id=judge_model),
            terminal_tool,
        )
    if runtime == "openai":
        triager, judge = build_provider_runtimes(
            runtime=runtime,
            llm_model="openai:gpt-5" if llm_model == RuleBasedTriageEngine.default_model_id else llm_model,
            judge_model="openai:gpt-5" if judge_model == RuleBasedJudge.default_model_id else judge_model,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        return triager, judge, terminal_tool
    if runtime == "anthropic":
        triager, judge = build_provider_runtimes(
            runtime=runtime,
            llm_model="anthropic:claude-sonnet-4-5" if llm_model == RuleBasedTriageEngine.default_model_id else llm_model,
            judge_model="anthropic:claude-sonnet-4-5" if judge_model == RuleBasedJudge.default_model_id else judge_model,
            api_key=api_key,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        return triager, judge, terminal_tool
    raise ValueError(f"unsupported runtime: {runtime}")


def _build_terminal_tool_runtime(
    *,
    terminal_runtime: str,
    mcp_server_command: str | None,
    mcp_server_args: tuple[str, ...],
    mcp_tool_name: str,
    mcp_startup_timeout_seconds: float,
    mcp_call_timeout_seconds: float,
) -> TerminalToolRuntime:
    if terminal_runtime == "local":
        return InvestigationStepToolRuntime()
    if terminal_runtime == "mcp":
        if not mcp_server_command:
            raise ValueError("mcp_server_command is required when terminal_runtime='mcp'")
        return MCPTerminalToolRuntime(
            server_command=(mcp_server_command, *mcp_server_args),
            tool_name=mcp_tool_name,
            startup_timeout_seconds=mcp_startup_timeout_seconds,
            call_timeout_seconds=mcp_call_timeout_seconds,
        )
    raise ValueError(f"unsupported terminal_runtime: {terminal_runtime}")


def run_phase4_reasoning(
    fixture_dir: Path,
    out_json: Path,
    out_trace: Path,
    *,
    threads: int,
    rerank_depth: int,
    runtime: str = "local",
    runtime_fixture: Path | None = None,
    terminal_runtime: str = "local",
    mcp_server_command: str | None = None,
    mcp_server_args: tuple[str, ...] = (),
    mcp_tool_name: str = "propose_investigation_step",
    mcp_startup_timeout_seconds: float = 5.0,
    mcp_call_timeout_seconds: float = 10.0,
    llm_model: str,
    judge_model: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_base_url: str | None = None,
    timeout_seconds: float = 30.0,
    transport: JSONTransport | None = None,
) -> dict[str, object]:
    if threads < 1:
        raise ValueError("threads must be >= 1")

    corpus_ids, docs = _load_corpus(fixture_dir / "corpus_fp16.npz")
    query_ids, queries = _load_queries(fixture_dir / "queries_fp16.npz")
    query_text_map = _load_text_map(fixture_dir / "query_texts.json")
    corpus_text_map = _load_text_map(fixture_dir / "corpus_texts.json")
    targets = load_reasoning_targets(fixture_dir / "reasoning_targets.json")

    retriever, calibrator = _build_binary_rerank_retriever(
        corpus_ids,
        docs,
        rerank_depth=rerank_depth,
        threads=threads,
    )
    triager, judge, terminal_tool = _build_reasoning_runtimes(
        runtime=runtime,
        llm_model=llm_model,
        judge_model=judge_model,
        runtime_fixture=runtime_fixture,
        terminal_runtime=terminal_runtime,
        mcp_server_command=mcp_server_command,
        mcp_server_args=mcp_server_args,
        mcp_tool_name=mcp_tool_name,
        mcp_startup_timeout_seconds=mcp_startup_timeout_seconds,
        mcp_call_timeout_seconds=mcp_call_timeout_seconds,
        api_key=api_key,
        api_key_env=api_key_env,
        api_base_url=api_base_url,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    samples = []
    try:
        for query_id, query_fp16 in zip(query_ids, queries, strict=True):
            query_text = query_text_map[query_id]
            query_bin = calibrator.apply(_pad_to_binary_dim(query_fp16.astype(np.float32, copy=False)))
            hits = retriever.search(
                QueryBundle(query_bin=query_bin, query_fp16=query_fp16.astype(np.float32, copy=False)),
                k=3,
            )
            evidence = [
                EvidenceHit(
                    alert_id=hit.alert_id,
                    score=hit.score,
                    stage=hit.stage,
                    text=corpus_text_map[hit.alert_id],
                )
                for hit in hits
            ]
            samples.append(
                build_reasoning_sample(
                    query_id=query_id,
                    query_text=query_text,
                    evidence=evidence,
                    target=targets[query_id],
                    triager=triager,
                    judge=judge,
                    terminal_tool=terminal_tool,
                )
            )

        metrics = {
            metric: float(np.mean([sample.sample_metrics[metric] for sample in samples]) if samples else 0.0)
            for metric in ("action_validity", "evidence_grounding", "specificity", "calibration")
        }
        summary = summarize_audit(
            [sample.audit_records for sample in samples],
            expected_llm_names={triager.model_id, judge.model_id},
            expected_llm_roles={"triager", "judge"},
            expected_tool_names={terminal_tool.tool_name},
        )
        terminality_ok = all(
            sample.audit_records[-1].kind == "tool_call"
            and sample.audit_records[-1].name == terminal_tool.tool_name
            and sample.terminal_output_kind == "tool_call"
            and sample.terminal_tool == terminal_tool.tool_name
            for sample in samples
        )
        auditability_ok = summary.orphan_llm_calls == 0 and summary.orphan_tool_calls == 0

        report = {
            "experiment_id": "tier2-phase4",
            "retriever": "binary-then-fp16-rerank",
            "runtime": runtime,
            "sample_count": len(samples),
            "query_count": len(samples),
            "llm_model": triager.model_id,
            "judge_model": judge.model_id,
            "metrics": metrics,
            "terminal_output_kind": "tool_call",
            "terminal_tool": terminal_tool.tool_name,
            "orphan_llm_calls": summary.orphan_llm_calls,
            "orphan_tool_calls": summary.orphan_tool_calls,
            "audit_summary": summary.model_dump(mode="json"),
            "gate_results": {
                "g_agent_terminality": terminality_ok,
                "g_auditability": auditability_ok,
            },
            "samples": [sample.model_dump(mode="json") for sample in samples],
        }

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        write_trace_jsonl(out_trace, samples)
        return report
    finally:
        close = getattr(terminal_tool, "close", None)
        if callable(close):
            close()


def main() -> None:
    args = _parse_args()
    run_phase4_reasoning(
        args.fixture_dir,
        args.out_json,
        args.out_trace,
        threads=args.threads,
        rerank_depth=args.rerank_depth,
        runtime=args.runtime,
        runtime_fixture=args.runtime_fixture,
        terminal_runtime=args.terminal_runtime,
        mcp_server_command=args.mcp_server_command,
        mcp_server_args=tuple(args.mcp_server_arg),
        mcp_tool_name=args.mcp_tool_name,
        mcp_startup_timeout_seconds=args.mcp_startup_timeout_seconds,
        mcp_call_timeout_seconds=args.mcp_call_timeout_seconds,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        api_key_env=args.api_key_env,
        api_base_url=args.api_base_url,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
