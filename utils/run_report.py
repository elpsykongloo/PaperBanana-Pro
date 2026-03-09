"""Helpers for summarizing pipeline run results."""

from __future__ import annotations

from typing import Any

from utils.pipeline_state import collect_parse_error_round_keys


def _has_rendered_output(result: dict[str, Any]) -> bool:
    eval_image_field = result.get("eval_image_field")
    if isinstance(eval_image_field, str) and result.get(eval_image_field):
        return True

    for key, value in result.items():
        if key.endswith("_base64_jpg") and value:
            return True
    return False


def _collect_parse_error_rounds(result: dict[str, Any]) -> list[str]:
    return collect_parse_error_round_keys(result)


def build_result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    failed_candidates = []
    missing_render_candidates = []
    parse_error_candidates = []

    for idx, result in enumerate(results):
        candidate_id = result.get("candidate_id", idx)
        if result.get("status") == "failed":
            failed_candidates.append(candidate_id)
        if not _has_rendered_output(result):
            missing_render_candidates.append(candidate_id)
        parse_error_rounds = _collect_parse_error_rounds(result)
        if parse_error_rounds:
            parse_error_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "round_keys": parse_error_rounds,
                }
            )

    return {
        "total_candidates": total,
        "successful_candidates": total - len(failed_candidates),
        "failed_candidates": len(failed_candidates),
        "failed_candidate_ids": failed_candidates,
        "rendered_candidates": total - len(missing_render_candidates),
        "missing_render_candidates": missing_render_candidates,
        "parse_error_candidates": parse_error_candidates,
    }


def build_failure_manifest(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failure_items = []
    for idx, result in enumerate(results):
        candidate_id = result.get("candidate_id", idx)
        parse_error_rounds = _collect_parse_error_rounds(result)

        if result.get("status") == "failed":
            failure_items.append(
                {
                    "candidate_id": candidate_id,
                    "type": "pipeline_failure",
                    "error": result.get("error", "Unknown error"),
                    "error_detail": result.get("error_detail", ""),
                }
            )
            continue

        if parse_error_rounds:
            failure_items.append(
                {
                    "candidate_id": candidate_id,
                    "type": "critic_parse_error",
                    "round_keys": parse_error_rounds,
                }
            )

        if not _has_rendered_output(result):
            failure_items.append(
                {
                    "candidate_id": candidate_id,
                    "type": "missing_render",
                    "eval_image_field": result.get("eval_image_field"),
                }
            )

    return failure_items
