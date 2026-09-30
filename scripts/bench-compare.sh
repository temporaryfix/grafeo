#!/usr/bin/env bash
# Compare named Criterion baselines. Reports locally unless posting is explicit.
# Usage: bench-compare.sh BASELINE CANDIDATE CONFIG [PR_NUMBER] [--post-comment]
# Requires: critcmp and Python 3.11+; gh only with --post-comment.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bench-compare.sh BASELINE CANDIDATE CONFIG [PR_NUMBER] [--post-comment]" >&2
    exit 2
fi
BASELINE="$1"
CANDIDATE="$2"
CONFIG="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
shift 3
PR_NUMBER=""
POST_COMMENT=false
for argument in "$@"; do
    if [[ "$argument" == --post-comment && "$POST_COMMENT" == false ]]; then
        POST_COMMENT=true
    elif [[ "$argument" =~ ^[0-9]+$ && -z "$PR_NUMBER" ]]; then
        PR_NUMBER="$argument"
    else
        echo "Unexpected argument: $argument" >&2
        exit 2
    fi
done
if [[ "$POST_COMMENT" == true ]]; then
    : "${PR_NUMBER:?--post-comment requires PR_NUMBER}"
    : "${GITHUB_REPOSITORY:?--post-comment requires GITHUB_REPOSITORY}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# Validate policy before invoking tools. Do not hide subprocess failure in a
# process substitution, which does not propagate its status to the caller.
python3 "$SCRIPT_DIR/parse-thresholds.py" "$CONFIG" > /dev/null

BENCH_TMP="$(mktemp -d)"
cleanup() {
    rm -f "$BENCH_TMP/base.json" "$BENCH_TMP/candidate.json" "$BENCH_TMP/report.md"
    rmdir "$BENCH_TMP"
}
trap cleanup EXIT

# Upstream documents --export NAME. Its JSON preserves full-precision means and
# baseline identity; human table ranks are rounded and relative to the fastest.
critcmp --export "$BASELINE" > "$BENCH_TMP/base.json"
critcmp --export "$CANDIDATE" > "$BENCH_TMP/candidate.json"

STATUS=0
python3 "$SCRIPT_DIR/parse-thresholds.py" "$CONFIG" --compare \
    "$BASELINE" "$CANDIDATE" "$BENCH_TMP/base.json" "$BENCH_TMP/candidate.json" \
    "${CARGO_TARGET_DIR:-target}/criterion" > "$BENCH_TMP/report.md" || STATUS=$?
# Only 0 and the comparator's dedicated blocking result (10) denote validated
# evidence. In particular, an unexpected interpreter failure (1) is not a
# threshold failure and must never display or publish partial output.
if [[ "$STATUS" -ne 0 && "$STATUS" -ne 10 ]]; then
    echo "Benchmark comparison failed (status $STATUS); no report published." >&2
    exit "$STATUS"
fi
# The marker is emitted last, after every input and the whole report validate.
# Require completion even when a subprocess reports a recognized result.
if [[ ! -s "$BENCH_TMP/report.md" ]] || \
    [[ "$(tail -n 1 "$BENCH_TMP/report.md")" != '<!-- grafeo-bench-comparison -->' ]]; then
    echo "Benchmark comparison did not produce a complete report." >&2
    exit 2
fi
cat "$BENCH_TMP/report.md"
if [[ "$STATUS" -eq 10 ]]; then
    STATUS=1
fi

if [[ "$POST_COMMENT" == true ]]; then
    COMMENT_ID="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
        --jq '[.[] | select(.body | contains("<!-- grafeo-bench-comparison -->")) | .id][0] // empty')"
    if [[ -n "$COMMENT_ID" ]]; then
        gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${COMMENT_ID}" \
            -F "body=@$BENCH_TMP/report.md"
    else
        gh pr comment "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --body-file "$BENCH_TMP/report.md"
    fi
fi
exit "$STATUS"
