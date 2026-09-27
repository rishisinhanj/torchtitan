#!/usr/bin/env python3
"""Convert torchtitan RL structured_logs/*.jsonl into a single Perfetto-loadable
chrome trace, so trainer/rollout/generation phases can be viewed as parallel
timeline tracks on one shared clock (all actors log epoch-microsecond
timestamps, and a single-node run shares one host clock).

Usage: python3 structured_logs_to_timeline.py <structured_logs_dir> <out.json>
"""
import json
import os
import sys
from collections import defaultdict


def actor_label(first_event: dict) -> str:
    source = first_event.get("source", "unknown")
    rank = first_event.get("global_rank", 0)
    return f"{source}[rank{rank}]"


def convert(logs_dir: str):
    trace_events = []
    pid_of = {}
    next_pid = [1]

    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(logs_dir, fname)
        # open_stack[(span_base, task_name)] -> list of start events (stack, for nesting/reentrancy)
        open_stack = defaultdict(list)
        pid = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if pid is None:
                    label = actor_label(d)
                    if label not in pid_of:
                        pid_of[label] = next_pid[0]
                        next_pid[0] += 1
                        trace_events.append(
                            {
                                "ph": "M",
                                "pid": pid_of[label],
                                "name": "process_name",
                                "args": {"name": label},
                            }
                        )
                    pid = pid_of[label]

                log_type = d.get("log_type")
                name = d.get("log_type_name") or ""
                if log_type != "event":
                    continue
                if name.endswith("_start"):
                    base = name[: -len("_start")]
                    open_stack[(base, d.get("task_name"))].append(d)
                elif name.endswith("_end"):
                    base = name[: -len("_end")]
                    key = (base, d.get("task_name"))
                    stack = open_stack.get(key)
                    if not stack:
                        continue
                    start_d = stack.pop()
                    ts_start = start_d["time_us"]
                    ts_end = d["time_us"]
                    dur = max(ts_end - ts_start, 1)
                    trace_events.append(
                        {
                            "ph": "X",
                            "pid": pid,
                            "tid": 1,
                            "ts": ts_start,
                            "dur": dur,
                            "name": base,
                        }
                    )
    return trace_events


def main():
    logs_dir, out_path = sys.argv[1], sys.argv[2]
    events = convert(logs_dir)
    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(f"wrote {len(events)} events to {out_path}")


if __name__ == "__main__":
    main()
