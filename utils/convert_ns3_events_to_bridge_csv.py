import argparse
import csv
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ns-3 packet event CSV into bridge link-trace CSV format."
    )
    parser.add_argument("--input", required=True, help="Input ns-3 events CSV path")
    parser.add_argument("--output", default="logs/ns3_link_trace.csv", help="Output bridge CSV path")
    parser.add_argument("--num-drones", type=int, required=True, help="Number of UAV nodes")
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Bucket size in seconds")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for output steps. If omitted, inferred from input",
    )
    parser.add_argument(
        "--default-disconnected-delay-ms",
        type=float,
        default=300.0,
        help="Delay value used for disconnected links",
    )
    return parser.parse_args()


def _pick(row, candidates, required=False, default=None):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    if required:
        raise KeyError(f"Missing required columns among: {candidates}")
    return default


def _event_to_delivery_tag(event_text):
    v = str(event_text).strip().lower()
    if v in {"tx", "send", "enqueue", "s"}:
        return "tx"
    if v in {"rx", "recv", "receive", "r"}:
        return "rx"
    if v in {"drop", "d", "lost"}:
        return "drop"
    return "other"


def load_events(input_path, num_drones, step_seconds):
    # keyed by (step, i, j)
    stats = {}
    max_step = 1

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_s = float(_pick(row, ["time", "time_s", "t", "timestamp"], required=True))
            src = int(_pick(row, ["src", "source", "src_id", "from", "i"], required=True))
            dst = int(_pick(row, ["dst", "dest", "dst_id", "to", "j"], required=True))
            if src < 0 or src >= num_drones or dst < 0 or dst >= num_drones:
                continue

            event = _event_to_delivery_tag(_pick(row, ["event", "ev", "type", "state"], default="rx"))
            delay_ms = _pick(row, ["delay_ms", "delay", "latency_ms", "e2e_delay_ms"], default="")
            hop = _pick(row, ["hop", "hops", "hop_count"], default="")

            step = int(math.floor(time_s / step_seconds)) + 1
            max_step = max(max_step, step)

            key = (step, src, dst)
            bucket = stats.setdefault(
                key,
                {
                    "tx": 0,
                    "rx": 0,
                    "drop": 0,
                    "delay_sum": 0.0,
                    "delay_count": 0,
                    "hop_sum": 0.0,
                    "hop_count": 0,
                },
            )

            if event == "tx":
                bucket["tx"] += 1
            elif event == "rx":
                bucket["rx"] += 1
            elif event == "drop":
                bucket["drop"] += 1

            if delay_ms not in (None, ""):
                d = float(delay_ms)
                if d >= 0.0:
                    bucket["delay_sum"] += d
                    bucket["delay_count"] += 1

            if hop not in (None, ""):
                h = float(hop)
                if h >= 0.0:
                    bucket["hop_sum"] += h
                    bucket["hop_count"] += 1

    return stats, max_step


def main():
    args = parse_args()
    stats, inferred_max_step = load_events(args.input, args.num_drones, args.step_seconds)
    max_steps = args.max_steps if args.max_steps is not None else inferred_max_step

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "i", "j", "connected", "delay_ms", "delivery", "hop"])

        for step in range(1, max_steps + 1):
            for i in range(args.num_drones):
                for j in range(args.num_drones):
                    if i == j:
                        writer.writerow([step, i, j, 1, "0.00", "1.0000", "0.0"])
                        continue

                    b = stats.get((step, i, j))
                    if b is None:
                        connected = 0
                        delay_ms = args.default_disconnected_delay_ms
                        delivery = 0.0
                        hop = 99.0
                    else:
                        tx = b["tx"]
                        rx = b["rx"]
                        drop = b["drop"]

                        total_attempt = tx + rx + drop
                        if tx > 0:
                            delivery = rx / tx
                        elif total_attempt > 0:
                            # fallback when logs only have rx/drop events
                            delivery = rx / total_attempt
                        else:
                            delivery = 0.0

                        connected = 1 if rx > 0 else 0
                        if b["delay_count"] > 0:
                            delay_ms = b["delay_sum"] / b["delay_count"]
                        else:
                            delay_ms = args.default_disconnected_delay_ms if connected == 0 else 50.0

                        if b["hop_count"] > 0:
                            hop = b["hop_sum"] / b["hop_count"]
                        else:
                            hop = 1.0 if connected == 1 else 99.0

                    writer.writerow(
                        [
                            step,
                            i,
                            j,
                            connected,
                            f"{delay_ms:.2f}",
                            f"{max(0.0, min(1.0, delivery)):.4f}",
                            f"{hop:.1f}",
                        ]
                    )

    print(f"Converted ns-3 events to bridge CSV: {args.output}")
    print("Expected bridge columns: step,i,j,connected,delay_ms,delivery,hop")


if __name__ == "__main__":
    main()
