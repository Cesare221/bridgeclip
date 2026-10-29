import json
import sys
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from clip_engine.services.local_planning import generate_candidates


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: local_candidates.py <transcript.json> <candidates.json>")

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Transcript JSON must be an object")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Transcript JSON must contain a segments array")

    candidates = generate_candidates(
        segments,
        source_duration_seconds=data.get("duration_seconds", data.get("video_duration_seconds")),
    )
    destination.write_text(
        json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Sentence-bounded candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
