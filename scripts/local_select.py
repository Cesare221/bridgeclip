import asyncio
import json
import sys
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from clip_engine.services.local_planning import LocalPlanningService


async def select(candidates):
    planner = LocalPlanningService()
    try:
        return await planner.select_candidates(candidates)
    finally:
        await planner.close()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: local_select.py <candidates.json> <clips.json>")

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Candidate JSON must be an object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate JSON must contain a candidates array")

    clips = asyncio.run(select(candidates))
    destination.write_text(
        json.dumps({"clips": clips}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Selected clips: {len(clips)}")


if __name__ == "__main__":
    main()
