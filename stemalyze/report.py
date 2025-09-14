from pathlib import Path
import json
from typing import Dict, List, Tuple
from .analysis import write_report_txt

def save_all(outdir: str, meta: Dict, bars: List[Tuple[int,float,float]], chords: List[Dict], bar_melody: Dict[int, List[Dict]]):
    out = Path(outdir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)

    # JSON dumps for reuse
    with open(out / "analysis" / "bars.json", "w", encoding="utf-8") as f:
        json.dump(bars, f, indent=2)
    with open(out / "analysis" / "chords.json", "w", encoding="utf-8") as f:
        json.dump(chords, f, indent=2)
    with open(out / "analysis" / "melody_by_bar.json", "w", encoding="utf-8") as f:
        json.dump({int(k): v for k,v in bar_melody.items()}, f, indent=2)

    # human-readable report
    report_path = out / "reports" / f"{Path(meta['source']).stem}_stemalyze_report.txt"
    write_report_txt(str(report_path), meta, bars, chords, bar_melody)
    return str(report_path)
