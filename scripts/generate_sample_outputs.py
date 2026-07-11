from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.screening_pipeline import ScreeningPipeline
from src.utils import export_candidates_to_csv, export_candidates_to_json


def main() -> None:
    job_description_path = ROOT / "data" / "job_description" / "ai_ml_engineer_jd.txt"
    resumes_dir = ROOT / "data" / "sample_resumes"
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    job_description = job_description_path.read_text(encoding="utf-8")
    resume_files = sorted(resumes_dir.glob("*.txt"))

    pipeline = ScreeningPipeline(job_description=job_description, resume_dir=resumes_dir)
    results, failures = pipeline.run(resume_files)

    export_candidates_to_csv(results, outputs_dir / "ranked_candidates.csv")
    export_candidates_to_json(results, outputs_dir / "ranked_candidates.json")

    print(f"Processed {len(results)} candidates")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure['filename']}: {failure['error']}")


if __name__ == "__main__":
    main()
