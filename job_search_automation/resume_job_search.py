from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd
from pypdf import PdfReader

try:
    from jobspy import scrape_jobs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'python-jobspy'. Run install_jobspy.ps1 first."
    ) from exc


SKILL_VOCAB = {
    "sql",
    "nosql",
    "mongodb",
    "elasticsearch",
    "oracle agile plm",
    "baan erp",
    "erp",
    "plm",
    "rpa",
    "data analysis",
    "data validation",
    "data integrity",
    "master data management",
    "kpi reporting",
    "metrics reporting",
    "business intelligence",
    "process automation",
    "workflow automation",
    "supply chain operations",
    "bom management",
    "engineering change orders",
    "product lifecycle support",
    "new product introduction",
    "stakeholder communication",
    "cross-functional collaboration",
    "quality assurance",
    "audit documentation",
    "sop",
    "fmea",
    "crm data enrichment",
}

ROLE_EXPANSIONS = {
    "data analyst": [
        "data analyst",
        "reporting analyst",
        "mis analyst",
        "business data analyst",
        "operations data analyst",
    ],
    "scm analyst": [
        "supply chain analyst",
        "scm analyst",
        "operations analyst",
        "procurement analyst",
        "inventory analyst",
        "master data analyst",
    ],
}

ALLOWED_TITLE_TERMS = {
    "analyst",
    "specialist",
    "associate",
    "executive",
    "coordinator",
    "planner",
    "consultant",
}

REJECT_TITLE_TERMS = {
    "developer",
    "engineer",
    "programmer",
    "architect",
    "full stack",
    "frontend",
    "backend",
    "machine learning",
    "scientist",
    "devops",
    "sde",
    "software",
    "ios",
    "android",
    "react",
    "node",
    "java",
    "python developer",
}


@dataclass
class ResumeProfile:
    raw_text: str
    matched_skills: list[str]
    location_hint: str
    summary_keywords: list[str]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_resume_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return re.sub(r"\s+", " ", text).strip()


def build_resume_profile(text: str, fallback_location: str) -> ResumeProfile:
    text_lower = text.lower()
    matched_skills = sorted(skill for skill in SKILL_VOCAB if skill in text_lower)
    summary_keywords = [
        "data analysis",
        "reporting",
        "business intelligence",
        "supply chain",
        "master data",
        "process improvement",
        "stakeholder management",
        "quality compliance",
    ]
    return ResumeProfile(
        raw_text=text,
        matched_skills=matched_skills,
        location_hint=fallback_location,
        summary_keywords=[item for item in summary_keywords if item in text_lower],
    )


def expanded_roles(base_roles: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered_roles: list[str] = []
    for role in base_roles:
        for item in ROLE_EXPANSIONS.get(role.lower(), [role]):
            lowered = item.lower()
            if lowered not in seen:
                seen.add(lowered)
                ordered_roles.append(item)
    return ordered_roles


def build_google_query(role: str, config: dict[str, Any]) -> str:
    location = config["location"]
    exclusions = " ".join(f"-{term}" for term in config["exclude_keywords"])
    return (
        f'"{role}" jobs in {location} since yesterday {exclusions}'.strip()
    )


def scrape_role(role: str, config: dict[str, Any]) -> pd.DataFrame:
    jobs = scrape_jobs(
        site_name=config["sites"],
        search_term=role,
        google_search_term=build_google_query(role, config),
        location=config["location"],
        results_wanted=config["results_wanted_per_site"],
        hours_old=config["hours_old"],
        country_indeed=config["country_indeed"],
        linkedin_fetch_description=False,
        description_format="markdown",
        verbose=1,
    )
    frame = pd.DataFrame(jobs)
    if frame.empty:
        return frame
    frame["searched_role"] = role
    return frame


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def title_is_allowed(title: str) -> bool:
    title_lower = title.lower()
    if any(term in title_lower for term in REJECT_TITLE_TERMS):
        return False
    return any(term in title_lower for term in ALLOWED_TITLE_TERMS)


def keyword_overlap_score(text: str, keywords: list[str], weight: int) -> int:
    text_lower = text.lower()
    return sum(weight for keyword in keywords if keyword.lower() in text_lower)


def compute_match_score(row: pd.Series, profile: ResumeProfile, config: dict[str, Any]) -> int:
    title = normalize_text(row.get("title"))
    description = normalize_text(row.get("description"))
    searched_role = normalize_text(row.get("searched_role"))
    combined_text = f"{title} {description}"

    score = 0
    if title_is_allowed(title):
        score += 25
    score += keyword_overlap_score(title, config["roles"], 12)
    score += keyword_overlap_score(title, config["preferred_title_keywords"], 8)
    score += keyword_overlap_score(combined_text, profile.matched_skills, 2)
    score += keyword_overlap_score(combined_text, profile.summary_keywords, 5)
    score += keyword_overlap_score(combined_text, config["domain_keywords"], 4)
    if searched_role and searched_role.lower() in title.lower():
        score += 10
    if "bengaluru" in normalize_text(row.get("location")).lower():
        score += 5
    return score


def prepare_jobs_dataframe(
    frame: pd.DataFrame,
    profile: ResumeProfile,
    config: dict[str, Any],
    run_timestamp: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    renamed = frame.copy()
    renamed.columns = [str(column).lower() for column in renamed.columns]

    for column in [
        "title",
        "company",
        "location",
        "job_url",
        "description",
        "site",
        "date_posted",
        "job_type",
        "searched_role",
    ]:
        if column not in renamed.columns:
            renamed[column] = ""

    renamed["title"] = renamed["title"].map(normalize_text)
    renamed["description"] = renamed["description"].map(normalize_text)
    renamed["location"] = renamed["location"].map(normalize_text)
    renamed["job_url"] = renamed["job_url"].map(normalize_text)
    renamed["searched_at"] = run_timestamp
    renamed["resume_skills_used"] = ", ".join(profile.matched_skills)
    renamed["match_score"] = renamed.apply(
        lambda row: compute_match_score(row, profile, config),
        axis=1,
    )
    renamed["title_allowed"] = renamed["title"].map(title_is_allowed)
    renamed = renamed[renamed["title_allowed"]]
    renamed = renamed[renamed["match_score"] >= config["match_threshold"]]
    renamed = renamed.drop_duplicates(subset=["job_url", "title", "company"], keep="first")
    renamed = renamed.sort_values(
        by=["match_score", "date_posted"],
        ascending=[False, False],
        na_position="last",
    )
    return renamed


def state_path(output_dir: Path) -> Path:
    return output_dir / "state_seen_jobs.csv"


def load_seen_jobs(output_dir: Path) -> set[str]:
    path = state_path(output_dir)
    if not path.exists():
        return set()
    previous = pd.read_csv(path)
    return set(previous["job_key"].dropna().astype(str).tolist())


def make_job_key(row: pd.Series) -> str:
    job_url = normalize_text(row.get("job_url"))
    title = normalize_text(row.get("title")).lower()
    company = normalize_text(row.get("company")).lower()
    return job_url or f"{title}|{company}"


def split_new_and_all_jobs(frame: pd.DataFrame, seen_jobs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()

    all_jobs = frame.copy()
    all_jobs["job_key"] = all_jobs.apply(make_job_key, axis=1)
    new_jobs = all_jobs[~all_jobs["job_key"].isin(seen_jobs)].copy()
    return new_jobs, all_jobs


def update_seen_jobs(output_dir: Path, all_jobs: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(all_jobs["job_key"].dropna().astype(str).tolist()))
    pd.DataFrame({"job_key": keys}).to_csv(state_path(output_dir), index=False)


def export_results(
    output_dir: Path,
    profile: ResumeProfile,
    config: dict[str, Any],
    new_jobs: pd.DataFrame,
    all_jobs: pd.DataFrame,
    run_timestamp: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / "job_matches_latest.xlsx"
    snapshot_path = history_dir / f"job_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    summary = pd.DataFrame(
        [
            {
                "searched_at": run_timestamp,
                "resume_path": config["resume_path"],
                "location": config["location"],
                "roles": ", ".join(config["roles"]),
                "sites": ", ".join(config["sites"]),
                "matched_resume_skills": ", ".join(profile.matched_skills),
                "new_jobs_found": len(new_jobs),
                "all_matching_jobs_found": len(all_jobs),
            }
        ]
    )

    for target in [latest_path, snapshot_path]:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            summary.to_excel(writer, index=False, sheet_name="run_summary")
            new_jobs.to_excel(writer, index=False, sheet_name="new_matches")
            all_jobs.to_excel(writer, index=False, sheet_name="all_matches")
    return latest_path, snapshot_path


def run_search(config_path: Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    resume_path = Path(config["resume_path"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = build_resume_profile(
        extract_resume_text(resume_path),
        fallback_location=config["location"],
    )

    roles = expanded_roles(config["roles"])
    collected_frames: list[pd.DataFrame] = []
    for role in roles:
        try:
            frame = scrape_role(role, config)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] failed for role '{role}': {exc}")
            continue
        if not frame.empty:
            collected_frames.append(frame)

    if collected_frames:
        raw_jobs = pd.concat(collected_frames, ignore_index=True)
    else:
        raw_jobs = pd.DataFrame()

    all_matching_jobs = prepare_jobs_dataframe(raw_jobs, profile, config, run_timestamp)
    seen_jobs = load_seen_jobs(output_dir)
    new_jobs, all_jobs = split_new_and_all_jobs(all_matching_jobs, seen_jobs)
    if not all_jobs.empty:
        update_seen_jobs(output_dir, all_jobs)
    latest_path, snapshot_path = export_results(
        output_dir=output_dir,
        profile=profile,
        config=config,
        new_jobs=new_jobs,
        all_jobs=all_jobs,
        run_timestamp=run_timestamp,
    )
    print(f"Latest workbook: {latest_path}")
    print(f"Snapshot workbook: {snapshot_path}")
    print(f"New jobs: {len(new_jobs)} | All matching jobs: {len(all_jobs)}")
    return latest_path, snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for resume-matched analyst jobs.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.json"),
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_search(Path(args.config).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
