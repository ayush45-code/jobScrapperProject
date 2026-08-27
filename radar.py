import argparse
import hashlib
import json
import os
import re
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import yaml
import pandas as pd
from openai import OpenAI
from jobspy import scrape_jobs
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# ============================================================
# CONFIG
# ============================================================

CONFIG_DIR = Path("./config")
RESUME_PATH = CONFIG_DIR / "resume.txt"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
STATE_FILE = ".radar_state.json"
SAVED_FILE = Path("./saved.txt")

console = Console()


# ============================================================
# TARGET JOB SEARCH
# ============================================================

DEFAULT_SEARCH_TERMS = [
    "Java Developer",
    "Java Backend Developer",
    "Java Software Engineer",
    "Spring Boot Developer",
    "Java Microservices Developer",
    "Java Full Stack Developer",
    "Java Full Stack React",
    "Java React Developer",
    "Full Stack Java Developer",
    "Java Backend Spring Boot",
]


# ============================================================
# JOB MODEL
# ============================================================

@dataclass
class Job:
    jid: str
    title: str
    link: str
    published: str
    published_dt: Optional[object]
    summary: str
    source: str
    company: str
    location: str


# ============================================================
# LOAD CONFIG / RESUME
# ============================================================

def load_resume() -> str:
    if not RESUME_PATH.exists():
        console.print(
            f"[yellow]No resume found at {RESUME_PATH}[/yellow]"
        )
        return ""

    return RESUME_PATH.read_text(encoding="utf-8")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        console.print(
            f"[yellow]No config found at {CONFIG_PATH}[/yellow]"
        )
        return {"min_score": 0}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"min_score": 0}


# ============================================================
# OPENROUTER / AI
# ============================================================

def init_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None

    try:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    except Exception as e:
        console.print(
            f"[red]Failed to initialize OpenRouter:[/red] {e}"
        )
        return None


def _to_score_percent(score) -> int:
    try:
        val = float(score)
    except Exception:
        return 0

    if val <= 1.0:
        val = val * 100.0

    return max(0, min(100, int(round(val))))


def score_job_with_ai(
    client: OpenAI,
    job: Job,
    resume: str,
    config: dict
) -> dict:

    goals = config.get("goals", "")
    background = config.get("background", "")
    pay = config.get("pay", "")
    location = config.get("location", "")
    evaluation_factors = config.get("evaluation_factors", "")

    current_time = datetime.now().strftime("%A %I:%M %p")

    prompt = f"""
Score this job 0.0-1.0.

IMPORTANT CANDIDATE PROFILE:
- Java Backend Developer
- 2+ years professional experience
- Java
- Spring Boot
- Microservices
- REST APIs
- Oracle / SQL
- React.js
- Java Full Stack is acceptable

TARGET ROLES:
- Java Developer
- Java Backend Developer
- Java Software Engineer
- Spring Boot Developer
- Java Microservices Developer
- Java Full Stack Developer
- Java + React Developer
- Full Stack Java Developer

DO NOT RECOMMEND:
- Python-only roles
- Node.js-only roles
- .NET / C# roles
- PHP roles
- Ruby roles
- Go roles
- C++ roles
- QA / Testing-only roles
- SDET roles
- DevOps-only roles
- Data Engineer roles
- Data Scientist roles
- Frontend-only React roles
- Mobile Developer roles

EXPERIENCE:
- Candidate has 2+ years.
- Prefer jobs targeting around 2 years.
- Jobs such as 2-3 or 2-4 years are acceptable.
- Jobs requiring 3+ years minimum should NOT be recommended.
- Jobs requiring 4+ or 5+ years should NOT be recommended.

SCORING:
0.0-0.2: Skip
0.2-0.4: Stretch
0.4-0.6: Solid
0.6-0.8: Strong
0.8-1.0: RARE

CURRENT TIME: {current_time}

Late night/weekend posts by big US companies = potentially ghost jobs.

RESUME:
{resume}

GOALS:
{goals}

SITUATION:
{background}

PAY:
{pay}

LOCATION:
{location}

FACTORS:
{evaluation_factors}

JOB:
{job.title} at {job.company}

Location:
{job.location}

DESCRIPTION:
{job.summary[:6000] if job.summary else 'No description'}

Return JSON:
- score: 0.0-1.0
- reasoning: LENGTH DEPENDS ON SCORE:
  * Under 0.4: One short sentence max
  * 0.4-0.6: Two short lines with +/-
  * 0.6-0.8: 2-3 lines with +/-
  * 0.8+: 3-4 lines with +/-
- should_apply: true/false

Keep each reasoning line under 60 chars.
"""

    try:
        response = client.chat.completions.create(
            model="x-ai/grok-4.3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "job_score",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "reasoning": {
                                "type": "string"
                            },
                            "should_apply": {
                                "type": "boolean"
                            }
                        },
                        "required": [
                            "score",
                            "reasoning",
                            "should_apply"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(
            response.choices[0].message.content
        )

        result["score"] = _to_score_percent(
            result.get("score", 0)
        )

        result["reasoning"] = str(
            result.get("reasoning", "") or ""
        )

        result["should_apply"] = bool(
            result.get("should_apply", False)
        )

        return result

    except Exception as e:
        console.print(
            f"[dim]AI scoring failed: {e}[/dim]"
        )

        return {
            "score": 0,
            "reasoning": "Scoring unavailable",
            "should_apply": False
        }


# ============================================================
# JOB ID
# ============================================================

def stable_job_id(link: str) -> str:
    return hashlib.sha256(
        link.strip().lower().encode("utf-8")
    ).hexdigest()

    
jid = stable_job_id(job_url)


# ============================================================
# STATE
# ============================================================

def load_state(path: str) -> Dict:

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}

                if "seen" not in data:
                    data["seen"] = []

                if "last_poll" not in data:
                    data["last_poll"] = {}

                if "saved" not in data:
                    data["saved"] = []

                return data

        except Exception:
            pass

    return {
        "seen": [],
        "last_poll": {},
        "saved": []
    }


def save_state(
    path: str,
    seen: List[str],
    last_poll: Dict,
    saved: List[str],
    max_seen: int
):

    if len(seen) > max_seen:
        seen = seen[-max_seen:]

    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seen": seen,
                "last_poll": last_poll,
                "saved": saved
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(tmp, path)


# ============================================================
# SAFE STRING
# ============================================================

def safe_str(val) -> str:

    if val is None:
        return ""

    if pd.isna(val):
        return ""

    return str(val).strip()


# ============================================================
# LOCATION
# ============================================================

def _row_location(row) -> str:

    loc_val = row.get("location")

    location_str = ""

    if isinstance(loc_val, dict):

        c = safe_str(loc_val.get("city"))
        s = safe_str(loc_val.get("state"))
        country = safe_str(loc_val.get("country"))

        parts = [
            p for p in [c, s, country]
            if p
        ]

        location_str = (
            ", ".join(parts)
            if parts
            else ""
        )

    else:
        location_str = safe_str(loc_val)

    if location_str:
        return location_str

    city = safe_str(row.get("city"))
    state = safe_str(row.get("state"))
    country = safe_str(row.get("country"))

    parts = [
        p for p in [city, state, country]
        if p
    ]

    return (
        ", ".join(parts)
        if parts
        else "Unknown"
    )


# ============================================================
# EXPERIENCE FILTER
# ============================================================



def experience_matches_target(
    text: str,
    candidate_min: float = 2.0,
    candidate_max: float = 3.0
) -> bool:
    """
    Determine whether the job is reasonably suitable
    for a candidate with 2+ years experience.

    We use an eligibility/overlap model instead of requiring
    the JD itself to be strictly below 3 years.

    Accepted:
        2 years
        2+ years
        2-3 years
        2-4 years
        1-3 years

    Rejected:
        3+ years
        3-5 years
        4+ years
        5+ years
    """

    if not text:
        return False

    requirements = extract_experience_requirements(text)

    if not requirements:
        # No explicit experience found.
        # Don't automatically reject because many job descriptions
        # put experience requirements in unusual formats.
        return True

    for minimum, maximum, has_plus in requirements:

        # If the JD requires 3+ years minimum,
        # a 2-year candidate is not eligible.
        if minimum >= candidate_max:
            continue

        # "2+ years" means minimum 2 and no upper bound.
        # Accept because the candidate meets the minimum.
        if has_plus:
            if minimum <= candidate_min:
                return True

        # Explicit range such as 2-4 or 1-3.
        elif maximum is not None:

            # Candidate range and JD range overlap.
            if (
                minimum < candidate_max
                and maximum >= candidate_min
            ):
                return True

    return False


# ============================================================
# JAVA / ROLE FILTER
# ============================================================

def is_target_java_role(job: Job) -> bool:
    """
    Strictly identify Java Backend / Java Full Stack roles.

    The job must have Java as an important technology
    and should be backend/full-stack oriented.
    """

    title = (job.title or "").lower()
    description = (job.summary or "").lower()

    combined = f"{title}\n{description}"

    # --------------------------------------------------------
    # Must contain Java somewhere
    # --------------------------------------------------------

    java_pattern = re.compile(
        r"\bjava\b|\bjava\s*17\b|\bjava\s*21\b|\bjava\s*8\b"
    )

    if not java_pattern.search(combined):
        return False

    # --------------------------------------------------------
    # Explicitly reject clearly unrelated job categories
    # --------------------------------------------------------

    excluded_title_patterns = [
        r"\bpython\b",
        r"\bnode\.?js\b",
        r"\bjavascript developer\b",
        r"\bfrontend developer\b",
        r"\bfront end developer\b",
        r"\breact developer\b",
        r"\bangular developer\b",
        r"\bvue developer\b",
        r"\b\.net developer\b",
        r"\bc# developer\b",
        r"\bphp developer\b",
        r"\bruby developer\b",
        r"\bgo developer\b",
        r"\bc\+\+ developer\b",
        r"\bdata engineer\b",
        r"\bdata scientist\b",
        r"\bdevops\b",
        r"\bsre\b",
        r"\bsite reliability\b",
        r"\bqa\b",
        r"\bquality assurance\b",
        r"\btester\b",
        r"\bsdet\b",
        r"\btest engineer\b",
        r"\bandroid developer\b",
        r"\bios developer\b",
        r"\bmobile developer\b",
        r"\bsalesforce\b",
    ]

    for pattern in excluded_title_patterns:

        if re.search(pattern, title):
            return False

    # --------------------------------------------------------
    # Java backend technologies
    # --------------------------------------------------------

    backend_terms = [
        "spring boot",
        "spring",
        "microservices",
        "rest api",
        "restful",
        "hibernate",
        "jpa",
        "backend",
        "back-end",
        "oracle",
        "mysql",
        "postgresql",
        "sql",
        "kafka",
        "web services",
        "servlets",
    ]

    has_backend = any(
        term in combined
        for term in backend_terms
    )

    # --------------------------------------------------------
    # React / Full Stack
    # --------------------------------------------------------

    has_react = bool(
        re.search(r"\breact(?:\.js|js)?\b", combined)
    )

    full_stack_terms = [
        "full stack",
        "full-stack",
        "fullstack",
    ]

    has_full_stack = any(
        term in combined
        for term in full_stack_terms
    )

    # --------------------------------------------------------
    # Title should ideally identify Java
    # --------------------------------------------------------

    title_has_java = bool(
        re.search(r"\bjava\b", title)
    )

    # Strong Java title = accept if Java exists.
    if title_has_java:
        return True

    # Full-stack title + Java + React = accept.
    if has_full_stack and has_react:
        return True

    # Backend title + strong Java backend stack = accept.
    backend_title_terms = [
        "backend engineer",
        "back-end engineer",
        "backend developer",
        "back-end developer",
        "software engineer",
        "software developer",
        "application developer",
        "server-side developer",
    ]

    has_backend_title = any(
        term in title
        for term in backend_title_terms
    )

    if has_backend_title and has_backend:
        return True

    return False


# ============================================================
# FETCH JOBS
# ============================================================

def fetch_jobs_from_source(
    source: str,
    search_term: str,
    location: str,
    results_wanted: int,
    hours_old: int,
    proxy: str = None
) -> List[Job]:

    jobs = []

    try:

        params = {
            "site_name": [source],
            "search_term": search_term,
            "location": location,
            "results_wanted": results_wanted,
        }

        # Dynamically add parameters based on what the installed jobspy.scrape_jobs accepts
        import inspect
        sig = inspect.signature(scrape_jobs)
        
        if "hours_old" in sig.parameters and hours_old is not None:
            params["hours_old"] = hours_old
        if "verbose" in sig.parameters:
            params["verbose"] = 0
        if "proxies" in sig.parameters and proxy:
            params["proxies"] = proxy
        elif "proxy" in sig.parameters and proxy:
            params["proxy"] = proxy

        if source == "google" and "google_search_term" in sig.parameters:
            params["google_search_term"] = (
                f"{search_term} jobs near "
                f"{location} since yesterday"
            )

        if source in ["indeed", "glassdoor"] and "country_indeed" in sig.parameters:
            params["country_indeed"] = "India"

        if source == "linkedin" and "linkedin_fetch_description" in sig.parameters:
            params["linkedin_fetch_description"] = True

        df = scrape_jobs(**params)

        if df is None or df.empty:
            return jobs

        for _, row in df.iterrows():

            job_url = safe_str(
                row.get("job_url")
            )

            title = safe_str(
                row.get("title")
            )

            if not job_url or not title:
                continue

            date_posted = row.get(
                "date_posted"
            )

            published_dt = None
            published_str = ""

            if (
                date_posted is not None
                and not pd.isna(date_posted)
            ):

                try:

                    if hasattr(
                        date_posted,
                        "strftime"
                    ):

                        published_dt = date_posted

                        published_str = (
                            published_dt.strftime(
                                "%Y-%m-%d"
                            )
                        )

                    else:

                        published_str = str(
                            date_posted
                        )

                except Exception:

                    published_str = (
                        str(date_posted)
                        if date_posted
                        else ""
                    )

            location_str = _row_location(row)

            company = safe_str(
                row.get("company")
            ) or "Unknown"

            description = safe_str(
                row.get("description")
            )

            jid = stable_job_id(
                job_url,
                ""
            )

            if any(
                j.jid == jid
                for j in jobs
            ):
                continue

            jobs.append(
                Job(
                    jid=jid,
                    title=title,
                    link=job_url,
                    published=published_str,
                    published_dt=published_dt,
                    summary=description,
                    source=source,
                    company=company,
                    location=location_str,
                )
            )

    except Exception as ex:

        console.print(
            f"[red]Error fetching from "
            f"{source}:[/red] {ex}"
        )

    return jobs


# ============================================================
# SITE NAME
# ============================================================

def get_site_name(url: str) -> str:

    if "indeed.com" in url:
        return "Indeed"

    if "linkedin.com" in url:
        return "LinkedIn"

    if "ziprecruiter.com" in url:
        return "ZipRecruiter"

    if "glassdoor.com" in url:
        return "Glassdoor"

    if "google.com" in url:
        return "Google"

    return "Link"


# ============================================================
# RENDER JOB CARD
# ============================================================

def render_job_card(
    job: Job,
    ai_reasoning: str = "",
    match_score: int = 0
):

    from rich.box import ROUNDED

    if match_score >= 80:
        color = "magenta"

    elif match_score >= 60:
        color = "yellow"

    elif match_score >= 40:
        color = "blue"

    else:
        color = "white"

    company = (
        job.company
        if job.company
        and job.company != "Unknown"
        else "Unknown"
    )

    location = (
        job.location
        if job.location
        else "Unknown"
    )

    site_name = get_site_name(
        job.link
    )

    max_width = min(
        76,
        console.width - 4
    )

    header_base = (
        f"{company} | "
        f"{job.title} | "
        f"{location} | "
        f"{site_name}"
    )

    if len(header_base) > max_width:

        available = (
            max_width
            - len(
                f"{company} |  | "
                f"{location} | "
                f"{site_name}"
            )
            - 3
        )

        title = (
            job.title[:available] + "..."
            if available > 0
            else job.title[:20] + "..."
        )

    else:
        title = job.title

    body = Text()

    body.append(
        company,
        style=f"bold {color}"
    )

    body.append(
        " | ",
        style="dim"
    )

    body.append(
        title,
        style=f"bold {color}"
    )

    body.append(
        " | ",
        style="dim"
    )

    body.append(
        location,
        style=color
    )

    body.append(
        " | ",
        style="dim"
    )

    body.append(
        site_name,
        style=(
            f"underline {color} "
            f"link {job.link}"
        )
    )

    body.append("\nLink: ", style="dim")
    body.append(job.link, style="cyan")

    if ai_reasoning:

        body.append("\n")

        for line in ai_reasoning.split("\n"):

            line = line.strip()

            if line.startswith("+"):

                body.append(
                    "+",
                    style=f"bold {color}"
                )

                body.append(
                    line[1:] + "\n",
                    style="white"
                )

            elif line.startswith("-"):

                body.append(
                    "-",
                    style=f"bold {color}"
                )

                body.append(
                    line[1:] + "\n",
                    style="white"
                )

            elif line:

                body.append(
                    line + "\n",
                    style="white"
                )

    title_text = (
        f"{match_score}%"
        if match_score > 0
        else None
    )

    panel = Panel(
        body,
        title=title_text,
        title_align="right",
        border_style=color,
        box=ROUNDED,
        padding=(0, 1),
        width=min(80, console.width),
    )

    console.print(panel)


# ============================================================
# TERMINAL HELPERS
# ============================================================

def hide_cursor():
    print(
        "\033[?25l",
        end="",
        flush=True
    )


def show_cursor():
    print(
        "\033[?25h",
        end="",
        flush=True
    )


def _status_write(s: str):

    width = shutil.get_terminal_size(
        (120, 20)
    ).columns

    if len(s) >= width:

        s = s[:max(0, width - 1)]

    print(
        "\033[2K\r" + s,
        end="\r",
        flush=True
    )


def _status_counts(
    found: Dict[str, int],
    saved: int
) -> str:

    m = (
        f"\033[1;35m"
        f"{found['magenta']:3d}"
        f"\033[0m"
    )

    y = (
        f"\033[1;33m"
        f"{found['yellow']:3d}"
        f"\033[0m"
    )

    b = (
        f"\033[34m"
        f"{found['blue']:3d}"
        f"\033[0m"
    )

    w = (
        f"\033[37m"
        f"{found['white']:3d}"
        f"\033[0m"
    )

    s = (
        f"\033[1m"
        f"{saved:3d}"
        f"\033[0m"
    )

    return (
        f"found: {m} | {y} | {b} | {w} "
        f" saved: {s}"
    )


# ============================================================
# SAVE JOB
# ============================================================

def append_saved_job(
    path: Path,
    job: Job,
    score: int,
    reasoning: str
):

    ts = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    company = (
        job.company or "Unknown"
    ).replace(
        "\n",
        " "
    ).strip()

    title = (
        job.title or ""
    ).replace(
        "\n",
        " "
    ).strip()

    location = (
        job.location or "Unknown"
    ).replace(
        "\n",
        " "
    ).strip()

    source = (
        job.source or ""
    ).replace(
        "\n",
        " "
    ).strip()

    link = (
        job.link or ""
    ).strip()

    reason = (
        reasoning or ""
    ).strip()

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    is_new = not path.exists()

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        if is_new:
            f.write(
                "Saved\n\n"
            )

        f.write(
            f"{ts} | "
            f"{score:>3d}% | "
            f"{company} — "
            f"{title} "
            f"({location}) "
            f"[{source}]\n"
        )

        f.write(
            f"{link}\n"
        )

        if reason:

            for line in reason.split("\n"):

                line = line.strip()

                if line:
                    f.write(
                        f"{line}\n"
                    )

        f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Job Radar - "
            "Java Backend / Java Full Stack"
        )
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    parser.add_argument(
        "--search",
        type=str,
        default="",
        help=(
            "Optional custom search. "
            "Default searches multiple Java roles."
        )
    )

    parser.add_argument(
        "--location",
        type=str,
        default="India",
        help="Job location. Default: India"
    )

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    parser.add_argument(
        "--min-experience",
        type=float,
        default=2.0,
        help=(
            "Minimum candidate experience. "
            "Default: 2"
        )
    )

    parser.add_argument(
        "--max-experience",
        type=float,
        default=3.0,
        help=(
            "Target upper experience boundary. "
            "Default: 3"
        )
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    parser.add_argument(
        "--state",
        default=STATE_FILE
    )

    parser.add_argument(
    "--reset-state",
    action="store_true",
    help="Reset previously seen jobs"
)

    parser.add_argument(
        "--max-seen",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--initial-limit",
        type=int,
        default=20
    )

    # --------------------------------------------------------
    # JOB FRESHNESS
    # --------------------------------------------------------

    parser.add_argument(
        "--hours-old",
        type=int,
        default=168,
        help=(
            "Only fetch jobs posted within "
            "this many hours. Default: 168"
        )
    )

    # --------------------------------------------------------
    # SOURCE INTERVALS
    # --------------------------------------------------------

    parser.add_argument(
        "--indeed-interval",
        type=int,
        default=3
    )

    parser.add_argument(
        "--zip-interval",
        type=int,
        default=5
    )

    parser.add_argument(
        "--google-interval",
        type=int,
        default=8
    )

    parser.add_argument(
        "--results",
        type=int,
        default=25
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    parser.add_argument(
        "--indeed-only",
        action="store_true"
    )

    parser.add_argument(
        "--with-linkedin",
        action="store_true"
    )

    parser.add_argument(
    "--linkedin-only",
    action="store_true",
    help="Fetch jobs only from LinkedIn"
)

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    parser.add_argument(
        "--proxy",
        type=str,
        default=None
    )

    parser.add_argument(
        "--no-ai",
        action="store_true"
    )

    parser.add_argument(
        "--dev",
        action="store_true"
    )

    args = parser.parse_args()

    # ========================================================
    # SEARCH TERMS
    # ========================================================

    if args.search:

        search_terms = [
            args.search
        ]

    else:

        search_terms = (
            DEFAULT_SEARCH_TERMS.copy()
        )

    # ========================================================
    # LOAD RESUME / CONFIG
    # ========================================================

    resume = load_resume()
    config = load_config()

    min_score = config.get(
        "min_score",
        0
    )

    # ========================================================
    # AI
    # ========================================================

    api_key = os.environ.get(
        "OPENROUTER_API_KEY",
        ""
    )

    client = None

    if not args.no_ai:

        if not api_key:

            console.print(
                "[yellow]"
                "No API key - "
                "AI scoring disabled"
                "[/yellow]\n"
            )

        elif not resume:

            console.print(
                "[yellow]"
                "No resume - "
                "AI scoring disabled"
                "[/yellow]\n"
            )

        else:

            client = init_client(
                api_key
            )

            if client:

                console.print(
                    "[green]"
                    "AI scoring enabled"
                    "[/green]\n"
                )

    # ========================================================
    # SOURCES
    # ========================================================

    if args.indeed_only:

        sources = [
            {
                "name": "indeed",
                "interval":
                    args.indeed_interval
            }
        ]

    elif args.linkedin_only:

        sources = [
            {
                "name": "linkedin",
                "interval": 30
            }
        ]

    else:

        sources = []
        import jobspy
        available_sites = set(x.lower() for x in jobspy.scrapers.Site.__members__.keys())

        if "indeed" in available_sites:
            sources.append({
                "name": "indeed",
                "interval": args.indeed_interval
            })
        if "zip_recruiter" in available_sites:
            sources.append({
                "name": "zip_recruiter",
                "interval": args.zip_interval
            })
        if "google" in available_sites:
            sources.append({
                "name": "google",
                "interval": args.google_interval
            })

        if args.with_linkedin and "linkedin" in available_sites:

            sources.append(
                {
                    "name": "linkedin",
                    "interval": 30
                }
            )

    # ========================================================
    # RESET STATE
    # ========================================================

    if args.reset_state and os.path.exists(
        args.state
    ):

        try:
            os.remove(
                args.state
            )

        except Exception:
            pass

    # ========================================================
    # STATE
    # ========================================================

    state = load_state(
        args.state
    )

    seen_list = state.get(
        "seen",
        []
    )

    seen = set(
        seen_list
    )

    last_poll = state.get(
        "last_poll",
        {}
    )

    saved_list = state.get(
        "saved",
        []
    )

    saved_set = set(
        saved_list
    )

    first_run = (
        len(seen_list) == 0
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    found = {
        "magenta": 0,
        "yellow": 0,
        "blue": 0,
        "white": 0
    }

    saved = 0

    # ========================================================
    # HEADER
    # ========================================================

    console.print(
        "[bold]Job Materializer[/bold]"
    )

    console.print(
        "Target: Java Backend / "
        "Java Full Stack / Java + React"
    )

    console.print(
        f"Location: {args.location}"
    )

    console.print(
        f"Experience: "
        f"{args.min_experience}+ years "
        f"(target < {args.max_experience})"
    )

    console.print(
        f"Freshness: "
        f"last {args.hours_old} hours"
    )

    console.print(
        "Search terms: "
        f"{len(search_terms)}"
    )

    if min_score > 0:

        console.print(
            f"Min AI score: "
            f"{min_score}%"
        )

    console.print(
        "Sources: "
        + ", ".join(
            s["name"]
            for s in sources
        )
    )

    console.print()

    if first_run:

        console.print(
            "[dim]"
            "Loading recent Java jobs..."
            "[/dim]\n"
        )

    console.print(
        "[dim]"
        "Ctrl+C to stop"
        "[/dim]\n"
    )

    dots_cycle = [
        "   ",
        ".  ",
        ".. ",
        "..."
    ]

    dots_i = 0
    last_queue = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        hide_cursor()

        while True:

            now = time.time()

            for source_cfg in sources:

                source_name = (
                    source_cfg["name"]
                )

                interval = (
                    source_cfg["interval"]
                )

                last = last_poll.get(
                    source_name,
                    0
                )

                if (
                    now - last < interval
                    and not first_run
                ):
                    continue

                # ------------------------------------------------
                # Poll source
                # ------------------------------------------------

                ts = datetime.now().strftime(
                    "%I:%M %p"
                ).lstrip("0")

                dots = dots_cycle[
                    dots_i
                ]

                dots_i = (
                    dots_i + 1
                ) % len(dots_cycle)

                _status_write(
                    f"{ts} Polling{dots}  "
                    f"In queue: "
                    f"{last_queue:4d}  "
                    f"{_status_counts(found, saved)}"
                )

                # ------------------------------------------------
                # Fetch ALL target Java search terms
                # ------------------------------------------------

                all_jobs = []

                for search_term in search_terms:

                    jobs = (
                        fetch_jobs_from_source(
                            source=source_name,
                            search_term=search_term,
                            location=args.location,
                            results_wanted=args.results,
                            hours_old=args.hours_old,
                            proxy=args.proxy,
                        )
                    )

                    all_jobs.extend(
                        jobs
                    )

                last_poll[
                    source_name
                ] = time.time()

                # ------------------------------------------------
                # Deduplicate results
                # ------------------------------------------------

                unique_jobs = {}

                for job in all_jobs:

                    if job.jid not in unique_jobs:

                        unique_jobs[
                            job.jid
                        ] = job

                jobs = list(
                    unique_jobs.values()
                )

                # ------------------------------------------------
                # First-run limit
                # ------------------------------------------------

                if first_run:

                    jobs = jobs[
                        :args.initial_limit
                    ]

                # ------------------------------------------------
                # HARD FILTER:
                # Java + Backend / Full Stack
                # ------------------------------------------------

                java_jobs = []

                for job in jobs:

                    if not is_target_java_role(
                        job
                    ):
                        continue

                    java_jobs.append(
                        job
                    )

                # ------------------------------------------------
                # HARD FILTER:
                # Experience
                # ------------------------------------------------

                target_jobs = []

                for job in java_jobs:

                    experience_text = (
                        f"{job.title}\n"
                        f"{job.summary}"
                    )

                    if not experience_matches_target(
                        experience_text,
                        candidate_min=
                            args.min_experience,
                        candidate_max=
                            args.max_experience
                    ):
                        continue

                    target_jobs.append(
                        job
                    )

                # ------------------------------------------------
                # Remove previously seen jobs
                # ------------------------------------------------

                pending_jobs = [
                    job
                    for job in target_jobs
                    if job.jid not in seen
                ]

                total_pending = len(
                    pending_jobs
                )

                last_queue = total_pending

                # ------------------------------------------------
                # Process jobs
                # ------------------------------------------------

                for i, job in enumerate(
                    pending_jobs
                ):

                    seen.add(
                        job.jid
                    )

                    seen_list.append(
                        job.jid
                    )

                    score = 0
                    reasoning = ""

                    # ------------------------------------------------
                    # AI SCORING
                    # ------------------------------------------------

                    if client and resume:

                        remaining = (
                            total_pending - i
                        )

                        ts = (
                            datetime.now()
                            .strftime("%I:%M %p")
                            .lstrip("0")
                        )

                        _status_write(
                            f"{ts} "
                            f"Scoring "
                            f"({remaining} pending):  "
                            f"{_status_counts(found, saved)}"
                        )

                        job_score = (
                            score_job_with_ai(
                                client,
                                job,
                                resume,
                                config
                            )
                        )

                        score = job_score.get(
                            "score",
                            0
                        )

                        reasoning = (
                            job_score.get(
                                "reasoning",
                                ""
                            )
                        )

                        if score < min_score:
                            continue

                    # ------------------------------------------------
                    # DISPLAY
                    # ------------------------------------------------

                    if score >= 80:

                        found[
                            "magenta"
                        ] += 1

                    elif score >= 60:

                        found[
                            "yellow"
                        ] += 1

                    elif score >= 40:

                        found[
                            "blue"
                        ] += 1

                    else:

                        found[
                            "white"
                        ] += 1

                    render_job_card(
                        job,
                        ai_reasoning=reasoning,
                        match_score=score
                    )

                    # ------------------------------------------------
                    # SAVE 60%+
                    # ------------------------------------------------

                    if (
                        score >= 60
                        and job.jid
                        not in saved_set
                    ):

                        append_saved_job(
                            SAVED_FILE,
                            job,
                            score,
                            reasoning
                        )

                        saved_set.add(
                            job.jid
                        )

                        saved_list.append(
                            job.jid
                        )

                        saved += 1

                # ------------------------------------------------
                # Save state
                # ------------------------------------------------

                save_state(
                    args.state,
                    seen_list,
                    last_poll,
                    saved_list,
                    args.max_seen
                )

            if first_run:

                first_run = False

            time.sleep(0.5)

    except KeyboardInterrupt:

        _status_write("")

        show_cursor()

        console.print(
            "\n[dim]Stopped[/dim]"
        )

        save_state(
            args.state,
            seen_list,
            last_poll,
            saved_list,
            args.max_seen
        )

    finally:

        show_cursor()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()