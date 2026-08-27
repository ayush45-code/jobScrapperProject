# Job Materializer

A small, agentic terminal tool that uses LLMs (via OpenRouter) to rank, display, and save job postings based on a mix of your resume and personal preferences.

---

## Features

- **Multi-Source Scraping**: Scrapes jobs concurrently using [JobSpy](https://github.com/BloopAI/JobSpy) from:
  - Indeed
  - ZipRecruiter
  - Google Jobs
  - LinkedIn (optional via `--with-linkedin`)
- **Experience Suitability Filtering**: Automatically filters out jobs that don't match the target experience range (default: 2 to 3 years) before scoring.
- **AI-Powered Rank & Score**: Uses OpenRouter (configured for the `x-ai/grok-4.3` model) to match jobs against your detailed resume and preferences, assigning a score (0-100%).
- **Interactive Terminal Display**: Rich terminal panel styling with colored matching indicators:
  - `Magenta` (>= 80% score)
  - `Yellow` (>= 60% score)
  - `Blue` (>= 40% score)
  - `White` (< 40% score)
- **Automatic State Management**: Saves state to track already-seen jobs and avoid duplicate scoring.
- **Auto-Save High Matches**: Appends high-scoring jobs (60% or higher) to `saved.txt` complete with metadata and the AI's reasoning.

---

## Installation & Setup

### 1. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Linux/macOS
source .venv/bin/activate  
# On Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set Up API Key
The tool uses an API key to communicate with OpenRouter. Define it in your environment or a `.env` file:
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

---

## Configuration

The script reads its settings from the `./config/` directory:

1. **`config/resume.txt`** (Plain Text Resume)
   - Store your plain text resume here.
   - *Note:* The file name is case-sensitive (`resume.txt` in lowercase). If it is missing, the script still runs, but AI scoring is disabled.

2. **`config/config.yaml`** (Scoring & Preferences)
   - Contains your personalized preferences:
     - `goals`: Roles you target (e.g., "Java Backend Developer").
     - `background`: Your skills and experience summary.
     - `pay`: Salary/compensation preferences.
     - `location`: Location preferences.
     - `evaluation_factors`: Detailed lists of technologies to prioritize (e.g., Spring Boot, Microservices, React.js) and exclusions to skip (e.g., Python-only, QA, .NET).
     - `min_score`: The minimum AI matching score (0-100) below which jobs will be skipped (default is `20`).

---

## State and Saved Output

- **Seen Jobs State**: Tracked in `.radar_state.json` to prevent re-scraping and duplicate processing of jobs.
- **Saved Results**: Jobs scoring **60% or above** are appended to `saved.txt` in the root directory.

---

## Command-Line Usage

```bash
python radar.py [options]
```

### Supported CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--search` | string | *Predefined Java roles* | Custom job search keyword. If omitted, queries multiple default Java backend keywords. |
| `--location` | string | `"India"` | Location target for scraping. |
| `--min-experience` | float | `2.0` | Minimum candidate experience required. |
| `--max-experience` | float | `3.0` | Target upper limit of experience required by the JD. |
| `--hours-old` | int | `168` (7 days) | Only fetch jobs posted within this many hours. |
| `--reset-state` | flag | *Disabled* | Resets the previously seen jobs list (deletes state file). |
| `--indeed-only` | flag | *Disabled* | Limit scraper to search *only* Indeed. |
| `--with-linkedin` | flag | *Disabled* | Enable LinkedIn job search in addition to default sources. |
| `--no-ai` | flag | *Disabled* | Runs the search pipeline but disables LLM scoring. |
| `--proxy` | string | `None` | Proxy URL for scrape requests (e.g., `http://user:pass@host:port`). |
| `--state` | string | `".radar_state.json"` | Path to the state tracking JSON file. |

### Usage Examples

#### Run default Java backend search (includes Indeed, ZipRecruiter, Google Jobs) in India:
```bash
python radar.py
```

#### Search for "Frontend React" jobs in USA (overriding default search/location):
```bash
python radar.py --search "frontend React" --location "USA"
```

#### Run without AI ranking/scoring (just print jobs):
```bash
python radar.py --no-ai
```

#### Search *only* Indeed and reset previously seen jobs:
```bash
python radar.py --indeed-only --reset-state
```

#### Include LinkedIn scraping:
```bash
python radar.py --with-linkedin
```
