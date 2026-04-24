# Swiss Army Agent (Streamlit + Gemini)

A beginner-friendly research dashboard powered by Gemini.

- `ARXIV` mode: finds recent papers and lets Gemini pick the most relevant ones.
- `GITHUB` mode: finds new repos and lets Gemini rank the most worthwhile ones.

## 1) Prerequisites

- Python 3.10+ recommended
- A Gemini API key (from Google AI Studio)
- A GitHub token (for `GITHUB` mode only)

## 2) Project files

- `agent.py` — main Streamlit app
- `.env` — secret keys
- `requirements.txt` — Python dependencies

## 3) Setup (first run)

### A. Create venv + install dependencies

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

**Windows (CMD)**

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
py -m pip install -r requirements.txt
```

If PowerShell blocks activation, run once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### B. Configure API keys

Open `.env` and set:

```env
GEMINI_API_KEY="YOUR_GEMINI_KEY"
GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
```

Notes:
- `GEMINI_API_KEY` is required for all modes.
- `GITHUB_TOKEN` is only required for `GITHUB` mode.
- `ARXIV` mode does not require an ArXiv API key.

## 4) Run the app

**macOS / Linux**

```bash
python3 -m streamlit run agent.py
```

**Windows**

```powershell
py -m streamlit run agent.py
```

Then open: `http://localhost:8501`

## 5) How to use

1. Pick settings at the top of `agent.py`:
   - `MODE = "ARXIV"` or `MODE = "GITHUB"`
   - `KEYWORDS = ["Agentic AI", "Python"]`
   - `MAX_RESULTS = 5` (rows shown in table)
   - `CANDIDATE_LIMIT = 8` (items Gemini analyzes in one batch)
2. Save the file.
3. Click **Run Agentic Loop** in the UI.
4. Watch the **Thought log** and review the results table.

## 6) Customize the project

Quick edits in `agent.py`:

- Change research topic:
  - `KEYWORDS = ["LLM Agents", "RAG"]`
- Switch workflow:
  - `MODE = "ARXIV"` (papers) or `MODE = "GITHUB"` (repos)
- Make demo faster:
  - lower `CANDIDATE_LIMIT` (e.g., `6`)
- Show more/less output:
  - adjust `MAX_RESULTS` (e.g., `3` to `8`)

## 7) Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED`**
  - Run: `pip install -r requirements.txt` (includes `certifi`)
- **Gemini key error**
  - Verify `GEMINI_API_KEY` in `.env` and restart Streamlit
- **GitHub auth/rate issues**
  - Check `GITHUB_TOKEN` validity and permissions

## 8) Security tip

Do not expose real keys publicly. If you publish the repo, replace `.env` values with placeholders first.
