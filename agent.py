# --- LIVE DEMO CONFIG ---
MODE = "ARXIV"                          # Options: "ARXIV", "GITHUB"
KEYWORDS = ["Agentic AI", "Python"]
MAX_RESULTS = 5
CANDIDATE_LIMIT = 8                     # Max items sent to Gemini per run
# ------------------------

import json
import os
import re
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import certifi
import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


GEMINI_MODEL_CANDIDATES = (
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview"
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

load_dotenv()


def _https_ssl_context() -> ssl.SSLContext:
    """Use certifi CA bundle (fixes macOS Python SSL: CERTIFICATE_VERIFY_FAILED)."""
    return ssl.create_default_context(cafile=certifi.where())


def _urlopen_https(req: Request, timeout: float) -> Any:
    return urlopen(req, timeout=timeout, context=_https_ssl_context())


def _is_placeholder_secret(value: str | None) -> bool:
    if not value or not value.strip():
        return True
    v = value.strip().strip('"').strip("'")
    return "REPLACE_WITH" in v.upper()


def env_gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def env_github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN")


def truncate_title(title: str, max_words: int = 10) -> str:
    words = title.split()
    if len(words) <= max_words:
        return title
    return " ".join(words[:max_words]) + "..."


def parse_json_from_gemini(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def generate_gemini_json(
    prompt: str,
    log_fn: Callable[[str], None],
) -> Any:
    last_error: Exception | None = None
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            gen_cfg: dict[str, Any] = {"temperature": 0.2}
            try:
                gen_cfg_json = {**gen_cfg, "response_mime_type": "application/json"}
                resp = model.generate_content(
                    prompt,
                    generation_config=gen_cfg_json,
                )
            except Exception:
                resp = model.generate_content(
                    prompt,
                    generation_config=gen_cfg,
                )
            if not resp.text:
                raise ValueError("Empty Gemini response")
            return parse_json_from_gemini(resp.text)
        except Exception as e:
            last_error = e
            log_fn(f"Model `{model_name}` failed ({e!s}); trying next…")
            continue
    raise RuntimeError(f"All Gemini models failed: {last_error}")


def fetch_arxiv_candidates(
    keywords: list[str],
    max_results: int,
    log_fn: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Query the ArXiv Atom API (no API key). Returns title, authors, abstract, published, url."""
    if not keywords:
        log_fn("No keywords configured for ArXiv search.")
        return []
    query_parts = [f"all:{quote_plus(k.strip())}" for k in keywords if k.strip()]
    search_query = "+AND+".join(query_parts)
    api_url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={search_query}"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    req = Request(
        api_url,
        headers={
            "User-Agent": "SwissArmyAgent/1.0 (Gemini demo; +https://arxiv.org/help/api)",
        },
    )
    try:
        with _urlopen_https(req, 60) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        raise RuntimeError(f"ArXiv API HTTP error: {e.code} — {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"ArXiv network error: {e.reason}") from e

    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise RuntimeError(f"ArXiv response was not valid XML: {e}") from e

    items: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)
        id_el = entry.find("atom:id", ATOM_NS)
        title = (title_el.text or "").strip().replace("\n", " ")
        abstract = (summary_el.text or "").strip().replace("\n", " ")
        published = (published_el.text or "").strip()
        paper_id = (id_el.text or "").strip()
        author_names: list[str] = []
        for author in entry.findall("atom:author", ATOM_NS):
            name_el = author.find("atom:name", ATOM_NS)
            if name_el is not None and name_el.text:
                author_names.append(name_el.text.strip())
        authors = ", ".join(author_names) if author_names else ""
        items.append(
            {
                "title": title,
                "abstract": abstract,
                "author": authors,
                "published_at": published,
                "url": paper_id,
            }
        )

    log_fn(f"Fetched {len(items)} ArXiv candidate(s).")
    return items


def fetch_github_candidates(
    token: str,
    keywords: list[str],
    per_page: int,
    log_fn: Callable[[str], None],
) -> list[dict[str, Any]]:
    since = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    q_terms = " ".join(keywords)
    q = f"{q_terms} created:>{since}"
    encoded = quote_plus(q)
    url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&order=desc&per_page={min(100, per_page)}"
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SwissArmyAgent/1.0",
        },
    )
    try:
        with _urlopen_https(req, 60) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        items_out: list[dict[str, Any]] = []
        for repo in data.get("items", []):
            items_out.append(
                {
                    "title": repo.get("full_name", repo.get("name", "")),
                    "description": repo.get("description") or "",
                    "author": (repo.get("owner") or {}).get("login", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "created_at": repo.get("created_at", ""),
                    "url": repo.get("html_url", ""),
                }
            )
        log_fn(
            f"Fetched {len(items_out)} GitHub repo(s) created after {since}."
        )
        return items_out
    except HTTPError as e:
        raise RuntimeError(
            f"GitHub API HTTP error: {e.code} — {e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"GitHub network error: {e.reason}") from e


def _extract_batch_selection_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("selections", "top", "results", "items", "selected"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _normalize_batch_selections(
    raw: list[dict[str, Any]],
    n_items: int,
    max_pick: int,
    source: Literal["arxiv", "github"],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in raw:
        try:
            idx = int(row.get("index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= n_items or idx in seen:
            continue
        seen.add(idx)
        reason = str(row.get("reason", "")).strip() or "—"
        entry: dict[str, Any] = {"index": idx, "reason": reason}
        if source == "github":
            try:
                ws = int(row.get("worth_score", 5))
                entry["worth_score"] = max(1, min(10, ws))
            except (TypeError, ValueError):
                entry["worth_score"] = 5
        out.append(entry)
        if len(out) >= max_pick:
            break
    return out


def _fallback_batch_selections(
    candidates: list[dict[str, Any]],
    source: Literal["arxiv", "github"],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(min(MAX_RESULTS, len(candidates))):
        e: dict[str, Any] = {
            "index": i,
            "reason": "Fallback: batch analysis unavailable.",
        }
        if source == "github":
            e["worth_score"] = 5
        out.append(e)
    return out


def batch_analyze_candidates(
    candidates: list[dict[str, Any]],
    keywords: list[str],
    *,
    source: Literal["arxiv", "github"],
    log_fn: Callable[[str], None],
) -> list[dict[str, Any]]:
    """
    One Gemini call: pick up to MAX_RESULTS indices from candidates (len up to CANDIDATE_LIMIT).
    Each dict: index, reason, and worth_score (GitHub only).
    On any failure, returns fallback picks (first MAX_RESULTS items).
    """
    n = len(candidates)
    if n == 0:
        return []

    blocks: list[str] = []
    for i, c in enumerate(candidates):
        if source == "arxiv":
            blocks.append(
                f"### Item {i}\n"
                f"Title: {c.get('title', '')}\n"
                f"Authors: {c.get('author', '')}\n"
                f"Abstract: {(c.get('abstract') or '')[:1200]}"
            )
        else:
            blocks.append(
                f"### Item {i}\n"
                f"Repository: {c.get('title', '')}\n"
                f"Description: {(c.get('description') or '')[:1200]}\n"
                f"Stars: {c.get('stars', 0)}"
            )
    catalog = "\n\n".join(blocks)

    if source == "arxiv":
        task = (
            f"These are arXiv papers. Pick the top entries most **relevant** to the keywords "
            f"(scientific match, not just keyword overlap). Return exactly **{MAX_RESULTS}** "
            f"picks (or fewer if there are fewer than {MAX_RESULTS} items), ordered best first."
        )
        json_shape = (
            f'{{"selections": [{{"index": <int 0-{n - 1}>, "reason": "<one short sentence>"}}, ...]}}'
        )
    else:
        task = (
            f"These are GitHub repositories. Pick the top entries most **worthwhile** to inspect "
            f"for someone researching the keywords. Return exactly **{MAX_RESULTS}** picks "
            f"(or fewer if there are fewer items), ordered best first. "
            f"Assign each pick a **worth_score** from 1–10."
        )
        json_shape = (
            f'{{"selections": [{{"index": <int 0-{n - 1}>, "worth_score": <int 1-10>, '
            f'"reason": "<one short sentence>"}}, ...]}}'
        )

    prompt = f"""You are an expert research assistant.

Research keywords: {json.dumps(keywords)}

Here are **{n}** candidates (indices **0** through **{n - 1}**). Analyze them **all in one pass**.

{catalog}

{task}

Return **JSON only**, no markdown. Shape:
{json_shape}

Rules: `index` must be unique and in [0, {n - 1}]. At most {MAX_RESULTS} objects in `selections`."""

    try:
        data = generate_gemini_json(prompt, log_fn)
        raw_rows = _extract_batch_selection_rows(data)
        normalized = _normalize_batch_selections(raw_rows, n, MAX_RESULTS, source)
        if not normalized:
            raise ValueError("No valid selections in model output")
        return normalized
    except Exception as e:
        log_fn(f"Batch analysis failed ({e!s}). Using fallback: first **{MAX_RESULTS}** items.")
        return _fallback_batch_selections(candidates, source)


def format_date_display(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso_str[:16]


def run_arxiv_mode(
    log_fn: Callable[[str], None],
) -> pd.DataFrame:
    gkey = env_gemini_key()
    if _is_placeholder_secret(gkey):
        st.warning(
            "Gemini API key is missing or still a placeholder. "
            "Set `GEMINI_API_KEY` in `.env` (Google AI Studio)."
        )
        return pd.DataFrame()

    genai.configure(api_key=gkey.strip().strip('"').strip("'"))

    try:
        candidates = fetch_arxiv_candidates(
            KEYWORDS,
            CANDIDATE_LIMIT,
            log_fn,
        )
    except RuntimeError as e:
        st.warning(str(e))
        return pd.DataFrame()

    candidates = candidates[:CANDIDATE_LIMIT]

    log_fn("**Analyzing all candidates in one batch...**")
    picks = batch_analyze_candidates(
        candidates,
        KEYWORDS,
        source="arxiv",
        log_fn=log_fn,
    )
    rows: list[dict[str, Any]] = []
    for rank, pick in enumerate(picks, start=1):
        idx = pick["index"]
        reason = pick["reason"]
        item = candidates[idx]
        log_fn(f"**#{rank}** (item {idx}): {item.get('title', '')[:70]}… — {reason}")
        rows.append(
            {
                "Title": truncate_title(item["title"]),
                "Author": item["author"],
                "Date": format_date_display(item["published_at"]),
                "URL": item["url"],
            }
        )

    return pd.DataFrame(rows)


def run_github_mode(
    log_fn: Callable[[str], None],
) -> pd.DataFrame:
    token = env_github_token()
    if _is_placeholder_secret(token):
        st.warning(
            "GitHub token is missing or still a placeholder. "
            "Set `GITHUB_TOKEN` in `.env`."
        )
        return pd.DataFrame()

    gkey = env_gemini_key()
    if _is_placeholder_secret(gkey):
        st.warning(
            "Gemini API key is missing or still a placeholder. "
            "Set `GEMINI_API_KEY` in `.env` (Google AI Studio)."
        )
        return pd.DataFrame()

    genai.configure(api_key=gkey.strip().strip('"').strip("'"))

    try:
        candidates = fetch_github_candidates(
            token.strip().strip('"').strip("'"),
            KEYWORDS,
            CANDIDATE_LIMIT,
            log_fn,
        )
    except RuntimeError as e:
        st.warning(str(e))
        return pd.DataFrame()

    candidates = candidates[:CANDIDATE_LIMIT]

    log_fn("**Analyzing all candidates in one batch...**")
    picks = batch_analyze_candidates(
        candidates,
        KEYWORDS,
        source="github",
        log_fn=log_fn,
    )
    rows: list[dict[str, Any]] = []
    for rank, pick in enumerate(picks, start=1):
        idx = pick["index"]
        reason = pick["reason"]
        score = int(pick.get("worth_score", 5))
        item = candidates[idx]
        log_fn(
            f"**#{rank}** (item {idx}) **{score}/10:** {item.get('title', '')} — {reason}"
        )
        rows.append(
            {
                "Title": truncate_title(item["title"].replace("/", " / ")),
                "Author": item["author"],
                "Worth Score": score,
                "Stars": int(item["stars"]),
                "Date": format_date_display(item["created_at"]),
                "URL": item["url"],
            }
        )

    return pd.DataFrame(rows)


def column_config_for_mode(mode: str) -> dict[str, Any]:
    url_cfg = st.column_config.LinkColumn("URL", display_text="Open")
    if mode == "ARXIV":
        return {
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Author": st.column_config.TextColumn("Author"),
            "Date": st.column_config.TextColumn("Date"),
            "URL": url_cfg,
        }
    return {
        "Title": st.column_config.TextColumn("Title", width="large"),
        "Author": st.column_config.TextColumn("Author"),
        "Worth Score": st.column_config.NumberColumn("Worth Score", min_value=1, max_value=10),
        "Stars": st.column_config.NumberColumn("Stars", format="%d"),
        "Date": st.column_config.TextColumn("Date"),
        "URL": url_cfg,
    }


def main() -> None:
    st.set_page_config(page_title="Swiss Army Agent", layout="wide")
    st.title("Swiss Army Agent")
    st.caption(
        f"Reasoning: **Gemini (AI Studio)** · Mode: **{MODE}** · "
        f"Keywords: {', '.join(KEYWORDS)} · "
        f"**One batch** call on up to **{CANDIDATE_LIMIT}** candidates → top **{MAX_RESULTS}** rows"
    )

    _, c2, _ = st.columns([1, 2, 1])
    with c2:
        run_clicked = st.button(
            "Run Agentic Loop",
            type="primary",
            use_container_width=True,
        )

    st.markdown("**Thought log**")
    thought_container = st.container(height=300, autoscroll=True)
    table_slot = st.empty()

    if run_clicked:
        lines: list[str] = []

        with st.status("Agent running…", expanded=True) as status:
            st.write("Starting research workflow…")

            def log_fn(msg: str) -> None:
                lines.append(msg)
                thought_container.markdown("\n\n".join(lines))
                st.write(msg)

            try:
                if MODE == "ARXIV":
                    df = run_arxiv_mode(log_fn)
                elif MODE == "GITHUB":
                    df = run_github_mode(log_fn)
                else:
                    st.warning(f"Unknown MODE `{MODE}`. Use ARXIV or GITHUB.")
                    df = pd.DataFrame()
                status.update(label="Done", state="complete")
            except Exception as e:
                st.warning(f"Unexpected error: {e}")
                df = pd.DataFrame()
                status.update(label="Failed", state="error")

        if not df.empty:
            table_slot.dataframe(
                df,
                column_config=column_config_for_mode(MODE),
                hide_index=True,
                use_container_width=True,
            )
        elif run_clicked:
            table_slot.info("No rows to display. Check API keys and logs above.")


if __name__ == "__main__":
    main()
