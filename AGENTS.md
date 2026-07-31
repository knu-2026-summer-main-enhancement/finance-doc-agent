# Finance Document Agent — Development Guidelines

## Purpose

This repository contains a Korean financial-document Q&A agent for scholarship,
donation, dues, and support-payment documents. Preserve the distinction between
exact structured-data execution and evidence-grounded document search when
maintaining or extending the project.

Start with these documents:

- `README.md`: user-facing features and quick start
- `backend/README.md`: backend, API, storage, and operations
- `backend/utils/README.md`: ingestion and semantic schema
- `backend/rag/README.md`: routing and Vector RAG
- `backend/pandas_engine/README.md`: QueryPlan execution and interactive results
- `backend/tests/README.md`: tests and goldset evaluation

## Repository layout

- `backend/main.py`: FastAPI endpoints, request scope, routing, and dispatch
- `backend/rag/`: question decisions, deterministic planning, LLM planning,
  Vector retrieval, and autocomplete
- `backend/pandas_engine/`: typed QueryPlan validation, deterministic execution,
  formatting, presentation, and interactive payloads
- `backend/utils/`: Excel, PDF, HWP/HWPX, and image-table ingestion
- `backend/static/`: desktop and mobile chat UI
- `backend/tests/`: unit tests, goldsets, evaluators, and untracked result output

Use repository-relative paths in code and documentation. Do not add
developer-specific absolute paths.

## Core architecture contracts

### Structured data

Questions about rows, people, dates, amounts, counts, rankings, lists, or tables
belong to the structured-data path.

1. Try schema-grounded deterministic planning.
2. If that cannot safely produce a plan, use the LLM question decision and the
   restricted LLM QueryPlan generator.
3. Validate every DataFrame, column, filter, operator, type, and question
   grounding before execution.
4. Execute only through the deterministic executor.

Never execute LLM-generated Python or trust unvalidated JSON.

### Document search

Questions about purpose, background, criteria, procedures, schedules, or other
prose belong to the Vector path.

- Search only within the selected document scope.
- Preserve PDF/HWP section hierarchy and table-row metadata.
- Retrieve child chunks, rerank them, and expand relevant parent sections.
- Answer only from retrieved evidence; do not infer facts from a filename.
- Section-browser and document autocomplete questions must keep their explicit
  Vector route hint.

### Presentation

- `목록` is a readable, usually name-centered result.
- Explicit `표` or `테이블` requests return safe original columns as structured
  table data.
- Keep internal, OCR-quality, search, and identity-derived columns out of the
  table and person-card UI.
- Page large table results through the existing result reference instead of
  embedding every row in one response.
- The browser must render structured payloads directly. Do not scrape Korean
  answer text to recover people, amounts, rows, or calculation provenance.

## Schema and ingestion rules

- Prefer semantic-schema and metadata-driven behavior over document-specific
  column names, file names, years, people, or fixed column positions.
- Preserve original columns and their order. Add derived columns only as
  internal metadata.
- Support reordered, renamed, added, and missing columns without regressing the
  current reference documents.
- Keep date resolution explicit when a document contains full dates and
  separate year/month components or several business-date columns.
- Normalize numeric money, comma-form money, units, and Korean shorthand such
  as `2만원` consistently.
- Replace shared DataFrame state only after a new ingestion snapshot succeeds.
- Bump the relevant parser or semantic-schema version when a parsing contract
  changes and existing documents require reingestion.
- Do not hardcode a goldset answer, source document name, test ID, or known row
  value into production code.

## Identity, ranking, and privacy

- “How many people” means distinct people, not row count.
- Duplicate display names must remain explicit. Do not merge different people
  merely because their names match.
- A real name may match a masked source name only when length and every visible
  character position are compatible.
- Distinguish payment-row ranking from person-total ranking.
- Keep ordinal position, explicit limit, ascending/descending direction, and
  dense-tie behavior consistent across planner, validator, executor, formatter,
  API payload, and UI.
- Never write raw question text, names, phone numbers, email addresses, or
  expanded retrieval queries to application logs. Evaluation fixtures may
  contain the minimum expected identity data needed for correctness, but
  generated reports must not add contact data.
- Phone numbers and email addresses may appear only in a user-requested person
  detail view, not in general answers, bulk tables, calculation contributors,
  or logs.

## Autocomplete

- Suggest only questions that compile against the selected schema or are valid
  metadata/Vector actions.
- Person names must come from the selected source; never invent a name.
- General keystroke filtering stays local after catalog load.
- Large person collections use the bounded prefix endpoint and must not send the
  full name list to the browser.
- Keep list and table suggestions available for whole-document, date/range,
  person, year, and cohort conditions.
- Preserve the current desktop and mobile suggestion limits and avoid sending a
  request for every keystroke.

## Testing

Run commands from `backend/`.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

For server-backed evaluation, use an isolated port such as `8081` and verify
that the process belongs to this checkout before treating results as evidence.

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8081
.\venv\Scripts\python.exe tests\eval.py --url http://127.0.0.1:8081 --tag <tag>
```

When diagnosing a failure:

1. Inspect the source document and expected result.
2. Separate routing, planning, validation, execution, formatting/UI, and
   evaluator failures.
3. Compare filters, returned rows, scalar values, contributor counts, and
   distinct-person counts with the source.
4. Run focused tests after the change.
5. Run the full unit suite and the relevant goldset before handoff.

Keyword recall alone is not a sufficient correctness metric.

## Change and Git safety

- Inspect related code and tests before structural changes.
- Preserve unrelated changes and untracked local reports.
- Keep compatibility across the executor, formatter, structured API, and UI.
- Do not silently edit `.env`, delete stored data, reset branches, force-push,
  or overwrite another developer's work.
- Check `git status --branch` and remote divergence before Git operations.
- Commit only files related to the requested change.
- Pull, merge, commit, or push only with explicit user authorization.
- Report the exact tests and checks that actually ran.
