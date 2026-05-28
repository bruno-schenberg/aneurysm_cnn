## logging_utils.py

**File:** `/home/bruno/Projects/aneurysm_cnn/data_engine/src/logging_utils.py`

This module configures the pipeline logger and produces two artefacts at the end of every data engine run: a JSONL log file and a CSV audit log. It contains one class and three functions.

---

### Overall assessment

This is **not** a thin wrapper over stdlib `logging`. It adds meaningful, justified behavior: a custom JSONL handler, a guard against duplicate handler registration, dual-level fan-out (DEBUG to file, INFO to console), and a structured CSV audit trail separate from the runtime log. The design choices are well-motivated.

**No global logger state is created on import.** No module-level `logging.basicConfig()` or `getLogger()` calls are executed at import time. All setup happens inside `setup_logger()`, which must be called explicitly. There are no import-time side effects.

---

### `_JsonlHandler` (class)

**What it does.**
A `logging.FileHandler` subclass that overrides `emit()` to write each log record as a single JSON object on its own line (JSONL format). Each line has four fields: `timestamp` (ISO-8601), `level`, `logger`, and `message`. When `exc_info` is present, an `exception` field is appended.

**Why not use stdlib directly.**
Plain-text `FileHandler` output requires parsing for post-run analysis. JSONL records can be filtered by level, loaded into a dataframe, or grepped by field with zero parsing logic. Subclassing `FileHandler` rather than `Handler` means file opening, closing, and error handling are inherited — only the serialisation format changes.

**Design notes.**
- `self.flush()` is called after every `emit()` so records survive a mid-run crash.
- `emit()` is wrapped in `try/except` with `self.handleError(record)` as fallback — standard logging contract: a handler must never raise.
- Leading underscore signals implementation detail; callers interact only with the logger returned by `setup_logger`.

**Parameters / return values.**
No `__init__` override. Inherits `FileHandler.__init__(filename, mode='a', ...)`.
- `emit(record: logging.LogRecord) -> None` — called automatically by the logging framework.

---

### `setup_logger` (function)

**What it does.**
Creates and configures the named logger `"dicom_ingestion"`, attaches two handlers (DEBUG-level JSONL file, INFO-level console), and returns it.

**Why not use root logger / `basicConfig`.**
A named logger isolates pipeline output from third-party library logging. `basicConfig` configures the root logger globally and cannot attach two handlers with different levels and formatters in a single call.

**Design notes.**
- Guard `if logger.handlers: return logger` prevents duplicate handlers when `setup_logger` is called more than once (e.g., in tests).
- Logger level is `DEBUG`; each handler applies its own level filter — standard dual-destination pattern.

**Parameters.**
- `log_path: str` — path to the JSONL log file; defaults to `"ingestion.log"` (relative path — callers should pass an absolute path).

**Returns.**
- `logging.Logger` — the configured `"dicom_ingestion"` logger.

**Side effects.**
- Opens/creates the file at `log_path`.
- Mutates the global logging registry (persists for process lifetime).

---

### `write_audit_log` (function)

**What it does.**
Writes a CSV recording the final outcome of every processed case: `exam_name`, `status` (`success` / `skipped` / `failed`), `reason`, and `output_path`.

**Why a separate CSV rather than parsing the JSONL log.**
The JSONL log is a verbose event stream. The audit CSV is a flat one-row-per-case summary designed for quick inspection and joining with the validation summary from `data_cleaner.py`.

**Parameters.**
- `results: list[dict]` — list of dicts with keys `exam_name`, `status`, `reason`, `output_path`. Produced by `nifti_utils.process_and_convert_exams`.
- `output_path: str` — path to output CSV; overwritten if it exists; defaults to `"ingestion_summary.csv"`.

**Returns.** `None`.

**Side effects.**
- Writes/overwrites the file at `output_path`.
- Logs an `ERROR` on failure via a bare `logging.getLogger` call (slightly inconsistent with the rest of the module which uses the returned logger object).

---

### `log_jsonl` (function)

**What it does.**
Appends a single dict as a JSON line to a separate JSONL metrics file. Adds a `timestamp` key before serialisation.

**Current status: dead code.** The module docstring states this function is not called by the main pipeline. It is a hook for future ad-hoc metrics logging.

**Flags.**
- **In-place mutation:** `data['timestamp'] = datetime.now().isoformat()` mutates the caller's dict. Safer pattern: `{**data, 'timestamp': ...}`.
- Opens and closes the file on every call — harmless for a once-per-run hook but would be slow at high frequency.
- Logs to `"dicom_ingestion"` which may not be configured if `setup_logger` has not been called; will silently emit nothing in Python 3 via `lastResort`.

**Parameters.**
- `data: dict` — key-value pairs to record. **Mutated in-place.**
- `log_path: str` — path to the JSONL metrics file; defaults to `"ingestion_metrics.jsonl"`.

**Returns.** `None`.

---

### Global state and import-time side effects

None. No code executes at import time beyond the four `import` statements.

---

### Issues to address in rewrite

| Issue | Location | Suggestion |
|---|---|---|
| Dead code | `log_jsonl` | Remove or activate; don't leave unused hooks |
| In-place dict mutation | `log_jsonl` | Use `{**data, 'timestamp': ...}` |
| Relative path defaults | `setup_logger`, `write_audit_log` | Remove defaults or require callers to pass absolute paths |
| Logger name `"dicom_ingestion"` | `setup_logger` | Rename to match the module's actual scope (data pipeline, not just DICOM ingestion) |
