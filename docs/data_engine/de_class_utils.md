## class_utils.py

**File:** `data_engine/src/class_utils.py`

**Purpose:** Joins each validated DICOM case record to its ground-truth clinical label from `classes.csv`, then flags any `OK` case that has no label so it is excluded from NIfTI conversion.

This module is **specific to the aneurysm classification task**. The regex pattern `BP\d+`, the CSV column names (`exam`, `class`, `location`, `Age`), and the binary label semantics (`0` = healthy, `1` = aneurysm) are all hard-coded assumptions about the project's dataset conventions. Nothing here is general-purpose.

---

### Module-level state

| Name | Type | Notes |
|------|------|-------|
| `logger` | `logging.Logger` | Module-level logger named `"dicom_ingestion"`. **Global state**: shared with every other module that uses the same logger name. No handlers are configured here; configuration is assumed to happen in the entry-point (`data_cleaner.py`). |

No mutable default arguments are present. There are no module-level data structures that accumulate state across calls.

---

### `join_class_data`

**What it does:** Reads `classes.csv` into an in-memory lookup table and merges the label, anatomical location, and patient age into each validated case record.

**Why written this way:**
- The CSV is converted to a `dict` keyed by uppercase exam name on first pass so every subsequent lookup is O(1), avoiding a nested loop over the CSV for each case.
- Uppercasing both sides of the key comparison makes the join case-insensitive, defending against inconsistent capitalisation between the CSV and the pipeline-generated names.
- The regex strip (`BP\d+`) is done at join time rather than during earlier validation so that `file_utils` disambiguation suffixes (`A`, `B`) do not need to be undone before this step — the two concerns stay separated.
- Missing file is handled with a warning + early return rather than an exception so the pipeline can continue and report the situation without crashing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `validated_data` | `list[dict]` | Case records produced by `dicom_utils.validate_dcms`. Each dict must have at least a `fixed_name` key (e.g. `"BP001"`, `"BP001A"`). |
| `classes_csv_path` | `str` | Absolute path to `dataset/classes.csv`. Expected columns: `exam`, `class`, `location`, `Age`. |

**Returns:** `list[dict]` — the same list object passed in, with `'class'`, `'location'`, and `'patient_age'` merged into each record whose `fixed_name` matches a CSV entry. Records with no match are returned unchanged (no new keys added).

**Side effects:**
- Mutates the dicts inside `validated_data` in-place via `dict.update()`. The caller's list and its dicts are modified directly; no copy is made.
- Logs one `INFO` line at the start of the join.
- Logs one `WARNING` line if the CSV file is not found, then returns early.

**Exceptions:** `FileNotFoundError` is caught internally. No other exceptions are caught; a malformed CSV (e.g. wrong delimiter, encoding error) will propagate to the caller.

**Implicit coupling:**
- Assumes `fixed_name` values follow the pattern `BP<digits>[optional letter suffix]`. Any case name that does not match `r"(BP\d+)"` is silently skipped — no log entry is produced for non-matching names.
- Depends on the CSV column named exactly `Age` (capital A); all other columns are lowercase. This inconsistency is baked into the `row.get('Age')` call.

---

### `check_missing_class`

**What it does:** Iterates over the case list and sets `validation_status` to `'MISSING_CLASS'` for any record whose status is `'OK'` but that has no `'class'` key (or has a falsy class value).

**Why written this way:**
- Runs as a separate pass after `join_class_data` rather than being folded into the join loop, keeping the two concerns (label attachment and completeness checking) independent and individually testable.
- Only touches `OK` cases deliberately: cases already carrying a failure status (e.g. `VARIABLE_SPACING`, `MIXED_SERIES_ERROR`) would lose their diagnostic status if overwritten, so the guard `validation_status == 'OK'` is load-bearing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `list[dict]` | Case records after `join_class_data` has run. Each dict is expected to have `validation_status` and optionally `class`. |

**Returns:** `list[dict]` — the same list object passed in, with `validation_status` updated in-place for any newly flagged records.

**Side effects:**
- Mutates dicts inside `data` in-place.
- Logs one `INFO` line before the loop and one `INFO` line reporting the count of flagged cases.

**Exceptions:** None raised or caught. Missing keys are accessed via `.get()`.

**Note on falsy check:** `not item.get('class')` treats a missing key and `''` as falsy, but `'0'` (healthy) is truthy — correctly not flagged. A blank CSV cell would be flagged; this is the intended behaviour.

---

### Control-flow assessment

Both functions are structurally flat — no nested loops or long `if/elif` chains. No simplification needed.

---

### Summary of flags

| Category | Finding |
|----------|---------|
| Global state | `logger` is a module-level singleton shared by name with other modules. |
| In-place mutation | Both functions mutate the input list's dicts and return the same list. |
| Implicit coupling (naming) | `r"(BP\d+)"` hard-codes the project's case-naming convention. Non-BP names are silently ignored. |
| Implicit coupling (CSV schema) | Column `Age` differs in capitalisation from `exam`, `class`, `location`; must match CSV exactly. |
| Silent no-op | Cases whose `fixed_name` does not match the regex receive no label and no log entry. |
| Task specificity | Entirely specific to the aneurysm classification task; not general-purpose. |
