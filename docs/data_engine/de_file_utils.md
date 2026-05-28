## file_utils.py

**Location:** `data_engine/src/file_utils.py`

**Purpose:** First stage of the data engine pipeline. Discovers raw DICOM case folders on disk, standardises their names to a canonical `BP{NNN}` format, and classifies each folder's internal structure so downstream steps know exactly where to find the DICOM files.

**Domain specificity:** This module is **not generic I/O**. It has deep domain knowledge in several places:
- It knows about the `.dcm` / `.DCM` file extension (DICOM).
- It knows the hospital-specific naming convention `BP{NNN}` (Beneficência Portuguesa).
- It knows that the expected case number range is 1–999.
- It knows that PACS systems nest data at most two directory levels deep.
- It defines and interprets a domain-specific set of status codes (`READY`, `SUBFOLDER_PATH`, `DUPLICATE_DATA`, `EMPTY`, `MISSING`).

---

### Functions

---

#### `get_subfolders(path: str) -> list[str]`

**What it does:** Returns the names of all immediate subdirectories inside `path`.

**Why it was written this way:** Uses `os.scandir` (one syscall returning `DirEntry` objects with cached `stat` data) instead of `os.listdir` + `os.path.isdir`, which would require a separate `stat()` call per entry. The performance difference matters when scanning hundreds of DICOM series directories. Errors are caught and logged rather than raised so that a single missing folder does not abort the full pipeline scan.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `path` | `str` | Absolute path to the directory to scan. |

**Return value:** `list[str]` — subdirectory names only (not full paths). Order is filesystem-dependent.

**Side effects / exceptions:** Logs an error via the `dicom_ingestion` logger on `FileNotFoundError`; returns `[]` instead of raising.

**Overlap with stdlib:** `pathlib.Path.iterdir()` with a `.is_dir()` filter provides equivalent functionality with comparable performance. This function is a thin, focused wrapper that adds the silent-on-error behaviour. Not a problematic duplication, but callers should be aware the same scan is achievable with two lines of `pathlib`.

---

#### `count_dcm_files(path: str) -> int`

**What it does:** Counts `.dcm` files (case-insensitively) directly inside `path`, non-recursively.

**Why it was written this way:** Intentionally non-recursive. The caller (`get_folder_stats`) calls this function separately for the case folder and for each immediate subfolder, keeping the two counts distinct so the layout classification logic in `add_data_codes` can distinguish "files at the top level" from "files inside a subfolder". The case-insensitive extension check (`.lower().endswith('.dcm')`) handles PACS systems that export uppercase `.DCM` extensions. Errors return 0 so a single unreadable folder does not abort the scan.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `path` | `str` | Absolute path to the directory to scan. |

**Return value:** `int` — number of `.dcm` files at the top level of `path`.

**Side effects / exceptions:** Logs an error via the `dicom_ingestion` logger on `FileNotFoundError` or `OSError`; returns `0` instead of raising.

**Domain knowledge:** Hardcodes the `.dcm` extension — not a generic file counter.

---

#### `generate_new_names(folder_list: list[str]) -> list[dict]`

**What it does:** Maps each raw folder name to a canonical `BP{NNN}` name, handling zero-padding and duplicate collisions with alphabetic suffixes.

**Why it was written this way:**
- The input list is sorted before processing so that duplicate-suffix assignment (`A`, `B`, `C` …) is deterministic regardless of filesystem directory ordering. This makes the function idempotent across runs.
- The regex `^(bp)_?(\d+)` with `re.IGNORECASE` deliberately accepts `bp_001` (underscore separator) alongside `BP001` and `bp001`. Anything after the digit sequence is silently ignored so that PACS-appended suffixes (scan dates, operator codes) do not break matching.
- Non-matching names (e.g. `test_data`, `archive`) pass through unchanged rather than being dropped, so they still appear in the audit log and can be reviewed manually.
- `defaultdict(int)` is used for both the frequency count and the suffix counter to avoid key-existence checks.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `folder_list` | `list[str]` | Raw folder names as returned by `get_subfolders`. |

**Return value:** `list[dict]` — one dict per folder with keys `original_name` and `fixed_name`, in sorted order.

**Side effects / exceptions:** None. Pure function with no I/O or logging.

**Domain knowledge:** The `BP` prefix and the 3-digit zero-padding are hospital-specific conventions. The regex and suffix logic only make sense in this domain context.

**Complex control flow flag:** The duplicate-handling block uses two separate passes over `potential_names` (one to count, one to assign suffixes). This is correct and clear, but could be expressed more concisely with `itertools.groupby` on the sorted list. Not a bug; minor readability opportunity only.

---

#### `find_missing_cases(name_mapping: list[dict]) -> list[dict]`

**What it does:** Identifies case numbers in the range 1–999 that are absent from the raw data and returns placeholder records for them.

**Why it was written this way:** The dataset is expected to be a contiguous numbered sequence (BP001–BP999). Gaps represent cases that were never collected, lost, or not yet exported from the PACS. Returning placeholder dicts (rather than just a list of integers) means these records can be appended directly to `name_mapping` and processed by `add_data_codes` and `add_data_paths` without special-casing elsewhere in the pipeline.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `name_mapping` | `list[dict]` | Case record dicts, each with a `fixed_name` key (e.g. `'BP001'`). |

**Return value:** `list[dict]` — one dict per missing number, each `{'original_name': 'missing', 'fixed_name': 'BP{NNN}'}`.

**Side effects / exceptions:** Logs one `INFO` message. No exceptions raised.

**Domain knowledge:** The range 1–999 and the `BP` prefix are hard-coded domain constants. The function is not meaningful outside this dataset context.

**Complex control flow flag:** The range `range(1, 1000)` hard-codes the upper bound. If the dataset ever exceeds 999 cases this silently stops working. A named constant or a parameter with a default would make the constraint explicit and easier to change.

---

#### `get_folder_stats(base_path: str, folder_list: list[str]) -> list[dict]`

**What it does:** Inspects the internal layout of each case folder to produce counts of top-level `.dcm` files, non-empty subfolders, and `.dcm` files inside those subfolders.

**Why it was written this way:** Different PACS systems nest DICOM files at different depths. The function performs a deliberately shallow two-level scan (case folder → immediate subfolders only) because standard PACS exports never go deeper than two levels, and a full recursive walk would be slow and risk counting non-DICOM content in nested archive structures. A subfolder is considered "non-empty" if it contains any `.dcm` files **or** any sub-subfolders — the latter acts as a proxy for "data is nested one level deeper than expected", which the pipeline can then flag as `DUPLICATE_DATA` for manual review.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `base_path` | `str` | Absolute path to the root raw data directory. |
| `folder_list` | `list[str]` | Names of case folders to inspect (relative to `base_path`). |

**Return value:** `list[dict]` — one dict per successfully scanned folder with keys: `folder`, `direct_items`, `non_empty_subfolders` (list of names), `items_in_subfolders`. Folders that raise `OSError` are skipped entirely (not included in the output list).

**Side effects / exceptions:** Logs one `INFO` message at the start and one `INFO` per `OSError`. Note that an `OSError` on a folder causes it to be **silently omitted** from the result list rather than producing a record with zeroed counts — the caller (`organize_data`) must be aware that the output length may be shorter than `folder_list`.

**Domain knowledge:** Calls `count_dcm_files` (DICOM-specific) and interprets the two-level-depth heuristic based on known PACS export behaviour.

**Complex control flow flag:** The error handling inconsistency is worth noting: `count_dcm_files` and `get_subfolders` both return safe defaults (0 / []) on error, but an `OSError` at the top-level `try` block causes the entire folder to be omitted from `stats_list`. This means a folder that raises `OSError` will not appear in `stats_map` in `organize_data`, so its name-mapping record will lack `direct_items` and will be classified as `MISSING` rather than `EMPTY`. This is a subtle difference in error semantics that could mislead audit log readers.

---

#### `add_data_codes(name_mapping: list[dict]) -> list[dict]`

**What it does:** Assigns a `data_code` string to each case record based on the counts in its folder stats.

**Why it was written this way:** Acts as a simple decision table (a state machine over two integer inputs: `direct_items` and `non_empty_subfolders`). Encoding the logic here rather than inline in `organize_data` keeps the classification rules in one place and makes them independently testable. Records without `direct_items` (injected by `find_missing_cases`) are detected by key absence and immediately assigned `MISSING` without falling through the normal logic.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `name_mapping` | `list[dict]` | Case record dicts, optionally containing `direct_items` and `non_empty_subfolders`. |

**Return value:** The same `list[dict]`, mutated in-place with a `data_code` key added to every record. Also returned explicitly.

**Side effects / exceptions:** Mutates the input list in-place. Logs `INFO` messages at start and end.

**Complex control flow flag:** The function both mutates its input in-place **and** returns it. This dual behaviour (common to several functions in this module) is a design inconsistency — callers could either use the return value or rely on mutation, and new callers may not realise the input is modified. A consistent choice (pure function returning a new list, or in-place with no return value) would be clearer.

---

#### `add_data_paths(name_mapping: list[dict]) -> list[dict]`

**What it does:** Resolves the relative filesystem path the DICOM validator should use for each case and removes the now-redundant `non_empty_subfolders` key.

**Why it was written this way:** The `data_path` field is what all downstream steps actually consume; everything computed up to this point was in service of deriving it. For `SUBFOLDER_PATH` cases, the path is constructed by joining the original folder name with the single non-empty subfolder name using `os.path.join`. For cases that should be skipped (`MISSING`, `EMPTY`, `DUPLICATE_DATA`), the code string itself is used as a sentinel value that downstream validators will recognise as a non-path and skip. The `non_empty_subfolders` list is deleted here as an explicit memory/CSV-bloat reduction step.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `name_mapping` | `list[dict]` | Case record dicts with `data_code` and optionally `non_empty_subfolders` already populated. |

**Return value:** The same `list[dict]`, mutated in-place with `data_path` added and `non_empty_subfolders` removed. Also returned explicitly.

**Side effects / exceptions:** Mutates the input list in-place (deletes `non_empty_subfolders`). Logs one `INFO` message.

**Overlap with stdlib:** `os.path.join` could be replaced by `pathlib.Path` / operator, but this is not a concern.

**Complex control flow flag:** Same in-place-mutation-plus-return pattern noted under `add_data_codes`. The sentinel value approach (storing the code string as the `data_path` for non-processable cases) is an implicit protocol between this function and all downstream consumers. A `None` value or a dedicated `None`-check would be more idiomatic Python and less surprising to a new reader.

---

#### `organize_data(case_folders: list[str], RAW_DATA_PATH: str) -> list[dict]`

**What it does:** Runs the full name-standardisation and folder-analysis sub-pipeline and returns one fully-populated case record dict per discovered folder plus one per missing case number in BP001–BP999.

**Why it was written this way:** Acts as the single public entry point called by `data_cleaner.py`. Chaining all seven steps here rather than exposing them individually means the caller does not need to know the internal step order or the intermediate data shapes. The stats are indexed into a dict (`stats_map`) before the merge loop to achieve O(1) lookup per case rather than an O(n²) nested scan.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `case_folders` | `list[str]` | Raw folder names from `get_subfolders` (relative names only). |
| `RAW_DATA_PATH` | `str` | Absolute path to the raw data directory. Note: the parameter name uses `UPPER_CASE`, which is unconventional for a local variable in Python (PEP 8 reserves this style for module-level constants). |

**Return value:** `list[dict]` — fully-populated records. Each dict contains at minimum: `original_name`, `fixed_name`, `data_code`, `data_path`, and `total_dcms` (for records that have filesystem stats). Records for missing cases contain only `original_name`, `fixed_name`, and `data_code`.

**Side effects / exceptions:** Logs `INFO` messages (delegated to called functions). No exceptions raised directly.

**Complex control flow flag:** Step 4 (computing `total_dcms`) is a `for` loop that only updates records that already have `direct_items`. Records without that key (which will be `MISSING` cases appended in step 5) simply never get a `total_dcms` key. This means the output dicts have **different key sets** depending on their `data_code`, which is an implicit schema that downstream consumers must handle defensively (e.g. using `.get('total_dcms', 0)`). Explicit initialisation of all keys to `None` or `0` at record creation time would make the schema uniform and reduce surprises.

---

### Summary of flags

| Issue | Functions affected |
|---|---|
| In-place mutation AND explicit return (dual behaviour) | `add_data_codes`, `add_data_paths` |
| Non-uniform dict schema across records | `organize_data` (step 4), `find_missing_cases` |
| Hard-coded upper bound (999) for case range | `find_missing_cases` |
| `OSError` on a folder silently omits it vs. returning zeroed counts | `get_folder_stats` |
| `UPPER_CASE` parameter name (PEP 8 violation) | `organize_data` (`RAW_DATA_PATH`) |
| Sentinel string used as `data_path` for non-processable cases | `add_data_paths` |
| Partial overlap with `pathlib` (minor) | `get_subfolders`, `count_dcm_files` |

### Domain specificity verdict

**Domain-specific.** This module cannot be used as a general-purpose filesystem utility library. It encodes knowledge of DICOM file extensions, a hospital-specific folder naming convention, PACS export directory layouts, and a fixed expected case-number range. The generic I/O primitives (`get_subfolders`, `count_dcm_files`) are small enough that they could be replaced by `pathlib` one-liners, but all higher-level functions are tightly coupled to the aneurysm dataset ingestion domain.
