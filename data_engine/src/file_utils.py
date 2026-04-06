"""
file_utils.py

Handles the first stage of the data engine pipeline: discovering raw DICOM case
folders on disk, standardizing their names to a canonical format, and classifying
each folder's internal structure so downstream steps know exactly where to find
the DICOM files.

## Why this step is necessary

Raw DICOM exports from hospital PACS systems arrive with inconsistent folder names.
Different scan sessions, operators, and export tools produce names like:
  bp1 / BP1 / BP_001 / bp001 / case_001

All of these refer to the same patient case. Before any clinical data can be
processed reproducibly, every folder must be mapped to a single canonical name.
This module produces that mapping and records it so every rename decision is fully
traceable in the output audit CSV.

## Canonical name format

  BP{NNN}   — prefix 'BP' (Beneficência Portuguesa, the hospital where data was collected), zero-padded 3-digit number
  e.g.  bp1  →  BP001
        BP_02 →  BP002
        bp003 →  BP003

If two raw folders map to the same canonical name (e.g. both 'bp1' and 'bp001'
become 'BP001'), they receive alphabetic suffixes in sorted order:
  bp001 → BP001A,  bp1 → BP001B

## Folder structure classification

DICOM exports from different PACS systems place files at different levels:
  - Some put all .dcm files directly inside the case folder  → READY
  - Some create one subfolder per series                     → SUBFOLDER_PATH
  - Some produce empty folders or conflicting duplicate data → EMPTY / DUPLICATE_DATA

This module inspects each folder and assigns a data code that the rest of the
pipeline uses to decide whether and where to read DICOM files.

Public API (called by data_cleaner.py):
  - get_subfolders(path)      : list immediate subdirectory names
  - organize_data(folders, raw_path) : runs the full pipeline, returns a list of
                                       case record dicts with fixed names, codes, paths
"""

import logging
import os
import re
from collections import defaultdict

logger = logging.getLogger("dicom_ingestion")


# ----------------------------------------------------
# 1. Filesystem Utilities
# ----------------------------------------------------


def get_subfolders(path: str) -> list[str]:
    """
    Returns the names of all immediate subdirectories inside ``path``.

    Uses ``os.scandir`` (a single syscall) rather than ``os.listdir`` + ``os.path.isdir``
    to avoid a second stat() call per entry — important when scanning folders with
    hundreds of DICOM series.

    Returns an empty list if the directory does not exist, logging the error
    rather than raising so the pipeline can continue and mark the case as MISSING.

    Args:
        path: Absolute path to the directory to scan.

    Returns:
        List of subdirectory names (not full paths). Order is filesystem-dependent.
    """
    try:
        with os.scandir(path) as entries:
            subfolders = [entry.name for entry in entries if entry.is_dir()]
        return subfolders
    except FileNotFoundError:
        logger.error(f"Error: Directory not found at '{path}'")
        return []


def count_dcm_files(path: str) -> int:
    """
    Counts the number of ``.dcm`` files directly inside ``path``.

    Intentionally **non-recursive** — only files at the top level of ``path``
    are counted, not files inside any subdirectories. This is by design: the
    caller (``get_folder_stats``) separately handles files in subfolders so that
    the two counts remain distinct and can be used to classify the folder's layout.

    The extension check is case-insensitive (``.dcm`` and ``.DCM`` are both
    counted) because some PACS exports use uppercase extensions.

    Returns 0 (rather than raising) on filesystem errors so a single unreadable
    folder does not abort the full pipeline scan.

    Args:
        path: Absolute path to the directory to scan.

    Returns:
        Number of ``.dcm`` files at the top level of ``path``.
    """
    dcm_file_count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.dcm'):
                    dcm_file_count += 1
        return dcm_file_count
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: Could not count items in '{path}': {e}")
        return 0


# ----------------------------------------------------
# 2. Name Standardisation
# ----------------------------------------------------


def generate_new_names(folder_list: list[str]) -> list[dict]:
    """
    Maps each raw folder name to a canonical ``BP{NNN}`` name.

    **Matching rule:** the folder name must begin with ``bp`` or ``BP`` (optionally
    followed by ``_``) and then one or more digits. Anything after the digits is
    ignored (e.g. trailing scan dates or suffixes added by the PACS export tool).
    Non-matching names (e.g. ``test_data``, ``archive``) are left unchanged.

    **Zero-padding:** the numeric part is parsed as an integer (dropping any
    existing leading zeros) and then zero-padded back to exactly 3 digits so
    ``bp1``, ``BP01``, and ``bp001`` all produce ``BP001``.

    **Duplicate handling:** when two raw names produce the same canonical name,
    the list is first sorted alphabetically so duplicate suffixes (A, B, C …)
    are assigned in a deterministic order that does not depend on filesystem
    directory ordering. For example:
      Input (any order): ["bp1", "bp001"]
      Sorted:            ["bp001", "bp1"]
      Output:            bp001 → BP001A,  bp1 → BP001B

    This guarantees that re-running the pipeline on the same raw data always
    produces the same rename map.

    Args:
        folder_list: Raw folder names as returned by ``get_subfolders``.

    Returns:
        List of dicts, each with ``'original_name'`` and ``'fixed_name'`` keys,
        in sorted order.
    """
    # Sort before assigning suffixes so that duplicates always receive the same
    # suffix regardless of filesystem ordering (e.g., bp001 always → BP001A).
    sorted_folders = sorted(folder_list)

    potential_names = []
    # Regex: optional underscore between 'bp' and the digit sequence is allowed
    # (e.g. BP_001), but nothing else between prefix and number.
    pattern = re.compile(r"^(bp)_?(\d+)", re.IGNORECASE)

    for original_name in sorted_folders:
        match = pattern.match(original_name)
        if match:
            # Parse as int to drop leading zeros, then re-pad to 3 digits.
            number_part = str(int(match.group(2)))
            fixed_name = f"BP{number_part.zfill(3)}"
        else:
            # Non-matching names pass through unchanged so they still appear
            # in the audit log and can be manually reviewed.
            fixed_name = original_name
        potential_names.append({"original_name": original_name, "fixed_name": fixed_name})

    # --- Handle duplicate fixed names ---
    # Count how many raw folders map to each canonical name.
    fixed_name_counts = defaultdict(int)
    for item in potential_names:
        fixed_name_counts[item['fixed_name']] += 1

    rename_map = []
    suffix_counters = defaultdict(int)
    for item in potential_names:
        base_name = item['fixed_name']
        if fixed_name_counts[base_name] > 1:
            # Append 'A', 'B', 'C' … in the order we encounter each duplicate.
            # Because the list was sorted above, this order is deterministic.
            suffix = chr(ord('A') + suffix_counters[base_name])
            item['fixed_name'] = f"{base_name}{suffix}"
            suffix_counters[base_name] += 1
        rename_map.append(item)

    return rename_map


def find_missing_cases(name_mapping: list[dict]) -> list[dict]:
    """
    Identifies case numbers in the range 1–999 that are absent from the raw data.

    The dataset is expected to be a contiguous numbered sequence (BP001, BP002, …).
    Gaps in the sequence indicate cases that were either never collected, lost, or
    not yet exported from the PACS. Recording these gaps in the audit log makes
    the completeness of the dataset explicit.

    This function is purely informational — missing cases receive the data code
    ``MISSING`` and are excluded from DICOM validation and NIfTI conversion
    downstream.

    Args:
        name_mapping: List of case record dicts, each containing a ``'fixed_name'``
            key (e.g. ``'BP001'``).

    Returns:
        List of dicts, one per missing case number, with
        ``{'original_name': 'missing', 'fixed_name': 'BP{NNN}'}``.
    """
    logger.info("Checking for missing case numbers...")
    pattern = re.compile(r"BP(\d+)")
    existing_numbers = set()
    for item in name_mapping:
        match = pattern.match(item['fixed_name'])
        if match:
            existing_numbers.add(int(match.group(1)))

    missing_rows = []
    for i in range(1, 1000):
        if i not in existing_numbers:
            fixed_name = f"BP{i:03d}"
            missing_rows.append({'original_name': 'missing', 'fixed_name': fixed_name})
    return missing_rows


# ----------------------------------------------------
# 3. Folder Statistics
# ----------------------------------------------------


def get_folder_stats(base_path: str, folder_list: list[str]) -> list[dict]:
    """
    Inspects the internal layout of each case folder to determine where DICOM
    files actually reside.

    DICOM exports from different PACS systems use different directory layouts:
    - Some place all ``.dcm`` files directly inside the case folder.
    - Some create a single subfolder per series (e.g. ``bp001/DICOM/``).
    - Some produce multiple series subfolders (e.g. contrast + non-contrast).

    To classify each folder correctly, this function counts:
    1. ``.dcm`` files **directly** inside the case folder (``direct_items``).
    2. Immediate subfolders that are **non-empty** — i.e. contain at least one
       ``.dcm`` file or at least one further subfolder (``non_empty_subfolders``).
    3. The total ``.dcm`` files across all immediate subfolders, counted one
       level deep (``items_in_subfolders``).

    The scan is intentionally **shallow** (two levels maximum). DICOM data from
    standard PACS exports is never nested more than two levels deep, and a deep
    recursive scan would be slow and could accidentally count non-DICOM content
    in nested archive structures.

    Args:
        base_path: Absolute path to the root raw data directory.
        folder_list: Names of case folders to inspect (relative to ``base_path``).

    Returns:
        List of stat dicts, one per folder. Each dict contains:
        ``{'folder', 'direct_items', 'non_empty_subfolders', 'items_in_subfolders'}``.
    """
    logger.info("\nGathering folder statistics...")
    stats_list = []

    for folder_name in folder_list:
        folder_path = os.path.join(base_path, folder_name)
        items_in_subfolders = 0
        non_empty_subfolder_names = []

        try:
            direct_files = count_dcm_files(folder_path)
            subfolders = get_subfolders(folder_path)

            for subfolder_name in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                num_items = count_dcm_files(subfolder_path)
                # Check for sub-subfolders as a proxy for "non-empty" even if
                # the DICOM files are nested one level deeper than expected.
                num_sub_subfolders = len(get_subfolders(subfolder_path))

                if num_items > 0 or num_sub_subfolders > 0:
                    non_empty_subfolder_names.append(subfolder_name)

                items_in_subfolders += num_items

            stats_list.append({
                'folder': folder_name,
                'direct_items': direct_files,
                'non_empty_subfolders': non_empty_subfolder_names,
                'items_in_subfolders': items_in_subfolders
            })
        except OSError as e:
            logger.info(f"  - Error scanning folder '{folder_path}': {e}")
    return stats_list


# ----------------------------------------------------
# 4. Data Codes and Paths
# ----------------------------------------------------


def add_data_codes(name_mapping: list[dict]) -> list[dict]:
    """
    Assigns a ``data_code`` to each case record based on its folder layout.

    The data code is a classification of the folder's structure that drives all
    downstream decisions about whether to process the case and where to find its
    DICOM files. Think of it as a simple state machine over the folder contents.

    The possible codes and their meanings:

    - ``READY``          — ``.dcm`` files are at the top level; the folder is
                           immediately readable by the DICOM validator.
    - ``SUBFOLDER_PATH`` — No top-level ``.dcm`` files, but exactly one non-empty
                           subfolder. The subfolder is the real data path. This is
                           the typical layout from PACS systems that create a series
                           directory (e.g. ``bp001/DICOM/*.dcm``).
    - ``DUPLICATE_DATA`` — Either: top-level ``.dcm`` files AND a non-empty subfolder
                           (ambiguous source), or two or more non-empty subfolders
                           (multiple series). Both cases require manual review to
                           decide which series to use.
    - ``EMPTY``          — No ``.dcm`` files found anywhere in the folder.
    - ``MISSING``        — The case number was never found in the raw data at all.
                           These records are added by ``find_missing_cases`` and
                           do not have filesystem stats.

    Only ``READY`` and ``SUBFOLDER_PATH`` cases proceed to DICOM validation.
    All others are logged and excluded from the conversion pipeline.

    Args:
        name_mapping: List of case record dicts, each optionally containing
            ``'direct_items'`` and ``'non_empty_subfolders'`` from ``get_folder_stats``.
            Records without ``'direct_items'`` are treated as ``MISSING``.

    Returns:
        The same list with a ``'data_code'`` key added to every record (in-place).
    """
    logger.info("Assigning data codes...")
    for item in name_mapping:
        # Records injected by find_missing_cases have no filesystem stats.
        if 'direct_items' not in item:
            item['data_code'] = 'MISSING'
            continue

        direct_items = item['direct_items']
        non_empty_subfolders = len(item.get('non_empty_subfolders', []))
        action_code = ''

        if non_empty_subfolders == 0:
            if direct_items == 0:
                action_code = 'EMPTY'
            else:
                action_code = 'READY'
        elif non_empty_subfolders == 1:
            if direct_items == 0:
                # Clean single-subfolder layout — DICOMs are one level down.
                action_code = 'SUBFOLDER_PATH'
            else:
                # DCMs both at the top level and in a subfolder — ambiguous.
                action_code = 'DUPLICATE_DATA'
        else:
            # Two or more non-empty subfolders — multiple series present.
            action_code = 'DUPLICATE_DATA'

        item['data_code'] = action_code
    logger.info("Data codes assigned.")
    return name_mapping


def add_data_paths(name_mapping: list[dict]) -> list[dict]:
    """
    Resolves the filesystem path that the DICOM validator should read for each case.

    The path is relative to the raw data root directory (``base_path``). The
    DICOM validator will prepend ``base_path`` when constructing the full path.

    - ``READY``          → the original folder name (e.g. ``'bp1'``).
    - ``SUBFOLDER_PATH`` → the original name joined with the single non-empty
                           subfolder (e.g. ``'bp2/DICOM'``).
    - All other codes   → the code string itself (e.g. ``'MISSING'``, ``'EMPTY'``,
                           ``'DUPLICATE_DATA'``). The DICOM validator will see this
                           string, recognise it is not a valid path, and skip the case.

    The ``'non_empty_subfolders'`` key is deleted after the path is resolved
    because it is a large list that is no longer needed and would bloat the
    in-memory record and the output CSV.

    Args:
        name_mapping: List of case record dicts with ``'data_code'`` and optionally
            ``'non_empty_subfolders'`` keys already populated.

    Returns:
        The same list with a ``'data_path'`` key added to every record (in-place)
        and ``'non_empty_subfolders'`` removed.
    """
    logger.info("Determining data paths...")
    for item in name_mapping:
        code = item.get('data_code')
        path = ''

        if code == 'READY':
            path = item['original_name']
        elif code == 'SUBFOLDER_PATH':
            subfolder_name = item['non_empty_subfolders'][0]
            path = os.path.join(item['original_name'], subfolder_name)
        elif code in ('MISSING', 'EMPTY', 'DUPLICATE_DATA'):
            # Sentinel value — downstream steps treat non-path strings as skip signals.
            path = code

        item['data_path'] = path

        # Remove the subfolder list: it has served its purpose and is not needed
        # downstream or in the output CSV.
        if 'non_empty_subfolders' in item:
            del item['non_empty_subfolders']

    return name_mapping


# ----------------------------------------------------
# 5. Pipeline Orchestration
# ----------------------------------------------------


def organize_data(case_folders: list[str], RAW_DATA_PATH: str) -> list[dict]:
    """
    Runs the full name-standardisation and folder-analysis sub-pipeline.

    This is the only function called by ``data_cleaner.py``. It chains together
    all the steps in this module and returns a single list of case record dicts
    that is ready to be handed off to the DICOM validation step.

    Each returned dict contains at minimum:
      ``original_name``  — the raw folder name as found on disk
      ``fixed_name``     — the canonical BP{NNN} name
      ``data_code``      — folder layout classification (READY, SUBFOLDER_PATH, …)
      ``data_path``      — relative path to hand to the DICOM reader
      ``total_dcms``     — total DICOM file count across direct + subfolder levels

    Cases in the range BP001–BP999 that are absent from the raw data are also
    included, with ``data_code = 'MISSING'``, so the final audit log accounts for
    every expected case number.

    Processing steps:
      1. Map raw folder names to canonical BP{NNN} names.
      2. Collect folder layout statistics (DCM counts, subfolder counts).
      3. Merge the statistics into the name mapping records.
      4. Compute a total DCM count per case.
      5. Identify and append missing case numbers (BP001–BP999 gaps).
      6. Assign a data code to every record.
      7. Resolve the data path for each record.

    Args:
        case_folders: Raw folder names from ``get_subfolders`` (relative names only).
        RAW_DATA_PATH: Absolute path to the raw data directory.

    Returns:
        List of fully-populated case record dicts, one per discovered case plus
        one for each missing case number in the BP001–BP999 range.
    """
    # 1. Generate the mapping from original to new names.
    name_mapping = generate_new_names(case_folders)

    # 2. Get statistics for the original folders.
    folder_stats = get_folder_stats(RAW_DATA_PATH, case_folders)

    # 3. Merge statistics into the name mapping data.
    #    Index stats by folder name for O(1) lookup per case.
    stats_map = {stat['folder']: stat for stat in folder_stats}

    for item in name_mapping:
        original_name = item['original_name']
        if original_name in stats_map:
            item.update(stats_map[original_name])
            del item['folder']  # 'folder' duplicates 'original_name'; remove to avoid confusion

    # 4. Add total_dcms count: files directly in the folder + files in its subfolders.
    #    This single number is used in the audit CSV as a quick sanity check on
    #    how many slices were found before any were validated or discarded.
    for item in name_mapping:
        if 'direct_items' in item:
            item['total_dcms'] = item.get('direct_items', 0) + item.get('items_in_subfolders', 0)

    # 5. Find missing cases and add them to our main list.
    #    These are case numbers in BP001–BP999 that have no folder on disk at all.
    missing_cases = find_missing_cases(name_mapping)
    combined_mapping = name_mapping + missing_cases
    logger.info(f"Found {len(missing_cases)} missing case entries.")

    # 6. Add action codes to the complete list (including missing cases).
    data_with_codes = add_data_codes(combined_mapping)

    # 7. Add the final data paths.
    organized_data = add_data_paths(data_with_codes)

    return organized_data
