"""
logging_utils.py

Configures the pipeline logger and writes the two output log artefacts
produced at the end of every data engine run.

## Two output formats, two audiences

The pipeline produces logs in two formats simultaneously:

  1. **Console output** — human-readable progress messages printed to the
     terminal while the pipeline is running. INFO level only, so debug noise
     is suppressed. Format: ``LEVEL: message``.

  2. **JSONL file** (``ingestion.log``) — machine-readable log where every
     message is a JSON object on its own line. DEBUG level, so every event
     is captured. Format: one JSON object per line with ``timestamp``,
     ``level``, ``logger``, and ``message`` fields.

JSONL (JSON Lines) is chosen over plain text for the file log because it can
be parsed programmatically without any text parsing logic — each line is a
complete, self-contained JSON record. This makes it straightforward to filter
by level, search by message content, or load into a dataframe for post-run
analysis.

## Pipeline audit CSV

In addition to the runtime log, ``write_audit_log`` writes
``ingestion_summary.csv`` at the end of each run. This CSV records the final
outcome for every case that reached the NIfTI conversion step: whether it
succeeded, was skipped (already existed), or failed and why. Combined with
the validation summary written by ``data_cleaner.py``, these two files give
complete traceability from raw DICOM folder to output NIfTI file.

Public API (called by data_cleaner.py):
  - setup_logger(log_path)           : initialise and return the pipeline logger
  - write_audit_log(results, path)   : write the conversion outcome CSV
"""

import logging
import csv
import json
from datetime import datetime


# ----------------------------------------------------
# 1. JSONL Handler
# ----------------------------------------------------


class _JsonlHandler(logging.FileHandler):
    """
    A logging handler that writes each log record as a single JSON line.

    Extends Python's standard ``FileHandler``, overriding only ``emit()``
    to change the output format from plain text to JSON. Every other
    behaviour — file opening, rotation, error handling — is inherited.

    Each line written to the file is a complete, valid JSON object:
      ``{"timestamp": "...", "level": "INFO", "logger": "...", "message": "..."}``

    ``self.flush()`` is called after every write so that if the pipeline
    crashes mid-run, all records up to that point are flushed to disk rather
    than sitting in a buffer. Exception details are included in the record
    when ``exc_info`` is present (e.g. when the logger is called with
    ``logger.error(..., exc_info=True)``).

    ``handleError`` is called on any formatting or write failure — this is
    the standard Python logging contract and prevents a logging failure from
    crashing the pipeline itself.
    """

    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exception"] = self.formatException(record.exc_info)
            self.stream.write(json.dumps(entry) + '\n')
            self.flush()
        except Exception:
            self.handleError(record)


# ----------------------------------------------------
# 2. Logger Setup
# ----------------------------------------------------


def setup_logger(log_path: str = "ingestion.log") -> logging.Logger:
    """
    Initialises and returns the ``dicom_ingestion`` named logger.

    All modules in the data engine retrieve this logger by name with
    ``logging.getLogger("dicom_ingestion")``. Using a named logger rather
    than the root logger means the pipeline's output is isolated from any
    other logging that might be happening in third-party libraries.

    Two handlers are attached:

    **File handler** (JSONL, DEBUG level):
    Captures every log event, including debug messages, in machine-readable
    JSONL format. Useful for post-run auditing and error diagnosis.

    **Console handler** (plain text, INFO level):
    Prints human-readable progress to the terminal during the run. Debug
    messages are suppressed so the console output stays clean.

    The guard ``if logger.handlers: return logger`` prevents duplicate
    handlers from being added if ``setup_logger`` is called more than once
    in the same Python process (e.g. in tests). Without this guard, each
    call would add a new pair of handlers and every log message would be
    printed multiple times.

    Args:
        log_path: Path to the JSONL log file. Created if it does not exist.

    Returns:
        The configured ``dicom_ingestion`` logger instance.
    """
    logger = logging.getLogger("dicom_ingestion")
    logger.setLevel(logging.DEBUG)

    # Guard: do not add handlers again if already configured.
    if logger.handlers:
        return logger

    # File handler — JSONL, captures everything including DEBUG messages.
    file_handler = _JsonlHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Console handler — plain text, INFO and above only.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------
# 3. Audit Logging
# ----------------------------------------------------


def write_audit_log(results: list, output_path: str = "ingestion_summary.csv") -> None:
    """
    Writes the NIfTI conversion outcomes to a CSV file.

    This CSV is the final record of what happened to every case that reached
    the conversion step. It is written by ``data_cleaner.py`` after all
    conversions complete and is combined with the validation summary CSV to
    give full end-to-end traceability.

    Each row corresponds to one case and contains four fields:
      ``exam_name``   — canonical case name (e.g. ``BP001``)
      ``status``      — ``'success'``, ``'skipped'``, or ``'failed'``
      ``reason``      — empty on success; error description on failure;
                        ``'Already exists'`` when skipped
      ``output_path`` — absolute path to the ``.nii.gz`` file, or empty on failure

    Cases that never reached conversion (failed validation, missing class,
    etc.) are also included by ``data_cleaner._build_audit_log``, which adds
    them with ``status='failed'`` and the validation status as the reason.
    This ensures the CSV accounts for every input case, not just those
    that were converted.

    Args:
        results: List of result dicts as returned by
                 ``nifti_utils.process_and_convert_exams`` and extended by
                 ``data_cleaner._build_audit_log``.
        output_path: Path where the CSV will be written. Overwritten if it
                     already exists.
    """
    fieldnames = ['exam_name', 'status', 'reason', 'output_path']

    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    except Exception as e:
        logging.getLogger("dicom_ingestion").error(f"Failed to write audit log to {output_path}: {e}")


def log_jsonl(data: dict, log_path: str = "ingestion_metrics.jsonl") -> None:
    """
    Appends a single dict as a JSON line to a JSONL metrics file.

    A timestamp is added to ``data`` before writing. The file is opened in
    append mode so successive calls accumulate records rather than overwriting.

    This function is a standalone utility and is not called by the main
    pipeline in ``data_cleaner.py``. It is available for ad-hoc metrics
    logging if needed in future pipeline extensions.

    Args:
        data: Dict of key-value pairs to record. Modified in-place to add
              a ``'timestamp'`` key before serialisation.
        log_path: Path to the JSONL file. Created if it does not exist,
                  appended to if it does.
    """
    data['timestamp'] = datetime.now().isoformat()
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception as e:
        logging.getLogger("dicom_ingestion").error(f"Failed to write JSONL log: {e}")
