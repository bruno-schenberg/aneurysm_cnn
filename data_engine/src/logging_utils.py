import logging
import csv
import os
import json
from datetime import datetime


class _JsonlHandler(logging.FileHandler):
    """Logging handler that writes each log record as a JSON line (JSONL format)."""

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


def setup_logger(log_path: str = "ingestion.log"):
    """
    Configures the standard logging to write to both a file (JSONL format) and the console.
    """
    logger = logging.getLogger("dicom_ingestion")
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times if setup_logger is called again
    if logger.handlers:
        return logger

    # File handler writes JSONL for machine-readable audit trail (FR-007)
    file_handler = _JsonlHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Console handler for human-readable progress
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def write_audit_log(results: list, output_path: str = "ingestion_summary.csv"):
    """
    Writes the final audit log (CSV) containing the conversion results.
    
    Args:
        results: List of ConversionResult dictionaries (exam_name, status, reason, output_path).
        output_path: Path to the output CSV file.
    """
    fieldnames = ['exam_name', 'status', 'reason', 'output_path']
    
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    except Exception as e:
        logging.getLogger("dicom_ingestion").error(f"Failed to write audit log to {output_path}: {e}")

def log_jsonl(data: dict, log_path: str = "ingestion_metrics.jsonl"):
    """
    Logs data as a single line in a JSONL file for machine readability.
    """
    data['timestamp'] = datetime.now().isoformat()
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception as e:
        logging.getLogger("dicom_ingestion").error(f"Failed to write JSONL log: {e}")
