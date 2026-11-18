"""Shared helpers for cross-model analysis scripts."""

from dlorchestra.analysis.ids import regex_id_extractor

# Extract the BPxxxx exam ID from any path format (falls back to the file
# stem when the pattern doesn't match — see dlorchestra.analysis.ids).
extract_exam_id = regex_id_extractor(r"(BP\d{4})")
