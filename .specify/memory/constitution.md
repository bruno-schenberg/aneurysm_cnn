<!--
Sync Impact Report:
- Version change: Uninitialized → 1.0.0
- List of principles:
  - I. Scientific Reproducibility & Rigor (New)
  - II. Modularity & Separation of Concerns (New)
  - III. Modern Technical Standards (New)
  - IV. Testability & Verification (New)
  - V. Clean Architecture & Readable Code (New)
- Added sections: Governance, Organization
- Templates requiring updates: None (Generic templates are compatible with these principles)
-->

# Aneurysm CNN Constitution

## Core Principles

### I. Scientific Reproducibility & Rigor
As a scientific research project, the primary deliverable is not just code, but verifiable truth. Results must be replicable by independent researchers.
*   **Determinism**: Fixed random seeds MUST be used and logged for all non-deterministic operations (dataset splitting, weight initialization, training shuffling).
*   **Configurability**: All experimental parameters (hyperparameters, architecture choices) MUST be defined in configuration files (e.g., `experiments.json`), never hardcoded in logic.
*   **Artifacts**: Every training run MUST produce a unique, traceable output directory containing its configuration, logs, and results.

### II. Modularity & Separation of Concerns
The project consists of distinct phases that must not be entangled. 'Spaghetti code' is strictly prohibited.
*   **Phase Isolation**: The `data_engine` (dataset generation, run once) and `training_engine` (experimentation, run many times) MUST remain separate sub-projects. They should not import from each other directly; they communicate via file artifacts (datasets on disk).
*   **Single Responsibility**: Each module or script should have a single, clear purpose (e.g., `data_cleaner.py` cleans, `models.py` defines architectures).

### III. Modern Technical Standards
We utilize state-of-the-art ecosystem tools to ensure performance and standardization.
*   **Library Selection**: PyTorch, MONAI, and NiBabel are the standard libraries for deep learning and medical imaging. Custom implementations of standard layers or metrics should be avoided in favor of library implementations unless strictly necessary for research novelty.
*   **Type Hinting**: Python type hints SHOULD be used for function signatures to aid readability and static analysis.

### IV. Testability & Verification
Code must be demonstrably accurate. In scientific computing, a silent bug is worse than a crash.
*   **Unit Testing**: Core utility functions (especially mathematical or data manipulation logic) MUST be covered by unit tests.
*   **Validation Strategy**: Cross-validation (k-fold) and a strict Hold-Out Test Set are mandatory for final evaluation. Information leakage between train/val/test splits is a critical failure.
*   **Sanity Checks**: Pipelines SHOULD include assert statements or runtime checks for data shape and integrity (e.g., ensuring no patient ID overlaps between splits).

### V. Clean Architecture & Readable Code
The codebase is a communication tool for researchers.
*   **Self-Documenting**: Variable and function names MUST be descriptive and verbose enough to explain their scientific meaning (e.g., `aneurysm_volume_mm3` vs `vol`).
*   **Contextual Comments**: Comments should explain the *why* (scientific rationale), not the *what* (syntax).
*   **Directory Structure**: The project file structure MUST reflect the logical separation of the engines (see Organization section).

## Organization

The project is strictly divided into two logical engines:

1.  **Data Engine** (`data_engine/`):
    *   Responsible for parsing raw data, cleaning, and generating training-ready NIfTI/Tensor datasets.
    *   Output: Static dataset files on disk.
2.  **Training Engine** (`training_engine/`):
    *   Responsible for loading processed datasets, defining models, and executing training/evaluation loops.
    *   Input: Static dataset files from Data Engine.

## Governance

This constitution defines the non-negotiable standards for the Aneurysm CNN project.

*   **Amendments**: Changes to these principles require a version bump and justification (e.g., switching core libraries).
*   **Compliance**: All Pull Requests and Code Reviews MUST verify adherence to these principles.
*   **Version Strategy**: Semantic Versioning (MAJOR.MINOR.PATCH).
    *   MAJOR: Change in core libraries or fundamental workflow (e.g., merging engines).
    *   MINOR: Addition of new principles or significant modules.
    *   PATCH: Clarifications or non-breaking refactors.

**Version**: 1.0.0 | **Ratified**: 2026-02-18 | **Last Amended**: 2026-02-18
