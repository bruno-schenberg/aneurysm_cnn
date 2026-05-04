# Agent Guidelines

Welcome to the `aneurysm_cnn` project! When working on this codebase as an AI assistant, please adhere to the following principles and standards:

## 1. Code Quality & Style
- **Simplicity & Readability**: Code must be simple and easy to read. Avoid overly clever or convoluted logic.
- **Meaningful Comments**: Comments must explain *why* the code does what it does, not *what* the code does. Assume the reader understands the language, but may not understand the specific domain logic or reasoning behind a design choice.
- **Type Hinting**: All new or refactored Python code should include type hints for function arguments and return values.
- **Linting & Formatting**: We use `ruff` to lint and format our code. Ensure code complies with `ruff` standards.
- **Docstrings**: We use **Google-style docstrings** to describe the purpose, arguments, and return values of our functions and classes.

## 2. Workflow & Refactoring Strategy
- **Gradual Refactoring**: The long-term goal of this branch is to refactor and rewrite the application. This will take a long time. **Do not try to tackle everything at once.**
- **Focused Scope**: Work with only a few files at a time to keep changes manageable.
- **Write Tests as We Go**: Every new feature, bug fix, or refactored component must be accompanied by relevant tests.
- **Keep Documentation Updated**: Any changes to code, project structure, or architecture must be followed by corresponding updates to the documentation, specifically `README.md` and/or `PIPELINE.md`.

## 3. Local Development & Testing
- **Testing Framework**: We use `pytest` for all our testing.
- **Environment**: We run our code and tests locally using standard Python virtual environments. **There are two distinct `.venv` directories**:
  - `data_engine/.venv/` (Used exclusively when working in the `data_engine` folder)
  - `training_engine/.venv/` (Used exclusively when working in the `training_engine` folder)
  - When making code changes or running tests, **always activate the correct `.venv` for the engine you are working on.**
- **Sample Datasets**: Use sample datasets where needed for testing and validation to keep test execution fast and local.

## 4. Architecture & Isolation
- **Strict Separation of Engines**: The project is divided into `data_engine` and `training_engine`. **Their code must stay strictly separated.**
- **Modularity**: The `data_engine` runs once to generate datasets to disk. The `training_engine` later loads these datasets from disk to train the models. They do not overlap or import from each other.
- **Sequential Refactoring**: We will refactor and test one engine at a time to respect this isolation.

## 5. Collaboration & Project Standards
- If you have other questions, suggestions on project standards, or encounter ambiguities, ask! We will work them out together.
