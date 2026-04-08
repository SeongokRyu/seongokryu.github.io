---
title: "Code Quality for Beginners Part 4: Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup"
date: 2026-04-08 12:00:00 +0900
categories: [Engineering]
tags: [code-quality, claude-md, harness-engineering, ai-agent, system-prompt, claude-code, beginner]
---

You've built the rules. Now make them automatic — for humans and AI agents alike.

**Code Quality for Beginners**

*This is Part 4 of a 4-part series on Code Quality and Collaboration.*

- **Part 1:** [Write Clean Code — Formatting, Linting, and Data Validation](/posts/code-quality-beginners-part1-clean-code/)
- **Part 2:** [Test and Automate — From pytest to CI/CD Pipelines](/posts/code-quality-beginners-part2-test-and-automate/)
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4** (this post): Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup

---

Parts 1 through 3 established a complete set of rules: how to format code, how to test it, how to commit and review it. But rules that live in a wiki or in people's heads **decay**. New hires don't know them. AI agents don't know them. And even experienced teammates forget them at 11 PM on a Friday deploy.

This post is about making those rules **structural** — written once, enforced everywhere, followed by both humans and AI agents.

---

## What is CLAUDE.md?

**CLAUDE.md is a project-level system prompt that lives in your repository root. Every time an AI agent (such as Claude Code) opens your project, it reads this file first — like onboarding documentation that the agent always follows.**

Think of it this way:

```
A new human teammate:
  1. Reads the README
  2. Asks the team questions
  3. Learns conventions over weeks
  4. Sometimes forgets or drifts

An AI agent with CLAUDE.md:
  1. Reads CLAUDE.md
  2. Follows conventions immediately
  3. Never forgets
  4. Never drifts (unless the file drifts)
```

CLAUDE.md is not magic. It is **your team's standards, written in a format that a machine can follow.** The same rules you would explain to a new hire — but encoded once and applied consistently.

---

## What Goes Into CLAUDE.md?

Every section from Parts 1-3 maps directly to something in CLAUDE.md:

```
+-----------------------------------+----------------------------------------------+
| Series Section                    | CLAUDE.md Entry                              |
+-----------------------------------+----------------------------------------------+
| Part 1: Formatting & Linting      | Tool commands: ruff check, ruff format       |
| Part 1: Type Checking             | Command: mypy src/ --strict                  |
| Part 1: Data Validation           | Rule: "Use Pydantic for all external input"  |
| Part 2: Testing                   | Command: pytest --cov --cov-fail-under=80    |
| Part 2: CI/CD                     | Pipeline expectations and constraints        |
| Part 3: Commit Conventions        | Rule: "Use Conventional Commits"             |
| Part 3: Branch Naming             | Pattern: <type>/<ticket>-<description>       |
| Part 3: PR Practices              | Rule: "PRs under 400 lines"                  |
| Part 3: Security                  | Rule: "Never commit secrets"                 |
| Part 3: Dependencies              | Command: uv sync --frozen                    |
+-----------------------------------+----------------------------------------------+
```

### What NOT to Put in CLAUDE.md

- **Secrets** — never write API keys or passwords in any committed file
- **Ephemeral task details** — "we're working on feature X this sprint" goes stale immediately
- **Things derivable from code** — file paths and architecture the agent can discover by reading `src/`
- **Detailed API documentation** — that belongs in docstrings and OpenAPI specs

CLAUDE.md should contain **rules and commands** — things the agent cannot infer from the codebase alone.

---

## Writing an Effective CLAUDE.md — A Practical Example

Here is a complete CLAUDE.md for a Python/FastAPI project. Every line traces back to something we covered in Parts 1-3.

```markdown
# CLAUDE.md

## Project Overview
Python 3.12 / FastAPI / PostgreSQL / uv for dependency management

## Commands
- Install dependencies: `uv sync`
- Run dev server: `uvicorn src.myproject.main:app --reload`
- Lint: `ruff check . --fix && ruff format .`
- Type check: `mypy src/`
- Test: `pytest --cov=src --cov-fail-under=80`
- Test (unit only): `pytest tests/unit/`
- Test (integration only): `pytest -m integration`
- Pre-commit (all files): `pre-commit run --all-files`

## Code Conventions
- All external input (API requests, config, file reads) must pass through
  a Pydantic BaseModel before entering business logic
- Import order: stdlib → third-party → local (enforced by ruff I rules)
- Naming: snake_case for functions/variables, PascalCase for classes,
  UPPER_SNAKE_CASE for constants
- No raw string SQL — use SQLAlchemy parameterized queries only
- No bare except — catch specific exceptions
- Raise domain exceptions from core/exceptions.py
- Type annotations required on all public functions

## Git Conventions
- Branch naming: <type>/<ticket>-<description> (e.g., feat/PROJ-123-user-auth)
- Commit messages follow Conventional Commits: <type>(<scope>): <description>
- PR size: under 400 lines of logic changes. Split larger changes.
- Mechanical changes (formatting, renaming) in separate PRs from logic changes

## Architecture
src/myproject/
  core/       — config (pydantic-settings), exceptions, constants
  models/     — Pydantic schemas and SQLAlchemy models
  services/   — business logic layer
  api/routes/ — FastAPI endpoint definitions
  api/deps.py — FastAPI dependency injection

## Testing
- Unit tests: tests/unit/ — mock external dependencies, test one function
- Integration tests: tests/integration/ — real DB via fixtures, mark with
  @pytest.mark.integration
- Use pytest fixtures (conftest.py), not setUp/tearDown
- Test naming: test_<what>_<condition>_<expected> (e.g.,
  test_create_user_duplicate_email_raises_error)
```

### Why Each Section Exists

**Commands** — The agent needs to know exactly how to run things. "Make sure the code is tested" is vague. `pytest --cov=src --cov-fail-under=80` is unambiguous.

**Code Conventions** — These are rules the agent cannot discover by reading existing code. Existing code might have exceptions or legacy patterns. CLAUDE.md states the **current standard**.

**Git Conventions** — Without this, the agent might use any commit style. With it, every commit it creates follows your team's format.

**Architecture** — A brief map so the agent knows where to put new code. It won't create a `/helpers/random_utils.py` if it knows the project structure.

**Testing** — Tells the agent which test style to use, where to put tests, and how to name them. Consistency in tests matters as much as in production code.

---

## What is a Harness?

**A harness is the structured environment around an AI agent that constrains and guides its behavior. It includes the system prompt (CLAUDE.md), available tools, reusable skills, and automated hooks.**

```
+-------------------------------------------------------------+
|                         HARNESS                              |
|                                                              |
|  +-- CLAUDE.md (System Prompt) --------------------------+  |
|  |  Project rules, conventions, commands                  |  |
|  |  "What the agent should know and follow"               |  |
|  +--------------------------------------------------------+  |
|                                                              |
|  +-- Skills (Reusable Prompt Templates) ------------------+  |
|  |  /commit  — generate a conventional commit             |  |
|  |  /review  — review code against team checklist         |  |
|  |  /test    — write tests following project patterns     |  |
|  |  "Pre-built workflows the agent can execute"           |  |
|  +--------------------------------------------------------+  |
|                                                              |
|  +-- Hooks (Automated Triggers) --------------------------+  |
|  |  pre-commit: ruff check, ruff format, detect-secrets   |  |
|  |  post-edit:  auto-lint modified files                  |  |
|  |  "Rules enforced automatically, even if agent forgets" |  |
|  +--------------------------------------------------------+  |
|                                                              |
|  +-- Tools (External Capabilities) ----------------------+   |
|  |  MCP servers, CLI tools, database access, APIs         |  |
|  |  "What the agent can interact with"                    |  |
|  +--------------------------------------------------------+  |
|                                                              |
|           AI Agent operates within these boundaries          |
+-------------------------------------------------------------+
```

Without a harness, an AI agent is a powerful but **unpredictable** assistant. It might format code differently from your team. It might commit with random messages. It might skip tests.

With a harness, the agent's **default behavior matches your team's standards**. It doesn't need to be "smart" about your conventions — the conventions are embedded in its operating environment.

---

## Five Principles for Building a Good Harness

### Principle 1: Encode, Don't Repeat

If you've corrected the agent on the same thing twice, that correction belongs in CLAUDE.md. Your time is spent once; the agent follows the rule forever.

```
First conversation:   "Use Pydantic for API input, not raw dicts."
Second conversation:  "I told you before, use Pydantic..."
Third conversation:   (should never happen)

Fix: Add to CLAUDE.md:
  "All external input must pass through a Pydantic BaseModel"
```

### Principle 2: Commands Over Descriptions

Agents execute commands reliably. They interpret descriptions loosely.

```
Bad:    "Make sure the code is well-tested and properly formatted."
Good:   "Run: ruff format . && ruff check . --fix && pytest --cov=src --cov-fail-under=80"

Bad:    "Follow our coding style."
Good:   "Naming: snake_case for functions, PascalCase for classes, UPPER_SNAKE_CASE for constants."
```

Descriptions invite interpretation. Commands leave no room for ambiguity.

### Principle 3: Constraints Are Features

Limiting what the agent can do makes its output **predictable**. Unconstrained agents produce creative but inconsistent results.

```
Bad:    "Use appropriate error handling."
Good:   "Raise domain exceptions from core/exceptions.py. Never use bare except."

Bad:    "Put the code in a reasonable location."
Good:   "Business logic goes in services/. API endpoints go in api/routes/."
```

Every constraint eliminates a category of wrong answers.

### Principle 4: Layer Your Guardrails

CLAUDE.md states intent. Hooks enforce it. CI verifies it. Multiple layers mean no single point of failure.

```
Layer 1: CLAUDE.md says          "All code must pass ruff check."
Layer 2: Pre-commit hook runs    ruff check on every commit
Layer 3: CI pipeline runs        ruff check on every push
                                      |
                                      v
            Agent cannot skip the rule, even if it "forgets" CLAUDE.md.
            The hook physically blocks the commit.
```

This is **defense in depth** — the same principle that makes Part 2's feedback speed ladder effective.

### Principle 5: Iterate From Feedback

A harness is not a one-time setup. It is a **living document** that improves every time something goes wrong.

```
Agent generates a commit: "updated stuff"
     |
     v
You correct it: "Use Conventional Commits — feat(scope): description"
     |
     v
You update CLAUDE.md: add commit format rule
     |
     v
Agent never makes that mistake again
```

This is the same feedback loop as human team culture, except **the "new hire" never forgets and never needs to be told the same thing twice**.

---

## The Feedback Loop

Every standard from Parts 1-3 feeds into this virtuous cycle:

```
+------------------------------------------+
|                                          |
|   Team defines standards                 |
|   (Parts 1, 2, 3)                        |
|         |                                |
|         v                                |
|   Encode into CLAUDE.md                  |
|   + hooks + CI                           |
|         |                                |
|         v                                |
|   Agent follows standards automatically  |
|         |                                |
|         v                                |
|   Agent makes a mistake                  |
|         |                                |
|         v                                |
|   Human corrects the output              |
|   + updates CLAUDE.md                    |
|         |                                |
|         v                                |
|   Agent never repeats that mistake       |
|         |                                |
|         +--- loop back to top ---+       |
|                                          |
+------------------------------------------+
```

**Without CLAUDE.md:** Every conversation starts from zero. The agent asks "how should I format this?" every time.

**With CLAUDE.md:** Every conversation starts from your team's accumulated knowledge. The agent already knows your formatter, your test runner, your commit format, your project structure.

---

## Putting It ALL Together

Here are the complete configuration files that wire together everything from Parts 1-4. Copy, adapt, and commit them.

### The Complete `pyproject.toml`

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "httpx>=0.27",
    "sqlalchemy>=2.0",
    "uvicorn>=0.34",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "pytest-asyncio>=0.25",
    "pytest-xdist>=3.5",
    "ruff>=0.8",
    "mypy>=1.14",
    "pre-commit>=4.0",
    "pip-audit>=2.7",
    "bandit>=1.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# --- Part 1: Formatting & Linting ---

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes
    "I",         # isort
    "N",         # pep8-naming
    "UP",        # pyupgrade
    "B",         # flake8-bugbear
    "A",         # flake8-builtins
    "SIM",       # flake8-simplify
    "TCH",       # type-checking imports
    "RUF",       # ruff-specific
    "S",         # security (bandit rules)
    "PTH",       # pathlib
]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["myproject"]

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true

# --- Part 1: Type Checking ---

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# --- Part 2: Testing ---

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=src --cov-report=term-missing"
markers = [
    "slow: marks tests as slow-running",
    "integration: marks integration tests",
]

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.",
]
```

### The Complete `.pre-commit-config.yaml`

```yaml
# Part 2: Pre-commit hooks — runs before every git commit

repos:
  # Formatting and linting (Part 1)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # File hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: check-merge-conflict
      - id: detect-private-key    # Part 3: Security

  # Type checking (Part 1)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]

  # Secret detection (Part 3: Security)
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

### The Complete GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
# Part 2: CI pipeline — runs on every push and PR

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff mypy pydantic
      - run: ruff format --check .
      - run: ruff check .
      - run: mypy src/

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src --cov-report=xml --cov-fail-under=80
      - uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12'

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install pip-audit bandit
      - run: pip-audit
      - run: bandit -r src/
```

### Suggested Project Structure

```
myproject/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                 # Part 2: CI pipeline
│   └── CODEOWNERS                 # Part 3: Auto-assign reviewers
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── py.typed                # PEP 561 type marker
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Part 1: pydantic-settings
│       │   ├── exceptions.py       # Domain exceptions
│       │   └── constants.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── user.py             # Part 1: Pydantic models
│       ├── services/
│       │   ├── __init__.py
│       │   └── user_service.py     # Business logic
│       └── api/
│           ├── __init__.py
│           ├── deps.py             # FastAPI dependencies
│           └── routes/
│               └── users.py        # API endpoints
├── tests/
│   ├── conftest.py                 # Part 2: Shared fixtures
│   ├── unit/
│   │   └── test_user_service.py    # Part 2: Unit tests
│   ├── integration/
│   │   └── test_user_api.py        # Part 2: Integration tests
│   └── e2e/
│       └── test_checkout_flow.py   # Part 2: E2E tests
├── .env.example                    # Part 3: Template, no real secrets
├── .gitignore                      # Part 3: Exclude secrets, caches
├── .pre-commit-config.yaml         # Part 2: Pre-commit hooks
├── .secrets.baseline               # Part 3: detect-secrets baseline
├── CLAUDE.md                       # Part 4: AI agent system prompt
├── pyproject.toml                  # Parts 1-2: All tool configuration
└── uv.lock                         # Part 3: Exact dependency versions
```

Every file in this tree traces back to a specific practice from the series. Nothing is here "just in case." Everything earns its place.

---

## Series Conclusion

This series covered a lot of ground. Here is the entire journey in one diagram:

```
Part 1: WRITE CLEAN CODE
  ruff format + ruff check + mypy + Pydantic
  "Code is consistent, typed, and validated."
         |
         v
Part 2: TEST AND AUTOMATE
  pytest + coverage + GitHub Actions + pre-commit
  "Every change is verified automatically."
         |
         v
Part 3: COLLABORATE LIKE A PRO
  Conventional Commits + PR practices + code review + security
  "The team works together without friction."
         |
         v
Part 4: ENCODE YOUR STANDARDS
  CLAUDE.md + harness + integrated config
  "Both humans and AI agents follow the same rules."
```

Three takeaways:

1. **Start small.** Install ruff, write one pytest file, add a pre-commit config. That alone puts you ahead of most projects.
2. **Automate everything you can.** Formatting, linting, testing, security scanning — if a machine can check it, don't rely on a human to remember.
3. **Write a CLAUDE.md.** Even if you don't use AI agents today, the exercise of encoding your standards into a single file clarifies what your standards actually are. And when you do start using agents, they'll follow your rules from day one.

These practices compound. Each one is simple on its own. Together, they transform how a team builds software — predictable, reviewable, deployable, and now, encodable.

---

*Full series:*
- **Part 1:** [Write Clean Code — Formatting, Linting, and Data Validation](/posts/code-quality-beginners-part1-clean-code/)
- **Part 2:** [Test and Automate — From pytest to CI/CD Pipelines](/posts/code-quality-beginners-part2-test-and-automate/)
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)
