---
title: "Code Quality for Beginners Part 2: Test and Automate — From pytest to CI/CD Pipelines"
date: 2026-04-08 10:00:00 +0900
categories: [Engineering]
tags: [code-quality, testing, pytest, ci-cd, github-actions, pre-commit, coverage, beginner]
---

Your code is clean and validated. Now prove it works — and keep proving it automatically, every time something changes.

**Code Quality for Beginners**

*This is Part 2 of a 4-part series on Code Quality and Collaboration.*

- **Part 1:** [Write Clean Code — Formatting, Linting, and Data Validation](/posts/code-quality-beginners-part1-clean-code/)
- **Part 2** (this post): Test and Automate — From pytest to CI/CD Pipelines
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)

---

In [Part 1](/posts/code-quality-beginners-part1-clean-code/), we made code clean and validated. But clean code can still be **wrong** code. Tests are the evidence that your code does what you claim. Automation is what makes that evidence **continuous** — generated on every push, not just when someone remembers to run it.

This post covers the testing pyramid, how to write effective tests with pytest, what coverage actually means, and how CI/CD pipelines and pre-commit hooks catch problems before they reach your teammates.

---

## The Testing Pyramid

Not all tests are created equal. The **testing pyramid** is a model for how many tests of each type you should write:

```
                /          \
               /    E2E     \           Few — slow, expensive, fragile
              /              \
             /________________\
            /                  \
           /   Integration      \       Some — moderate speed and cost
          /                      \
         /________________________\
        /                          \
       /       Unit Tests           \   Many — fast, cheap, reliable
      /______________________________\
```

- **Unit tests** verify a single function or class in isolation. They run in **milliseconds**.
- **Integration tests** verify that multiple components work together (your code + a real database, for example). They run in **seconds**.
- **E2E (end-to-end) tests** verify the entire system from the user's perspective. They run in **seconds to minutes**.

The rule: **write many unit tests, some integration tests, and few E2E tests.** This gives you fast feedback on most changes and high-confidence checks on critical paths.

---

## Unit Tests

**A unit test verifies that a single function produces the correct output for a given input, in isolation from everything else.**

### Your First Test

```python
# src/myproject/calculator.py

def calculate_total(price: float, quantity: int, tax_rate: float = 0.1) -> float:
    """Calculate total price including tax."""
    subtotal = price * quantity
    return round(subtotal * (1 + tax_rate), 2)
```

```python
# tests/unit/test_calculator.py

from myproject.calculator import calculate_total

def test_basic_calculation():
    assert calculate_total(10.0, 3) == 33.0  # 10 * 3 * 1.1

def test_zero_quantity():
    assert calculate_total(10.0, 0) == 0.0

def test_custom_tax_rate():
    assert calculate_total(100.0, 1, tax_rate=0.2) == 120.0

def test_rounding():
    # 9.99 * 3 * 1.1 = 32.967 -> should round to 32.97
    assert calculate_total(9.99, 3) == 32.97
```

Run with:

```bash
$ pytest tests/unit/test_calculator.py -v
tests/unit/test_calculator.py::test_basic_calculation    PASSED
tests/unit/test_calculator.py::test_zero_quantity        PASSED
tests/unit/test_calculator.py::test_custom_tax_rate      PASSED
tests/unit/test_calculator.py::test_rounding             PASSED

4 passed in 0.02s
```

### Parametrize — Test Many Cases at Once

When you have many input/output pairs, `@pytest.mark.parametrize` avoids repetitive test functions:

```python
import pytest
from myproject.calculator import calculate_total

@pytest.mark.parametrize("price, quantity, tax_rate, expected", [
    (10.0,  3, 0.1,  33.0),
    (10.0,  0, 0.1,  0.0),
    (100.0, 1, 0.2,  120.0),
    (9.99,  3, 0.1,  32.97),
    (0.0,   5, 0.1,  0.0),
    (50.0,  2, 0.0,  100.0),   # Zero tax
])
def test_calculate_total(price, quantity, tax_rate, expected):
    assert calculate_total(price, quantity, tax_rate) == expected
```

Six test cases, one function. Each runs independently — if one fails, the others still execute.

### Fixtures — Reusable Test Setup

**Fixtures** are pytest's way of sharing setup logic across tests:

```python
# tests/conftest.py — shared fixtures available to all tests

import pytest
from myproject.models import User

@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(name="Alice", age=30, email="alice@example.com")

@pytest.fixture
def admin_user():
    return User(name="Admin", age=25, email="admin@example.com", is_admin=True)
```

```python
# tests/unit/test_user.py

def test_user_display_name(sample_user):
    # sample_user is automatically injected by pytest
    assert sample_user.name == "Alice"

def test_admin_has_elevated_access(admin_user):
    assert admin_user.is_admin is True
```

### When to Mock

**Mocking** replaces a real dependency with a fake one. Use it when a unit test would otherwise need a database, network call, or filesystem:

```python
from unittest.mock import patch
from myproject.services import fetch_weather

@patch("myproject.services.httpx.get")
def test_fetch_weather_returns_temperature(mock_get):
    # Simulate an API response without making a real HTTP call
    mock_get.return_value.json.return_value = {"temp": 22.5, "city": "Seoul"}
    mock_get.return_value.status_code = 200

    result = fetch_weather("Seoul")
    assert result["temp"] == 22.5
    mock_get.assert_called_once()
```

**When NOT to mock:** Don't mock your own internal code. If `ServiceA` calls `ServiceB`, test them together in an integration test rather than mocking `ServiceB` away. Mocking internal code makes tests pass even when the real interaction is broken.

---

## Integration Tests

**An integration test verifies that multiple components work correctly together — your code, your database, your external services — as a connected system.**

```python
# tests/integration/test_user_service.py

import pytest
from myproject.database import get_session
from myproject.services.user_service import UserService

@pytest.fixture
def db_session():
    """Provide a real database session, rolled back after each test."""
    session = get_session()
    yield session
    session.rollback()
    session.close()

def test_create_and_retrieve_user(db_session):
    service = UserService(db_session)

    # Create a user in the real database
    created = service.create_user(name="Bob", email="bob@example.com")
    assert created.id is not None

    # Retrieve and verify
    fetched = service.get_by_id(created.id)
    assert fetched.name == "Bob"
    assert fetched.email == "bob@example.com"

def test_duplicate_email_raises_error(db_session):
    service = UserService(db_session)
    service.create_user(name="Alice", email="alice@example.com")

    with pytest.raises(ValueError, match="Email already exists"):
        service.create_user(name="Bob", email="alice@example.com")
```

The key difference from unit tests: **nothing is mocked**. The test hits a real database session, exercises real SQL queries, and verifies real behavior. The `rollback()` in the fixture keeps tests isolated.

### Marking Test Types

Use **pytest markers** to separate test types so you can run them independently:

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow-running",
    "integration: marks integration tests",
]
```

```python
@pytest.mark.integration
def test_create_and_retrieve_user(db_session):
    ...
```

```bash
pytest -m "not integration"    # Run only fast unit tests
pytest -m integration          # Run only integration tests
pytest                         # Run everything
```

---

## E2E Tests

**An end-to-end test simulates what a real user does — clicking buttons, filling forms, calling APIs — and verifies the entire system responds correctly.**

### API-Level E2E

For a FastAPI application, you can test the full request/response cycle:

```python
# tests/e2e/test_checkout_flow.py

import pytest
from httpx import AsyncClient
from myproject.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_full_checkout_flow(client):
    # Step 1: Create a user
    response = await client.post("/api/users", json={
        "name": "Charlie",
        "email": "charlie@example.com",
    })
    assert response.status_code == 201
    user_id = response.json()["id"]

    # Step 2: Add item to cart
    response = await client.post(f"/api/users/{user_id}/cart", json={
        "product_id": "SKU-001",
        "quantity": 2,
    })
    assert response.status_code == 200

    # Step 3: Checkout
    response = await client.post(f"/api/users/{user_id}/checkout")
    assert response.status_code == 200
    order = response.json()
    assert order["status"] == "confirmed"
    assert order["total"] > 0
```

This test exercises the **full stack**: routing, validation (Pydantic), business logic, database — everything.

### Browser-Level E2E (Playwright)

For web applications with a frontend:

```python
# tests/e2e/test_login.py

from playwright.sync_api import Page

def test_user_can_login_and_see_dashboard(page: Page):
    page.goto("http://localhost:3000/login")
    page.fill("#email", "alice@example.com")
    page.fill("#password", "secure-password")
    page.click("button[type=submit]")

    # Verify redirect to dashboard
    assert page.url == "http://localhost:3000/dashboard"
    assert page.text_content("h1") == "Welcome, Alice"
```

> **Key point:** E2E tests are expensive — slow to run, brittle to maintain, and hard to debug when they fail. Write them only for **critical user paths** (login, checkout, signup). Everything else should be covered by unit and integration tests.

---

## Test Coverage

**Coverage measures what percentage of your code is executed by your tests. It tells you what is tested — but not how well.**

### Setup

```toml
# pyproject.toml

[tool.coverage.run]
source = ["src"]          # Only measure your source code
branch = true             # Count branch coverage (if/else paths)

[tool.coverage.report]
fail_under = 80           # CI fails if coverage drops below 80%
show_missing = true       # Show which lines are not covered
exclude_lines = [
    "pragma: no cover",   # Explicit exclusion marker
    "if TYPE_CHECKING:",  # Type-only imports
]
```

### Running Coverage

```bash
$ pytest --cov=src --cov-report=term-missing

---------- coverage: platform linux, python 3.12 ----------
Name                          Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/myproject/calculator.py       8      0   100%
src/myproject/services.py        45      7    84%   23-25, 41-44
src/myproject/models.py          32      2    94%   58, 72
------------------------------------------------------------
TOTAL                            85      9    89%
```

The `Missing` column tells you **exactly which lines** have no test coverage. This is where to focus next.

### What Coverage Does NOT Tell You

```python
def divide(a: float, b: float) -> float:
    return a / b

def test_divide():
    assert divide(10, 2) == 5.0    # 100% coverage!
```

This test achieves 100% coverage on `divide()`. But it never tests `b=0`, which causes a `ZeroDivisionError`. **Coverage measures execution, not correctness.** It is a useful signal, not a guarantee.

> **Practical target:** 80% overall, 95%+ for core business logic. Don't chase 100% — the last 10% often requires testing error handlers and edge cases that provide diminishing returns.

---

## CI — Continuous Integration

**CI is the practice of automatically running linting, tests, and builds every time code is pushed or a pull request is opened.**

```
Developer pushes code
         |
         v
+-------------------+
|    CI Pipeline     |
|                    |
|  1. Install deps   |
|  2. Lint (ruff)    |       Runs automatically on
|  3. Type check     |  <--  every push and every PR
|  4. Run tests      |
|  5. Check coverage |
|  6. Security scan  |
|                    |
+-------------------+
         |
    Pass or Fail
         |
         v
  Status shown on PR:  ✅  or  ❌
```

Without CI, "did you run the tests?" depends on individual discipline. With CI, **every change is verified by the same automated pipeline**. No exceptions.

### CD — Continuous Deployment

CI proves the code is correct. **CD takes it further by automatically deploying that code.**

```
+--------+     +--------+     +------------------+
|  Push  | --> |   CI   | --> |       CD         |
|  code  |     | passes |     |                  |
+--------+     +--------+     |  Delivery:       |
                              |  Deploy to staging|
                              |  (human approves) |
                              |                  |
                              |  Deployment:     |
                              |  Auto-deploy to  |
                              |  production      |
                              +------------------+
```

- **Continuous Delivery:** Code is always ready to deploy. A human clicks the button.
- **Continuous Deployment:** Code deploys to production automatically after CI passes. No human step.

Most teams start with Delivery and move to Deployment as confidence in their test suite grows.

### GitHub Actions — A Practical Workflow

```yaml
# .github/workflows/ci.yml

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

      - name: Install dependencies
        run: pip install ruff mypy

      - name: Check formatting
        run: ruff format --check .

      - name: Run linter
        run: ruff check .

      - name: Run type checker
        run: mypy src/

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

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml --cov-fail-under=80

      - name: Upload coverage
        if: matrix.python-version == '3.12'
        uses: codecov/codecov-action@v4

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install tools
        run: pip install pip-audit bandit

      - name: Check dependency vulnerabilities
        run: pip-audit

      - name: Check code security
        run: bandit -r src/
```

**What this does:**

1. **lint** job: checks formatting, linting rules, and types. Fast — finishes in ~30 seconds.
2. **test** job: runs tests across 3 Python versions. If coverage drops below 80%, it **fails the pipeline**.
3. **security** job: scans dependencies for known vulnerabilities and source code for security anti-patterns.

All three jobs run **in parallel**. If any one fails, the PR gets a red X and cannot be merged (with branch protection enabled).

---

## Pre-commit Hooks

**A pre-commit hook is a script that runs automatically before every `git commit`. If any check fails, the commit is blocked.**

```
You run: git commit -m "feat: add user validation"
                |
                v
     +---------------------+
     |   Pre-commit Hooks   |
     |                      |
     |  ruff format ... ✅  |
     |  ruff check .... ✅  |
     |  mypy .......... ❌  |   <-- Type error found
     |  detect-secrets. ✅  |
     +---------------------+
                |
         Hook FAILED
                |
                v
     Commit is BLOCKED.
     Fix the type error, then commit again.
```

### Setup

```yaml
# .pre-commit-config.yaml

repos:
  # Formatting and linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]        # Auto-fix what it can
      - id: ruff-format      # Auto-format

  # General file hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: check-merge-conflict
      - id: detect-private-key      # Catch accidentally committed keys

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]

  # Secret detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

Install and activate:

```bash
pip install pre-commit
pre-commit install           # Activates hooks for this repo
pre-commit run --all-files   # Run all hooks on every file (first-time check)
```

After `pre-commit install`, hooks run automatically on every `git commit`. No discipline required — the tool enforces the rules.

---

## The Feedback Speed Ladder

Every check we've discussed exists somewhere on a speed spectrum. **The earlier you catch a problem, the cheaper it is to fix.**

```
+----------+---------------------------+-----------------+
|  Speed   |       Check               |   Feedback      |
+----------+---------------------------+-----------------+
| Fastest  | Editor (red underline)    | Milliseconds    |
|          | Pre-commit hook           | Seconds         |
|          | CI pipeline               | Minutes         |
| Slowest  | Code review / Production  | Hours to days   |
+----------+---------------------------+-----------------+
```

The ideal setup: **your editor warns you as you type, pre-commit blocks bad commits, CI catches anything that slipped through, and code review focuses on design — not style.**

```
You type code
     |
     v
Editor underlines errors (ruff, mypy plugin)    <-- instant
     |
     v
git commit → pre-commit hooks block problems    <-- seconds
     |
     v
git push → CI pipeline runs full test suite     <-- minutes
     |
     v
PR opened → reviewer focuses on design/logic    <-- hours
     |
     v
Merged to main → CD deploys automatically       <-- minutes
```

Each layer is a safety net for the one above it. **If pre-commit catches a linting error, CI never has to.** If CI catches a test failure, your reviewer never has to.

---

## Putting Part 2 Together

| Layer | Tool | What It Catches | Speed |
|---|---|---|---|
| Editor | ruff + mypy plugins | Typos, type errors, style | Instant |
| Pre-commit | ruff, mypy, detect-secrets | Lint errors, secrets, format | Seconds |
| CI - Lint | `ruff check`, `mypy` | Full project lint + types | ~30s |
| CI - Test | `pytest --cov` | Logic bugs, regressions | ~1-5 min |
| CI - Security | `pip-audit`, `bandit` | Vulnerabilities, unsafe code | ~30s |

---

## What's Next

You have clean, validated, tested, and automated code. But software is a team sport. How do you name branches? Write commit messages? Review each other's code without starting arguments?

In **Part 3**, we cover Git workflow, Conventional Commits, pull request best practices, code review etiquette, dependency management, and security fundamentals.

---

*Next in the series:*
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)
