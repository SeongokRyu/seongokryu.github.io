---
title: "Code Quality for Beginners Part 1: Write Clean Code Before You Push — Formatting, Linting, and Data Validation"
date: 2026-04-08 09:00:00 +0900
categories: [Engineering]
tags: [code-quality, formatting, linting, ruff, mypy, pydantic, python, beginner]
---

Your code works. But can your teammate read it at 2 AM during an incident?

**Code Quality for Beginners**

*This is Part 1 of a 4-part series on Code Quality and Collaboration.*

- **Part 1** (this post): Write Clean Code — Formatting, Linting, and Data Validation
- **Part 2:** [Test and Automate — From pytest to CI/CD Pipelines](/posts/code-quality-beginners-part2-test-and-automate/)
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)

---

Writing code that works is the first step. Writing code that is **predictable, reviewable, and deployable** is the real job. This series walks you through the tools, conventions, and practices that separate "it works on my machine" from production-ready team code.

In Part 1, we start at the foundation: making your code look consistent, catching bugs before you run anything, and validating data before it poisons your logic.

---

## What is Formatting?

**Formatting is the automatic arrangement of your code's appearance — whitespace, indentation, quotes, line breaks — without changing any logic.**

Consider this snippet:

```python
# Before formatting — every developer writes differently
def calculate_total( items,tax_rate ):
    subtotal=sum([  item['price']*item['qty'] for item in items])
    total = subtotal+(subtotal * tax_rate)
    return  total
```

Now the same code after a formatter runs:

```python
# After formatting — one consistent style, automatically
def calculate_total(items, tax_rate):
    subtotal = sum([item["price"] * item["qty"] for item in items])
    total = subtotal + (subtotal * tax_rate)
    return total
```

No logic changed. No bugs fixed. But the second version is **instantly scannable**. Every file in the project looks like it was written by the same person.

**Why does this matter for teams?** Without a formatter, pull request diffs fill up with whitespace changes, quote style flips, and indentation wars. Reviewers waste time on cosmetics instead of logic. A formatter eliminates this entire category of noise.

> **Key point:** Formatting is not a matter of personal taste — it is a tool's job. Pick a tool, configure it once, and never think about it again.

---

## What is Linting?

**Linting is the static analysis of your code to catch potential bugs, unused variables, bad patterns, and style violations — all without running the code.**

```python
import os           # Linter warning: 'os' imported but unused
import json

def process_data(data):
    result = data * 2
    return data     # Linter warning: 'result' is assigned but never used

def risky_function():
    password = "admin123"   # Linter warning: possible hardcoded password
```

A linter reads your code and flags problems that are easy to miss in a manual review but obvious to a machine.

### Formatting vs Linting

These two concepts are often confused. Here is the key difference:

```
+------------------+-------------------------------+-------------------------------+
|                  |         Formatting            |           Linting             |
+------------------+-------------------------------+-------------------------------+
| What it checks   | Code appearance               | Code correctness              |
| Examples         | Spacing, quotes, line length  | Unused imports, type errors   |
| Changes logic?   | Never                         | Sometimes (auto-fix)          |
| Catches bugs?    | No                            | Yes                           |
| Human effort     | Zero — fully automated        | Minimal — review warnings     |
+------------------+-------------------------------+-------------------------------+
```

**Formatting** answers: "Does this code look right?"
**Linting** answers: "Does this code smell wrong?"

You need both. Formatting handles the surface. Linting catches what hides underneath.

---

## Ruff — One Tool to Replace Them All

**Ruff is a Python linter and formatter written in Rust. It replaces flake8 (linting), black (formatting), and isort (import sorting) in a single tool that runs 10–100x faster.**

Before Ruff, a typical Python project needed three separate tools:

```
black .          # formatting
isort .          # import sorting
flake8 .         # linting
```

Now, one tool does it all:

```bash
ruff format .         # Format all files (replaces black + isort)
ruff check .          # Lint all files (replaces flake8)
ruff check . --fix    # Lint and auto-fix what it can
```

### Practical Setup

All configuration lives in `pyproject.toml` — the single source of truth for Python project settings:

```toml
# pyproject.toml

[tool.ruff]
target-version = "py312"    # Target Python version
line-length = 88            # Max line length (black's default)

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors (basic style)
    "W",    # pycodestyle warnings
    "F",    # pyflakes (unused imports, undefined names)
    "I",    # isort (import ordering)
    "N",    # pep8-naming (naming conventions)
    "UP",   # pyupgrade (modernize syntax)
    "B",    # flake8-bugbear (common gotchas)
    "SIM",  # flake8-simplify (simplifiable code)
    "S",    # flake8-bandit (security issues)
]
ignore = [
    "E501",   # Line too long — the formatter handles this
]

[tool.ruff.lint.isort]
known-first-party = ["myproject"]   # Your package name

[tool.ruff.format]
quote-style = "double"              # Consistent double quotes
docstring-code-format = true        # Format code inside docstrings
```

Each rule code maps to a specific check. For example, `"F401"` catches unused imports, `"S105"` catches hardcoded passwords. Ruff includes **800+ rules** — the `select` list above is a practical starting point.

### What Ruff Catches in Practice

```python
# ruff check output examples:

src/utils.py:3:1: F401 `os` imported but unused
src/auth.py:15:5: S105 Possible hardcoded password assigned to "token"
src/models.py:42:9: B006 Do not use mutable data structures for argument defaults
src/api.py:8:1: I001 Import block is unsorted or unformatted
```

Each warning tells you **exactly where the problem is** and **what rule triggered it**. Most have auto-fixes available via `--fix`.

---

## Type Checking with mypy

**mypy is a static type checker for Python. It reads your type annotations and catches type errors before you run the code.**

Python is dynamically typed — this code runs fine until it explodes at runtime:

```python
def greet(name: str) -> str:
    return "Hello, " + name

# Python won't complain until this line actually executes
result = greet(42)
# TypeError: can only concatenate str (not "int") to str
```

With mypy, you catch this **before** any code runs:

```bash
$ mypy src/
src/main.py:4: error: Argument 1 to "greet" has incompatible type "int"; expected "str"
Found 1 error in 1 file
```

### Practical Setup

```toml
# pyproject.toml

[tool.mypy]
python_version = "3.12"
strict = true                    # Enable all strict checks
warn_return_any = true           # Warn when function returns Any
warn_unused_configs = true       # Warn about unused mypy config
plugins = ["pydantic.mypy"]      # Pydantic integration

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false    # Relax rules for test files
```

`strict = true` is the most important line. It enables checks like:

- **No untyped function definitions** — every function must have type annotations
- **No implicit Optional** — `str` and `str | None` are different types
- **No Any generics** — `list[str]` instead of plain `list`

### Type Annotation Basics

```python
from __future__ import annotations    # Enables modern syntax in all Python 3.7+

# Basic types
name: str = "Alice"
count: int = 42
ratio: float = 3.14
active: bool = True

# Collections
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95, "Bob": 87}

# Optional values (may be None)
middle_name: str | None = None

# Function signatures
def fetch_user(user_id: int, *, include_deleted: bool = False) -> User | None:
    ...
```

> **Key point:** Type annotations are documentation that the machine can verify. They tell your teammates what a function expects, and mypy proves they're not lying.

---

## Why Validate Data?

Every bug story starts the same way: **someone trusted external data.**

```python
# Somewhere in your API handler...
def create_order(data: dict):
    price = data["price"]               # KeyError if missing
    quantity = data["quantity"]
    total = price * quantity             # TypeError if price is a string
    discount = data.get("discount", 0)
    final = total - discount             # Negative total if discount > total
    return final
```

This function has no idea what `data` actually contains. If someone sends `{"price": "free", "quantity": -5}`, you get garbage output — or a crash. **Raw dictionaries are landmines.**

The fix: validate and parse external data **at the boundary**, before it enters your business logic.

---

## Pydantic Basics

**Pydantic is a Python library that validates data structure and types at runtime. You define what your data should look like, and Pydantic enforces it automatically.**

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Valid data — even coerces "25" (string) to 25 (int)
user = User(name="Alice", age="25", email="alice@example.com")
print(user.age)       # 25 (int, not str)
print(type(user.age)) # <class 'int'>

# Invalid data — Pydantic catches it immediately
user = User(name="Alice", age="twenty-five", email="alice@example.com")
# ValidationError: 1 validation error for User
# age
#   Input should be a valid integer, unable to parse string as an integer
```

Compare this to the raw dict approach:

```
+-----------------------------------+-----------------------------------+
|          Raw Dict                  |         Pydantic Model            |
+-----------------------------------+-----------------------------------+
| data["age"] — might be missing    | user.age — guaranteed to exist    |
| Could be str, int, None, anything | Guaranteed int, validated         |
| Errors surface deep in logic      | Errors surface at the boundary    |
| No autocomplete in IDE            | Full autocomplete + type hints    |
+-----------------------------------+-----------------------------------+
```

### Field Constraints and Custom Validators

Pydantic goes beyond basic type checking. You can define **constraints** and **custom validation logic**:

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., gt=0, description="Price in USD, must be positive")
    sku: str
    quantity: int = Field(default=0, ge=0)

    @field_validator("sku")
    @classmethod
    def validate_sku_format(cls, v: str) -> str:
        """SKU must follow the pattern: 3 letters + dash + 4 digits."""
        if not v[:3].isalpha() or v[3] != "-" or not v[4:].isdigit():
            raise ValueError("SKU must match format: ABC-1234")
        return v.upper()

# Valid
product = Product(name="Widget", price=9.99, sku="abc-1234", quantity=10)
print(product.sku)  # "ABC-1234" (uppercased by validator)

# Invalid — price is negative
product = Product(name="Widget", price=-5, sku="abc-1234")
# ValidationError: price - Input should be greater than 0
```

For validation that spans **multiple fields**, use `model_validator`:

```python
from datetime import datetime

class EventBooking(BaseModel):
    event_name: str
    start_time: datetime
    end_time: datetime
    max_attendees: int = Field(ge=1)

    @model_validator(mode="after")
    def check_time_range(self) -> "EventBooking":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        return self
```

### Config Management with pydantic-settings

Most applications need configuration from environment variables or `.env` files. The naive approach:

```python
import os

# Fragile — no validation, no defaults, no type safety
db_url = os.getenv("DATABASE_URL")          # Could be None
debug = os.getenv("DEBUG")                  # Returns "true" (string), not True (bool)
max_conn = int(os.getenv("MAX_CONN", "5"))  # Manual parsing everywhere
```

With **pydantic-settings**, your config becomes a validated, typed object:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file
        env_prefix="APP_",         # Only read APP_* variables
        case_sensitive=False,      # APP_DEBUG or app_debug both work
    )

    debug: bool = False
    database_url: str              # Required — app won't start without it
    max_connections: int = 5
    allowed_origins: list[str] = ["http://localhost:3000"]
    secret_key: str                # Required — no default

# .env file:
# APP_DATABASE_URL=postgresql://localhost/mydb
# APP_SECRET_KEY=super-secret-key-here
# APP_DEBUG=true

settings = Settings()              # Loads, validates, and parses automatically
print(settings.debug)              # True (bool, not "true" string)
print(settings.database_url)       # "postgresql://localhost/mydb"
```

If `APP_DATABASE_URL` or `APP_SECRET_KEY` is missing, the application **refuses to start** with a clear validation error — not a mysterious `None` that crashes 10 minutes later in a database call.

---

## Putting Part 1 Together

Here is what your development flow looks like after adopting these tools:

```
You write code
     |
     v
ruff format .          Auto-fix appearance (spacing, quotes, imports)
     |
     v
ruff check . --fix     Catch and fix bugs, unused code, bad patterns
     |
     v
mypy src/              Verify types are consistent
     |
     v
All external data flows through Pydantic models
     |
     v
Code is clean, typed, validated — ready for testing (Part 2)
```

### Quick Reference

| Tool | What It Does | Command |
|---|---|---|
| `ruff format` | Auto-format code appearance | `ruff format .` |
| `ruff check` | Lint for bugs and bad patterns | `ruff check . --fix` |
| `mypy` | Static type checking | `mypy src/` |
| `pydantic` | Runtime data validation | Define `BaseModel` classes |
| `pydantic-settings` | Typed config from env vars | Define `BaseSettings` class |

---

## What's Next

Your code is now consistent, typed, and validates its inputs. But how do you prove it works? And how do you prevent regressions when you change something?

In **Part 2**, we cover the testing pyramid, writing effective tests with pytest, measuring coverage, and automating everything with CI/CD pipelines and pre-commit hooks.

---

*Next in the series:*
- **Part 2:** [Test and Automate — From pytest to CI/CD Pipelines](/posts/code-quality-beginners-part2-test-and-automate/)
- **Part 3:** [Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety](/posts/code-quality-beginners-part3-collaborate-like-a-pro/)
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)
