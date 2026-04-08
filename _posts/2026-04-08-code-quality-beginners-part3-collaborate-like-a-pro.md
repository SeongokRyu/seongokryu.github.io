---
title: "Code Quality for Beginners Part 3: Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety"
date: 2026-04-08 11:00:00 +0900
categories: [Engineering]
tags: [code-quality, git, code-review, conventional-commits, pull-request, dependency-management, security, beginner]
---

Writing good code alone is one skill. Writing good code **in a team** is a different one entirely.

**Code Quality for Beginners**

*This is Part 3 of a 4-part series on Code Quality and Collaboration.*

- **Part 1:** [Write Clean Code — Formatting, Linting, and Data Validation](/posts/code-quality-beginners-part1-clean-code/)
- **Part 2:** [Test and Automate — From pytest to CI/CD Pipelines](/posts/code-quality-beginners-part2-test-and-automate/)
- **Part 3** (this post): Collaborate Like a Pro — Git Workflow, Code Review, and Dependency Safety
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)

---

In [Part 1](/posts/code-quality-beginners-part1-clean-code/) we made code clean. In [Part 2](/posts/code-quality-beginners-part2-test-and-automate/) we proved it works and automated the proof. Now we address the human side: how do you work on the same codebase with other people without stepping on each other's toes?

This post covers Git workflow, commit conventions, pull request practices, code review etiquette, dependency management, and security basics. These aren't technical flourishes — they are the **infrastructure of trust** between teammates.

---

## Git Workflow — How Code Flows From Your Machine to Production

### Trunk-Based Development (Recommended)

The simplest workflow that scales. The `main` branch is always deployable. Developers create **short-lived feature branches** that merge back quickly.

```
main ─────●──────●──────●──────●──────●──────●───── (always deployable)
           \    /        \    /        \    /
            feat          fix          feat
           (1-2 days)   (hours)      (1-2 days)
```

**Rules:**
- `main` is **always deployable** — never push broken code directly
- Feature branches live **1-2 days max** — long-lived branches cause merge hell
- Every merge goes through a **pull request** with at least one review
- Use **feature flags** to hide incomplete features, not long branches

### Why Not GitFlow?

GitFlow (with `develop`, `release`, `hotfix` branches) was designed for packaged software with long release cycles. For web services with continuous deployment, it adds unnecessary complexity. Most modern teams have moved to trunk-based or GitHub Flow.

```
+---------------------+---------------------------------------------+
|                     | Trunk-Based           | GitFlow              |
+---------------------+---------------------------------------------+
| Branch count        | main + short features | main, develop,       |
|                     |                       | feature/*, release/* |
| Branch lifetime     | Hours to 2 days       | Weeks to months      |
| Merge frequency     | Daily                 | Per release cycle    |
| Merge conflicts     | Rare (small diffs)    | Frequent (big diffs) |
| Best for            | Web services, CI/CD   | Packaged software    |
+---------------------+---------------------------------------------+
```

---

## Branch Naming

A good branch name tells you **what type of change it is, what issue it relates to, and what it does** — at a glance.

### Pattern

```
<type>/<ticket-id>-<short-description>
```

### Examples

```
feat/PROJ-123-user-authentication       # New feature
fix/PROJ-456-login-redirect-loop         # Bug fix
refactor/PROJ-789-extract-payment-module # Refactoring
docs/PROJ-101-api-endpoint-documentation # Documentation
chore/PROJ-102-upgrade-dependencies      # Maintenance
```

### Rules

- **Lowercase**, words separated by hyphens
- **Include the ticket/issue number** for traceability — you should be able to trace any branch back to a requirement
- **Keep the description short** — 3 to 5 words
- **Use the type prefix** — it matches your commit convention (covered next)

### Bad Branch Names

```
my-branch              # What does it do?
fix                    # Fix what?
john/stuff             # Not helpful for anyone, including John
feature-new-thing-v2   # No ticket, vague description
```

---

## Conventional Commits

**Conventional Commits is a standardized format for commit messages that makes the project history machine-readable and human-scannable.**

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### The Type Table

```
+-----------+--------------------------------------------+------------------+
| Type      | When to Use                                | Example          |
+-----------+--------------------------------------------+------------------+
| feat      | New feature for the user                   | feat: add search |
| fix       | Bug fix                                    | fix: null crash  |
| docs      | Documentation only                         | docs: update API |
| style     | Formatting, no logic change                | style: fix indent|
| refactor  | Code restructure, no behavior change       | refactor: extract|
| perf      | Performance improvement                    | perf: cache query|
| test      | Adding or fixing tests                     | test: add edge   |
| build     | Build system or dependencies               | build: upgrade uv|
| ci        | CI/CD configuration                        | ci: add matrix   |
| chore     | Other maintenance                          | chore: clean logs|
+-----------+--------------------------------------------+------------------+
```

### Good vs Bad Commit Messages

```
# Bad — vague, no type, no context
fix bug
update code
WIP
asdfasdf

# Good — structured, specific, traceable
feat(auth): add JWT refresh token rotation
fix(api): prevent null pointer when user has no profile
docs(readme): add local development setup instructions
refactor(payment): extract tax calculation into separate module

# Good — with body and footer for complex changes
feat(api): add paginated response for user endpoint

The /api/users endpoint now returns paginated results by default.
Page size defaults to 20, configurable via ?page_size parameter.
Existing clients must update to handle the wrapper object.

BREAKING CHANGE: response format changed from array to paginated object.
Closes #234
```

**Why bother?** Three reasons:
1. `git log --oneline` becomes a **readable changelog**
2. Tools like `commitlint` can **enforce the format in CI**
3. Tools like `release-please` can **auto-generate changelogs and version bumps** from your commit history

---

## Pull Request Best Practices

A pull request is not just a code delivery mechanism. It is a **communication artifact** — a record of what changed, why, and how to verify it.

### Size Matters

```
+-------------------+---------------+----------------------------+
| PR Size (lines)   | Review Time   | Quality                    |
+-------------------+---------------+----------------------------+
| < 200             | ~15 min       | Thorough, focused          |
| 200 - 400         | ~30 min       | Good balance               |
| 400 - 500         | ~1 hour       | Starting to skim           |
| > 500             | ???           | "LGTM" with no real review |
+-------------------+---------------+----------------------------+
```

**Ideal: 200-400 lines.** If your PR is over 500 lines, split it. Mechanical changes (renaming, formatting) should be in a **separate PR** from logic changes — mixing them makes review impossible.

### PR Description Template

Every PR should answer three questions: **What** changed? **Why** did it change? **How** can a reviewer verify it?

```markdown
## What
Add JWT refresh token rotation to the auth module.

## Why
Current tokens have a 24-hour lifetime with no rotation. If a token
is stolen, the attacker has access for up to 24 hours. Refresh token
rotation limits the window to a single use.

Closes #234

## How to Test
1. Login with test credentials
2. Wait for token to expire (set TTL to 10s for testing)
3. Verify a new access token is issued
4. Verify the old refresh token is invalidated

## Checklist
- [x] Tests added
- [x] Documentation updated
- [ ] Migration required (none)
- [ ] Breaking change (no)
```

### Self-Review First

Before requesting a review from anyone else:

1. **Read your own diff** on GitHub — you will catch things you missed in the editor
2. **Check for debug artifacts** — leftover `print()`, `console.log()`, commented-out code
3. **Verify CI passes** — don't waste a reviewer's time on code that doesn't build
4. **Use Draft PRs** for work-in-progress — signal "not ready for review yet"

---

## Code Review Etiquette

Code review is the highest-leverage activity on a team. Done well, it catches bugs, spreads knowledge, and aligns standards. Done poorly, it creates resentment and delays.

### Reviewer Feedback Prefixes

Use prefixes to signal the **weight** of each comment:

```
nit: rename `data` to `user_data` for clarity
  → Minor suggestion. Author decides whether to adopt.

suggestion: consider using a set here for O(1) lookup
  → Improvement idea. Worth discussing, not blocking.

question: why does this skip validation for admin users?
  → Seeking understanding. Not necessarily a problem.

blocking: this SQL query is vulnerable to injection
  → Must be fixed before merge. Non-negotiable.
```

Without prefixes, every comment feels like a blocker. With them, authors know **what requires action and what is optional**.

### The Reviewer Checklist

When reviewing a pull request, check these in order:

1. **Correctness** — Does the logic do what it claims? Edge cases?
2. **Design** — Is responsibility properly separated? Are abstractions appropriate?
3. **Tests** — Are the important paths tested? Could the tests pass with broken code?
4. **Security** — Input validation? Authentication checks? SQL injection?
5. **Performance** — N+1 queries? Unnecessary loops? Missing indexes?
6. **Readability** — Can you understand the code in 5 minutes? Are names clear?

**What NOT to review:** formatting, import order, quote style. These should be handled by **tools** (ruff, pre-commit), not humans. If you're commenting on whitespace, your tooling is broken.

### Review Response Time

- **Aim for 24 hours** — longer blocks the author and kills momentum
- If you can't review in 24 hours, **say so** and suggest another reviewer
- If a discussion goes back and forth more than 3 rounds, **switch to a synchronous conversation** (call, pair programming) — text is too slow for complex design debates

---

## CODEOWNERS

**CODEOWNERS is a file that automatically assigns reviewers based on which files a PR modifies.**

```
# .github/CODEOWNERS

# Default — team leads review everything
*                           @org/team-leads

# Backend team owns the API and data layers
/src/api/                   @org/backend-team
/src/models/                @org/backend-team @org/data-team
/src/services/              @org/backend-team

# Frontend team owns the UI
/frontend/                  @org/frontend-team

# DevOps owns infrastructure and CI
/infra/                     @org/devops-team
/.github/workflows/         @org/devops-team

# Sensitive files need lead approval
pyproject.toml              @org/team-leads
.github/CODEOWNERS          @org/team-leads
```

When a PR modifies `/src/api/routes/users.py`, GitHub automatically requests a review from `@org/backend-team`. Combined with branch protection ("Require review from Code Owners"), **no change reaches main without the right eyes on it**.

---

## Lock Files — Why Your Teammate Gets Different Bugs

**A lock file records the exact version of every dependency installed in your project. It ensures everyone — every developer, every CI runner — uses identical packages.**

### The Problem Without Lock Files

```
pyproject.toml says:    "httpx>=0.27"

Developer A installs:   httpx 0.27.0    (January)
Developer B installs:   httpx 0.28.1    (March, new release)
CI server installs:     httpx 0.28.0    (somewhere in between)
```

Same spec, three different versions. A bug that only exists in 0.28.0 appears in CI but not on Developer A's machine. "Works on my machine" begins.

### The Solution

```
pyproject.toml:   "httpx>=0.27"           # Intent: "I need at least 0.27"
uv.lock:          httpx==0.28.1           # Reality: "Everyone uses exactly 0.28.1"
```

The lock file is generated by your package manager (`uv lock`, `poetry lock`) and must be **committed to Git**. When anyone runs `uv sync` or `poetry install`, they get **exactly** the versions in the lock file.

```
pyproject.toml    =    Recipe (ingredients and rough quantities)
Lock file         =    Grocery receipt (exact brand, exact size, exact store)
```

**Rules:**
- **Always commit the lock file** to version control
- In CI, install with `--frozen` flag to fail if the lock file is out of date
- After adding or updating a dependency, run `uv lock` and commit both files

---

## Virtual Environments

**A virtual environment is an isolated Python installation for a single project. It prevents one project's dependencies from interfering with another's.**

Without virtual environments:

```
Project A needs Django 4.2
Project B needs Django 5.0
Both install to the same global Python
    → One of them breaks
```

With virtual environments:

```
Project A:  .venv/ has Django 4.2     # Isolated
Project B:  .venv/ has Django 5.0     # Isolated
System Python: untouched              # Clean
```

### Commands

```bash
# Modern approach (uv — fast, recommended)
uv venv              # Create .venv/
uv sync              # Install from lock file into .venv/

# Traditional approach
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
pip install -e ".[dev]"
```

---

## Security Basics

Security is not a feature you add later. It is a set of practices you follow **from the beginning**.

### Never Commit Secrets

```
# .gitignore — these must NEVER enter version control

# Environment and secrets
.env
.env.*
!.env.example        # Template is OK (no real values)
*.pem
*.key
credentials.json
service-account.json

# Python
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.mypy_cache/
.pytest_cache/
.coverage
htmlcov/

# IDE
.idea/
.vscode/settings.json
```

Provide a `.env.example` with dummy values so new developers know which variables to set:

```bash
# .env.example — commit this, never .env
APP_DATABASE_URL=postgresql://localhost/mydb
APP_SECRET_KEY=change-me-in-your-local-env
APP_DEBUG=false
```

### Dependency Vulnerability Scanning

Your dependencies have bugs too — including **security vulnerabilities**. Scan them automatically:

```bash
# pip-audit — checks PyPI packages against known vulnerability databases
pip install pip-audit
pip-audit
# Found 2 known vulnerabilities in 1 package
#   requests  2.28.0  CVE-2023-32681  MODERATE

# detect-secrets — catches accidentally committed passwords, API keys, tokens
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

Both should be in your **CI pipeline** (Part 2) and your **pre-commit hooks** (Part 2). If a vulnerable dependency is found, the pipeline fails and the PR cannot merge until the dependency is updated.

### Code Security Scanning

```bash
# bandit — finds common security issues in Python code
pip install bandit
bandit -r src/

# Example output:
# src/db.py:15  B608  Possible SQL injection via string concatenation
# src/auth.py:8 S105  Possible hardcoded password: 'admin123'
```

**The ruff `S` rules** (from Part 1) also catch many of these — another reason to enable them.

### Security Rules of Thumb

- **Validate all external input** through Pydantic models (Part 1)
- **Never use string formatting in SQL** — use parameterized queries
- **Never log secrets** — mask sensitive fields in structured logging
- **Set explicit CORS origins** — never use `*` in production
- **Pin your CI action versions** — `uses: actions/checkout@v4`, not `@main`

---

## Putting Part 3 Together

```
You write code on a feature branch
     |
     v
Branch naming:   feat/PROJ-123-user-auth
     |
     v
Commit messages: feat(auth): add login endpoint
     |
     v
Pre-commit hooks verify code quality (Part 2)
     |
     v
Push → CI verifies everything (Part 2)
     |
     v
Open PR (< 400 lines, clear description)
     |
     v
CODEOWNERS assigns the right reviewers
     |
     v
Reviewers use nit/suggestion/blocking prefixes
     |
     v
Merge to main → CD deploys automatically
```

### Quick Reference

| Practice | Rule |
|---|---|
| Branch naming | `<type>/<ticket>-<description>` |
| Commit format | `<type>(<scope>): <description>` |
| PR size | 200-400 lines, max 500 |
| Review time | < 24 hours |
| Lock file | Always commit, install with `--frozen` |
| Secrets | Never in Git, use `.env` + `.env.example` |
| Scanning | `pip-audit` + `detect-secrets` + `bandit` in CI |

---

## What's Next

You now have clean code (Part 1), automated tests (Part 2), and solid collaboration practices (Part 3). But all of these rules live in your head and your team wiki. What if you could encode them so that **an AI agent follows the same standards automatically?**

In **Part 4**, we cover CLAUDE.md, harness engineering, and how to wire everything from Parts 1-3 into a system that both humans and AI agents obey.

---

*Next in the series:*
- **Part 4:** [Encode Your Standards — CLAUDE.md, Harness Engineering, and the Full Setup](/posts/code-quality-beginners-part4-encode-your-standards/)
