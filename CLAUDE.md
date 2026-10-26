# CLAUDE.md: working notes for sessions on this repo

Hobby hard sci-fi space combat sim. Current job: milestone M0, the Python prototype in `proto/`.

## Documents (claude.ai artifacts: read with the docs tools, never WebFetch)

- Design: https://claude.ai/artifact/QnCWkYPm5U1tEm3668pNw1 (models in sections 3 to 9; resolved decisions in 15, which override earlier text; doctrines in 16)
- Implementation plan: https://claude.ai/artifact/WK5UK41zsJZPEGAzQd5D7K (M0 gate in section 10, tests in section 8)

## Environment facts (verified 2026-10-25)

- Python 3.13 with numpy, scipy and matplotlib preinstalled. Use `MPLBACKEND=Agg` for plots.
- pip works only through the session proxy: plain `pip install` fails because pypi.org is in NO_PROXY and DNS does not resolve directly. Use `pip install --break-system-packages --proxy "$HTTPS_PROXY" pytest nbformat nbconvert ipykernel` (verified 2026-10-26 once the user allowed package managers: pytest 9.1.1 installed and notebook execution through nbconvert works). If the proxy answers 403 for pypi.org, the policy has not allowed package managers in that environment: do not retry, and use the fallback below.
- Fallback without pytest: write tests as plain functions named `test_*` in `proto/tests/` and run them with `proto/run_tests.py` (a small stdlib runner that imports modules, calls each `test_*`, reports failures, exits nonzero on failure). Build the notebook as .ipynb JSON with the stdlib `json` module and check it by executing its code cells in order with `exec` in one namespace. Write tests so they work under both runners (plain functions with `assert`).
- Packages do not persist between sessions: reinstall at the start of every session.
- `cargo` exists, but the crates registry is probably blocked as well. Not needed for M0.
- No outside network except what the policy allows (GitHub works). Do not search for workarounds.
- The workspace does not persist between sessions. Always re-clone, and push before ending.

## Session routine

1. Load deferred tools in one call: `ToolSearch select:mcp__Claude_Docs__read,mcp__claude-code-remote__update_trigger,mcp__claude-code-remote__delete_trigger,mcp__claude-code-remote__list_triggers`.
2. Read PROGRESS.md, then only the doc sections you need for the next piece.
3. Work test-first, small pieces. Commit and push after each piece (`git fetch origin main`, rebase if needed, `git push origin HEAD`). A stop hook complains about unpushed commits, so end every session with nothing unpushed.
4. Commit messages end with the trailers given by the session's attribution reminder (Co-Authored-By and Claude-Session lines).
5. Keep PROGRESS.md current: done, in progress, decisions, open issues, next steps, one line per session with its start time.
6. Fallback if GitHub is unreachable: save the same state with the Projects tool (`project_write`, path `claude/progress.md`) if available, otherwise in a Docs doc titled "Prototype progress log".
7. No subagents unless the user asks. Do not ask the user questions in scheduled runs; choose a default and record it.

## Conventions

- SI units, float64, seeded RNG only (`numpy.random.default_rng(seed)`), no wall clock in models.
- Code in `proto/spaceops/` modules, tests in `proto/tests/`, notebook in `proto/notebooks/`, exported Rust test vectors as JSON in `proto/vectors/`, plots in `proto/plots/`.
- Respect the design decisions: sensor light delay stays; no command lag; no nuclear weapons; no debris (missed munitions are removed, friendly-fire check at launch); gravity and occultation are first-class; limited war; strict identification rules.
- Validate the salvo model against the published leak figures: 6 missiles, hit probability 0.34, 3 / 4 / 5 shots per missile leak with probability about 58 / 24 / 7 percent (tolerate Monte Carlo noise).
- Do not edit README.md beyond status lines. Do not rewrite history on main.

## Scheduling

One scheduled task (id trig_019NBrSFcYYffYPnzVGWdrHc) drives the sessions. At the start of a session, re-arm it for start time plus 5 hours with `update_trigger` and `run_once_at`. If that errors, create a new one-shot with `create_trigger` using the same prompt. When M0's gate is met, disable or delete it.
