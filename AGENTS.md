# Aleph - Agent Specification

## General rules

Keep things simple. Solve the problem that was asked for, in the smallest way
that fits the code around it.

- Do not build for requirements that do not exist yet. No extra abstraction
  layers, configuration options or generic helpers added on the assumption that
  something will need them later.
- Prefer the boring solution. A plain function beats a class, a decorator or a
  metaclass unless there is a concrete reason for the latter.
- Change as little as possible. Do not rename, reformat or restructure code that
  is unrelated to the task.
- Reuse what is already in the repository before reaching for a new dependency,
  pattern or file.
- If something cannot be done simply, say so and explain the trade-off rather
  than hiding the complexity in clever code.
- Leave working code behind, not scaffolding. Finish one thing properly instead
  of half-doing several.

## Writing

This applies to commit messages, documentation, comments, pull request
descriptions and anything else written for a human to read.

- Do not use em dashes or en dashes. Use a comma, a full stop, or rewrite the
  sentence.
- Write short, direct sentences in plain language, and say each thing once.
- Cut filler words such as "simply", "just", "seamlessly", "robust", "powerful"
  and "leverage".
- Drop the summary that repeats what was already said, and the closing sentence
  that adds nothing.
- Avoid the patterns that read as machine written, such as "it is not just X, it
  is Y", lists of three adjectives, and headings or bold text sprinkled through
  a few short paragraphs.
- Use British spelling, as the rest of the repository does.
- Do not overstate. If something was not tested or verified, write that instead
  of claiming it works.

## Git conventions

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org)
specification and should be written in the imperative mood, as recommended by Git.

## Comments

Prefer self-explanatory code over comments. Before writing a comment, try to make
it unnecessary. Use descriptive names for variables, functions, parameters and
types, extract a named function or constant, or restructure the code so the
intent is obvious.

Write a comment only when it explains something the code cannot, such as:

- Why a non-obvious approach was chosen, or why the obvious one does not work.
- Constraints, trade-offs, invariants and assumptions that are not visible locally.
- References to external context, such as a specification, ticket or upstream bug.
- Warnings about surprising behaviour or subtle edge cases.

Do not write comments that:

- Restate what the code already says.
- Narrate a change or its history. That belongs in the commit message.
- Assume the machine the code runs on, such as installed CLI tools, absolute
  paths or a particular operating system.

Docstrings on public modules, classes and functions are required and are not
covered by the rules above, but they should explain purpose, behaviour and
caveats rather than repeat the signature. Keep them up to date when the
surrounding code changes.

## Structure

Use the language's own structure to separate code, not comments. A file that
needs a banner to explain where one part ends and the next begins is a file that
wants to be split up.

Do not mark sections of a file with a comment, for example:

```
# ---------------------- Stuff here ----------------------
```

Split the code into smaller functions, modules or files instead, and let the
names carry the meaning the banner was trying to give.

## Command line tools

The environment provides faster replacements for the standard tools. Use them.

| Instead of   | Use   | Notes                                                       |
| ------------ | ----- | ----------------------------------------------------------- |
| `grep -r`    | `rg`  | Filter by language with `rg -t py`                           |
| `find -name` | `fd`  | `fd -e py`, `fd crop src`                                    |
| `sed -i`     | `sd`  | Literal by default, so no regex escaping for a plain rename  |
| `jq`         | `jaq` | For `faces/manifest.json` and other generated JSON           |

Search with `rg` before you search with anything else. Pair it with `sd` for a
mechanical rename, such as `rg -l old_name | xargs sd old_name new_name`, and
read the diff afterwards. Do not use a bulk replace for anything that needs
judgement.

Nothing activates the virtualenv for the non-interactive shell you run commands
in, so prefix every command that needs the project environment with `uv run`.

## Python

Python 3.14, pinned in `.python-version`. uv manages the interpreter, the
`.venv` and the lock file.

- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Add
  them with `uv add`, `uv add --dev` or `uv add --group lint`. Do not edit
  `uv.lock` by hand, and commit it together with the change that caused it.
- `uv run ruff format` formats, `uv run ruff check` lints and `uv run pytest`
  runs the tests. Leave formatting to ruff and do not hand-format code to look
  different from what it produces.
- The enabled lint rules live in `ruff.toml`. Fix the code rather than silencing
  the rule. If a rule is genuinely wrong for a whole file, add a
  `per-file-ignores` entry with a comment saying why, instead of scattering
  `# noqa`.
- Annotate parameters and return types where the type is stable. Objects from
  dependencies that have no usable stubs, such as Pillow images, Ultralytics
  results and the argparse namespace, stay unannotated rather than being given
  an invented type.
- Raise the narrowest exception that fits, with a message that tells the user
  what to do next. Chain the cause with `raise ... from err`, or `from None`
  when the original adds nothing. Do not catch bare `Exception`, except in the
  loop over input photos, where one bad file must not end the run.
- `torch`, `ultralytics`, `rawpy` and `pillow_heif` are imported inside the
  function that needs them. This keeps `--help` instant and turns a missing
  optional decoder into a readable error. Do not lift them to the top of the
  file.
- The command line interface is one argparse parser in `build_parser()`, with
  flags in named groups. Progress goes to stdout and problems to stderr.
  `main()` returns an exit code rather than calling `sys.exit` part way through.
- Names are snake_case, except for image dimensions such as `W` and `H`, which
  follow the geometry they come from.
- Tests live in `tests/`, in files named `*_test.py`, with shared fixtures in
  `conftest.py`. Use plain `assert`, with `pytest.approx` for floats and
  `tmp_path` for files. Cover the geometry, parsing and merging logic. Do not
  write a test that needs model weights, a GPU or a network.
- Model weights, source photos and the `faces/` output are gitignored because of
  their size. Never commit them.
