# python-project-template

My template for Python3 projects.

## Features

- Proper Python type checking
- Proper Python dependency locking
- Automatically managed environments
- Visual of outdated dependencies
- Very customizable linting
- Linting of code before push to main/master and staging branches

## Setup project

Prerequisites:

- [uv](https://docs.astral.sh/uv/)

Start off by giving the project a name. The name should be set in these places:

- [`package`](package): Change folder name.
- [`pyproject.toml`](pyproject.toml#L2): Change to project name.
- [`Dockerfile`](Dockerfile#L999): Change Dockerfile entrypoint.

First you have to make sure your environment is activate.

```shell
. .venv/bin/actiavte
```

That's it. Now sync missing dependencies for the environment.

```shell
$ uv sync
Installing dependencies from lock file
```

Now you are ready to start developing!

**NOTE:** If you have issues like missing dependencies, you might have to select the right interpreter.
You can do this by opening the VSCode command pallette (`F1`) and search for `Python: Select interpreter`.
Then locate the `.venv` interpreter for your project (it will say `Venv` or `Recommended`). The name of
the interpreter should look something like this:

```text
Python 3.13.0 ('.venv': venv)
```

## Tools

### Uv

This section is referenced from the [uv documentation](https://docs.astral.sh/uv/).

The `uv` CLI is an extremely fast tool used to manage projects and packages. It creates a dependency lock file
that will make sure the same version of packages will be used everytime they are installed. It also manages your
python version for you using the [`.python-version`](.python-version) file.

#### Dependencies

You can sync dependencies using `uv` with the following command. This will make sure that only the packages you
wan't will be available in the environment

```shell
$ uv sync
Resolved 40 packages in 11ms
Audited 32 packages in 0.20ms
```

You can also install only packages required for the application to run. This will also remove dev-dependencies
as well as optional dependencies if they are installed.

<!-- TODO: Find the right command (https://docs.astral.sh/uv/concepts/dependencies/#optional-dependencies) -->

```bash
$ uv sync --no-dev
Resolved 8 packages in 5ms
Audited 2 packages in 0.20ms
```

Install dependencies with extra optional groups (e.g. lint, docs...).

```bash
$ uv sync --group lint
Resolved 8 packages in 176ms
Audited 2 packages in 0.17ms
```

#### Adding dependencies

To add new dependencies using `uv`, you can use the following command:

```shell
$ uv add prefect
Resolved 121 packages in 637ms
Prepared 82 packages in 796ms
Installed 84 packages in 148ms

$ uv add prefect==2.18.3
Resolved 125 packages in 378ms
Prepared 19 packages in 1.63s
Uninstalled 7 packages in 104ms
Installed 19 packages in 34ms
```

Add packages only used for development.

```shell
$ uv add pytest --dev
Resolved 8 packages in 4ms
Installed 4 packages in 12ms
```

Add packages to optional extras.

```bash
$ uv add ruff --group lint
Resolved 8 packages in 4ms
Audited 2 packages in 0.49ms
```

### Updating dependencies

To update dependencies, it is very simple. Just run the following command.

```shell
$ uv lock --upgrade
Updating dependencies
```

Then make sure the project still runs and you are have updated.

### Ruff

This section is referenced from the [Ruff documentation](https://docs.astral.sh/ruff/).

Ruff is the default linter and formatter for this template. Ruff is extremely powerful
because it is very customizable. As en example you can freely choose between different linters
and even combine them. You can see the list of available linters running the command below.

```shell
$ ruff linter
     ...
   F Pyflakes
 E/W pycodestyle
 DOC pydoclint
     ...
```

These linter can be added to the [ruff.toml](ruff.toml) file.

#### Linting your code

Linting your code with Ruff is very easy. You just need to run the command below.

```shell
$ ruff check
...
Found 1 error.
```

You can also use the `--fix` flag to fix all autofixable issues.

```shell
$ ruff check --fix
...
Found 1 error.
```

### Githooks

Githooks is an opt-in feature. It will automatically use `ruff` to check code before pushing
code to the remote on branches main/master and staging. To enable this feature you will have to
change the default folder that `git` checks for hooks.

```bash
git config --local core.hooksPath .githooks/
```

**NOTE:** Here we used the `--local` flag which only enables it for this repository. If desired, this
can also be added as a global config value using the `--global` flag.
