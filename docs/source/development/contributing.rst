Contributing to aimz
====================

Thank you for your interest in contributing to aimz.
Whether you're fixing a bug, adding a feature, or improving documentation—your help is appreciated.
This page explains the expected workflow, coding standards, and how to submit quality changes.


Setting up Development Environment
----------------------------------
#. `Fork <https://github.com/markean/aimz/fork>`_ the repository.
#. Clone your fork to your local development environment and add the upstream remote::

    git clone https://github.com/your-username/aimz.git
    cd aimz
    git remote add upstream https://github.com/markean/aimz.git
    git fetch upstream

#. Create a feature branch::

    git checkout -b feature/my-change

#. Install development dependencies (requires Python 3.11+).
   We recommend using `uv <https://docs.astral.sh/uv/>`_ as the package and environment manager. ::

    uv venv                     # create a virtual environment
    source .venv/bin/activate   # activate it
    uv pip install -e ."[dev]"  # install aimz and development dependencies


Making Changes
--------------
Follow these principles when editing:

* Keep each commit conceptually atomic (focused on a single logical concern).
* Prioritize clarity over clever abstractions; refactor only when duplication becomes costly.
* Update or create tests and documentation alongside code changes (do not defer).

After making your changes, stage it, for example::

    git add <path/to/modified_file.py>


Testing
-------
We use `pytest <https://pytest.org/>`_ as our testing framework.
Tests should include:

* A representative (happy path) case.
* At least one edge or failure scenario (e.g., shape mismatch, empty input, invalid parameters).

Run a specific test::

    pytest tests/<test_new_feature>.py

Run the full suite (before committing or creating a PR)::

    pytest -q

Optional coverage check::

    pytest --cov=aimz


Writing Documentation
---------------------
We use `Sphinx <https://www.sphinx-doc.org/>`_ to build the documentation.
All public APIs should have Google-style docstrings.
For larger additions:

* Add or extend documentation under ``docs/source/``.
* Use cross references for API objects, e.g. ``:py:meth:`~aimz.ImpactModel.predict```.
* Build the Sphinx docs locally (requires ``[docs]`` extra)::

   uv pip install -e ."[docs]"
   make -C docs html


Linting & Pre-commit Hooks
--------------------------
We use `Ruff <https://docs.astral.sh/ruff>`_ (lint + optional formatting) and `pre-commit <https://pre-commit.com/>`_ hooks to keep diffs clean and reviews focused on design—not style nits.

One-time setup::

    pre-commit install

Fast local checks (iterate frequently)::

    ruff check .

Auto-fix what can be fixed::

    ruff check . --fix

Run the full hook suite (after staging)::

    pre-commit run --all-files


Submitting a Pull Request
-------------------------
When opening a PR, reference the relevant issue number (if any) in the title or description.

Checklist:

* [ ] Tests pass locally and CI is green.
* [ ] Linting passes (``ruff check . `` and ``pre-commit run --all-files``).
* [ ] Documentation updated or not required.
