# Copyright 2025 Eli Lilly and Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration for aimz documentation."""

import datetime
import inspect
import sys
from importlib import metadata
from pathlib import Path

import aimz

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aimz"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aimz"
copyright = f"2025-{datetime.datetime.now(tz=datetime.UTC).year}, Eli Lilly and Company"
author = "Eunseop Kim"
version = metadata.version("aimz")
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "jupyter_sphinx",
    "sphinx_copybutton",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["**.dill", "**.ipynb_checkpoints"]
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
tls_verify = False


def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    fn = str(Path(fn).relative_to(Path(aimz.__file__).parent))

    return f"https://github.com/markean/aimz/blob/main/aimz/{fn}{linespec}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "aimz"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_context = {
    "github_user": "markean",
    "github_repo": "aimz",
    "github_version": "main",
    "doc_path": "doc",
}
html_theme_options = {
    "github_url": "https://github.com/markean/aimz",
    "navbar_align": "left",
    "use_edit_page_button": True,
    "footer_end": None,
}
html_show_sourcelink = False
html_show_sphinx = False
