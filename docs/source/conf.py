# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ForecastFlowML"
copyright = "2023, Caner Turkseven"
author = "Caner Turkseven"
release = "0.0"


def setup(app):
    app.add_css_file("custom.css")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_theme_options = {
    # "show_nav_level": 2,
    "announcement": "If you like ForecastFlowML, please give us a <i class='fa-solid fa-star fa-bounce' style='color: #e6d733;'></i> on <a href='https://github.com/canerturkseven/ForecastFlowML'>GitHub!</a>",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/canerturkseven/ForecastFlowML",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/in/canerturkseven/",
            "icon": "fa-brands fa-linkedin",
        },
    ],
    "show_prev_next": False,
}
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]
nb_execution_timeout = -1
nb_execution_mode = "off"
