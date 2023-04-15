project = "ForecastFlowML"
copyright = "2023, Caner Turkseven"
author = "Caner Turkseven"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"
templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
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
