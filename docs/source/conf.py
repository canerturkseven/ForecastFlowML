project = "ForecastFlowML"
copyright = "2023, Caner Turkseven"
author = "Caner Turkseven"

extensions = [
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "announcement": (
        "If you like ForecastFlowML, please give us a"
        "<i class='fa-solid fa-star fa-bounce' style='color: #e6d733;'></i>"
        "on <a href='https://github.com/canerturkseven/ForecastFlowML'>GitHub!</a>"
    ),
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
