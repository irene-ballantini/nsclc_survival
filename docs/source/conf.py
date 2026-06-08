# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = 'nsclc_survival'
copyright = '2026, Irene Ballantini'
author = 'Irene Ballantini'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    #'rst2pdf.pdfbuilder',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting'
]

templates_path = []
exclude_patterns = ['build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for PDF output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples# (source start file, target name, title, author, documentclass [howto/manual]).
latex_engine = 'xelatex'
latex_documents = [('index', u'nsclc_survival.tex', u'nsclc_survival - Survival Time prediction using CT-extracted features and clinical data', u'Irene Ballantini', 'manual'),]
latex_show_pagerefs = True
latex_domain_indices = False

pdf_documents = [('index', u'nsclc_survival', u'nsclc_survival - Survival Time prediction using CT-extracted features and clinical data', u'Irene Ballantini'),]

nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_kernel_name = 'python3'
nbsphinx_output_prompt = 'Out[%s]:'
