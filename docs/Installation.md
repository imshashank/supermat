# Setup

## Installation

1. Clone this repository
2. Setup [python-poetry](https://python-poetry.org/docs/#installation) in your system.
3. Run `poetry install --with=frontend --all-extras` in your virtual environment to install all required dependencies.
4. In terminal run `python -m supermat.gradio` to the run the gradio interface to see it in action.

## Setting up Adobe

> We weren't able to capture hierarchical structure like section, paragraph, and sentences with our open source pdf parsing libraries. We are actively working on other alternatives that can parse the pdf files with this hierarchical structure. PyMuPDF provides in page, block and lines, which isn't exactly the same.

1. Setup with Adobe PDF Services API as shown [here](https://developer.adobe.com/document-services/docs/overview/pdf-services-api/)
2. Provide the credentials in the `.env` file
3. To cache the adobe results, you can set the `TMP_DIR` environment variable to a persistent location as well
