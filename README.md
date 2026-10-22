# Research Agent Studio

A multi-agent research system built with LangChain, Groq and Tavily, with a Streamlit UI.

Give it a topic and four stages run in order:

1. **Search agent** finds recent sources with Tavily
2. **Reader agent** scrapes the most relevant page with BeautifulSoup
3. **Writer chain** (LCEL) drafts a structured report
4. **Critic chain** (LCEL) scores the report and lists improvements

## Tech stack

Python, LangChain v1 (`create_agent`, LCEL), LangGraph, Groq, Tavily, BeautifulSoup, Streamlit

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env     # then add your own API keys
streamlit run app.py
```

You can also run the pipeline in the terminal with `python pipeline.py`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your `.env` file)
2. Create a new app on https://share.streamlit.io and select `app.py` as the main file
3. In Advanced settings, choose Python 3.11 or 3.12
4. Paste your keys in the app's Secrets box:

```toml
GROQ_API_KEY = "your_groq_key_here"
TAVILY_API_KEY = "your_tavily_key_here"
```

## Project structure

```
app.py            Streamlit UI
pipeline.py       Runs the four stages end to end
agents.py         Agents, prompts and LCEL chains
tools.py          web_search (Tavily) and scrape_url (BeautifulSoup)
requirements.txt  Dependencies
```
