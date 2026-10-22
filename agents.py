from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from tools import scrape_url, web_search

load_dotenv()

# ---------- Model setup ----------
# max_retries makes the Groq client wait and retry automatically on 429 rate-limit errors
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6)


# ---------- Agent 1: search ----------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a search agent. Always use the web_search tool to find "
            "recent, reliable information. Include titles and URLs in your answer."
        ),
    )


# ---------- Agent 2: reader ----------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a reader agent. Use the scrape_url tool on the URL you are given "
            "and return the useful text content. Never invent content."
        ),
    )


# ---------- Writer chain ----------
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.
Only cite URLs that appear in the research above. Never invent sources or statistics."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ---------- Critic chain ----------
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
