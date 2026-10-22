import re

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1: Search agent
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    # Prefer the raw tool output (real Tavily results with real URLs) over the LLM's rewritten summary
    raw_search = "\n\n".join(
        str(m.content) for m in search_result["messages"] if getattr(m, "type", "") == "tool"
    )
    state["search_results"] = raw_search or str(search_result["messages"][-1].content)

    print("\nSearch results:\n", state["search_results"])

    # Step 2: Reader agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is working...")
    print("=" * 50)

    urls = re.findall(r"URL:\s*(\S+)", state["search_results"])
    best_url = urls[0] if urls else None

    if best_url:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user", f"Scrape this URL and return the useful text content: {best_url}")]
        })
        state["scraped_content"] = str(reader_result["messages"][-1].content)
    else:
        state["scraped_content"] = "No URL available to scrape."

    print("\nScraped content:\n", state["scraped_content"])

    # Step 3: Writer chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    # Limit the research size to stay within the Groq tokens-per-minute limit
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results'][:3000]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content'][:3000]}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    print("\nFinal report:\n", state["report"])

    # Step 4: Critic chain
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({"report": state["report"]})

    print("\nCritic feedback:\n", state["feedback"])
    return state


if __name__ == "__main__":
    topic = input("\nEnter the research topic: ")
    run_research_pipeline(topic)
