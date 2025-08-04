import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# --- Load API key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Scraper ---
def fetch_sciencedaily_articles(category_url, max_articles=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(category_url, headers=headers)
    if response.status_code != 200:
        return f"Request failed with status code: {response.status_code}", []

    soup = BeautifulSoup(response.text, "html.parser")
    link_tags = soup.select("div.latest-head > a")

    articles = []
    for tag in link_tags[:max_articles]:
        title = tag.get_text(strip=True)
        link = "https://www.sciencedaily.com" + tag.get("href")
        summary_tag = tag.find_parent("div").find_next_sibling("div", class_="latest-summary")
        summary = summary_tag.get_text(strip=True) if summary_tag else ""

        articles.append({
            "title": title,
            "link": link,
            "summary": summary
        })

    return None, articles

def generate_content_idea(title, summary):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content strategist who creates engaging content ideas based on scientific articles."},
                {"role": "user", "content": f"Title: {title}\n\nSummary: {summary}\n\nCome up with a creative content idea (1–2 sentences). Content idea must be optimized for image creating AI with dall-e-3. Used for social media posts"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Failed to generate idea: {e}]"

def generate_prompt(title, summary):
    content_idea = generate_content_idea(title, summary)
    return f"""Scientific illustration based on the following concept:
            \n\n{content_idea} 
            \n\nImportant:
            \nDo not use Text!
            \nMake as realistic looking as possible.
            \nSafe route, no funky stuff
            \nUsed for social media posts
            """


def generate_image_for_article(title, summary, size="1024x1024"):
    prompt = generate_prompt(title, summary)
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=size,
            quality="standard"
        )
        return response.data[0].url
    except Exception as e:
        st.warning(f"Image generation failed for: {title}\n\n{e}")
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="ScienceDaily Scraper + AI Images")
st.title("🔬 ScienceDaily Article Scraper + AI Image Generator")

# Categories
categories = {
    "Physics": "https://www.sciencedaily.com/news/matter_energy/physics/",
    "Quantum Physics": "https://www.sciencedaily.com/news/matter_energy/quantum_physics/",
    "Astrophysics": "https://www.sciencedaily.com/news/space_time/astrophysics/",
    "Biology": "https://www.sciencedaily.com/news/plants_animals/biology/",
    "Health": "https://www.sciencedaily.com/news/health_medicine/",
}

selected_category = st.selectbox("Select a category", list(categories.keys()))
num_articles = st.slider("Number of articles", 1, 10, 5)

if st.button("Fetch Articles and Generate Images"):
    with st.spinner("Fetching articles..."):
        url = categories[selected_category]
        error, articles = fetch_sciencedaily_articles(url, max_articles=num_articles)

    if error:
        st.error(error)
    elif not articles:
        st.warning("No articles found.")
    else:
        st.success(f"Fetched {len(articles)} articles.")
        for article in articles:
            st.markdown(f"### [{article['title']}]({article['link']})")
            st.write(article["summary"])

            with st.spinner("Generating image..."):
                image_url = generate_image_for_article(article["title"], article["summary"])

            article["image_url"] = image_url

            if image_url:
                st.image(image_url, width=512)
            else:
                st.info("No image available.")

            st.markdown("---")

        # Export CSV with image URLs
        df = pd.DataFrame(articles)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="sciencedaily_articles.csv", mime="text/csv")
