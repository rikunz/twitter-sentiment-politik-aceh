import streamlit as st
from database import Database
from indobert import Indobert
from naivebayes import NaiveBayes
from nodes.LlmAgent import ChatAzure
from twitter_crawler import TwitterCrawler

# ********* functions utils
def create_middle_part():
    cols = st.columns([1, 2, 1])
    return cols[1]

def sidebar_button(label):
    if st.button(label, use_container_width=True):
        st.session_state.sidebar_value = label

def search_tweet():
    search_text = st.session_state.get("search_text", "")
    filter_type = st.session_state.get("search_filter", "contains")
    num_row_shows = st.session_state.get("num_row_shows", 8) if isinstance(st.session_state.get("num_row_shows"), int) else 0
    if filter_type == "exact":
        st.session_state.dataframe_show = st.session_state.dataset.exact_filter(search_text).head(num_row_shows) \
            if num_row_shows > 0 else st.session_state.dataset.exact_filter(search_text)
    elif filter_type == "contains":
        st.session_state.dataframe_show = st.session_state.dataset.contains_filter(search_text).head(num_row_shows) \
            if num_row_shows > 0 else st.session_state.dataset.contains_filter(search_text)
    elif filter_type == "regex":
        st.session_state.dataframe_show = st.session_state.dataset.regex_filter(search_text).head(num_row_shows) \
            if num_row_shows > 0 else st.session_state.dataset.regex_filter(search_text)
    else:
        st.session_state.dataframe_show = st.session_state.dataset.head(num_row_shows)

def select_text_from_row():
    selected_row = st.session_state.get("dataframe-value", None)
    if selected_row and "selection" in selected_row and "rows" in selected_row["selection"] and selected_row["selection"]["rows"]:
        row_idx = selected_row["selection"]["rows"][0]
        st.session_state.input_text = str(st.session_state.dataframe_show.iloc[row_idx].iloc[0])
    else:
        st.write("No row selected.")
    
def summarize_tweet():
    selected_row = st.session_state.get("dataframe-value", None)
    if selected_row and "selection" in selected_row and "rows" in selected_row["selection"] and selected_row["selection"]["rows"]:
        rows = selected_row["selection"]["rows"]
        tweets = st.session_state.dataframe_show.iloc[rows]["text"].tolist()
        joined_tweets = "\n".join([str(t) for t in tweets])
        summary = st.session_state.chat_agent.summarize_text(joined_tweets)
        if summary:
            st.session_state.summary = summary
        else:
            st.warning("Failed to summarize the selected tweets.")
    else:
        st.warning("Please select at least one row from the table for summarization.")

# Inisialisasi session_state
if "sidebar_value" not in st.session_state:
    st.session_state.sidebar_value = "Home"
if "num_row_shows" not in st.session_state:
    st.session_state.num_row_shows = 8
if "naive_bayes" not in st.session_state:
    st.session_state.naive_bayes = NaiveBayes("./model/model_naive.joblib", "./model/naive_vectorizer.joblib", "./model/kamuskatabaku.xlsx")
if "indobert" not in st.session_state:
    st.session_state.indobert = Indobert()
if "dataset" not in st.session_state:
    st.session_state.dataset = Database("./data/database.csv")
if "dataframe_show" not in st.session_state:
    st.session_state.dataframe_show = st.session_state.dataset.head(8)
if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ChatAzure()
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "twitter_crawler" not in st.session_state:
    st.session_state.twitter_crawler = TwitterCrawler()
if "crawler_summary" not in st.session_state:
    st.session_state.crawler_summary = ""
if "crawler_dataframe" not in st.session_state:
    st.session_state.crawler_dataframe = None

# *************** SIDEBAR ***************
with st.sidebar:
    st.markdown("")
    with create_middle_part():
        st.markdown("# 🚀 Menu 🚀")
    st.markdown("---")
    sidebar_button("Home")
    sidebar_button("Inference")
    # sidebar_button("Topic Modelling")
    sidebar_button("Live Crawling")

# *************** MAIN CONTENT ***************
st.title("Sentiment Analysis App")


selected = st.session_state.sidebar_value

if selected == "Home":
    st.header("📊 Welcome to the Aceh Local Party Sentiment Analysis Application")

    st.markdown("""
    Sentiment analysis of public opinion on social media X provides valuable insights into understanding how people perceive local political parties in Aceh. This application is designed to analyze public sentiment towards Aceh's local political parties on the X platform, using both machine learning and deep learning algorithms. The dataset consists of 8,000 tweets collected from X, covering the period from the 2019 Election (April 17, 2019) to the 2024 Aceh Pilkada (November 27, 2024).
    """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Project Overview")
        st.markdown("""
        This project aims to uncover public sentiment trends towards politics in Aceh by analyzing Twitter data. Our sentiment analysis leverages two robust models:

        - **Naive Bayes**: A classic machine learning approach for text classification
        - **IndoBERT**: An advanced deep learning model specifically trained for the Indonesian language

        By comparing the results from both models, we deliver comprehensive sentiment insights on political topics relevant to Aceh's social and political landscape.
        """)

    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Coat_of_arms_of_Aceh.svg/120px-Coat_of_arms_of_Aceh.svg.png", 
                caption="Aceh Province, Indonesia")

    st.subheader("📂 Dataset Information")
    st.markdown("""
    The dataset for this project consists of political tweets collected from users between 2019 and 2024, offering a comprehensive temporal perspective on public sentiment regarding political issues in Aceh.
    """)

    with st.expander("Data Collection Methodology"):
        st.markdown("""
        ### Data Collection Process

        Political tweets were gathered using the Twitter API with these parameters:
        - **Content Filter**: Keywords and hashtags related to local Aceh political parties
        - **Time Period**: April 17, 2019 to November 27, 2024
        - **Language**: Indonesian
        - **Volume**: Approximately 8,000+ political tweets after cleaning and preprocessing

        ### Preprocessing Steps

        1. **Data Cleaning**: Removing irrelevant elements such as repeated words, emojis/emoticons, URL links, usernames, and hashtags
        2. **Case Folding**: Converting all text to lowercase
        3. **Tokenization**: Using the `word_tokenize` function from the NLTK library
        4. **Normalization**: Converting non-standard words or sentences into standard forms
        5. **Stopwords Removal**: Eliminating unnecessary words using the NLTK library
        6. **Stemming**: Reducing words to their root form using Sastrawi
        """)

    with st.expander("Dataset Overview"):
        st.markdown("""
        ### Data Distribution

        - **Temporal Coverage**: Tweets span from 2019 to 2024, capturing key political events in Aceh
        - **Focus Topics**: Local political parties, elections, government policies, and public opinion
        - **User Demographics**: Diverse users from Aceh, including voters, political activists, and observers

        ### Example Topics Covered

        The dataset includes tweets related to:
        - Local and national elections in Aceh
        - Performance of local government and public policies
        - Political party activities and campaigns
        - Public sentiment towards local political figures and parties
        """)

    st.subheader("🚀 Get Started")
    st.markdown("""
    To explore sentiment analysis results:
    1. Go to the **Inference** section to analyze custom text input
    2. Visit the **Topic Modelling** section to discover key political themes in the dataset

    To explore the results of sentiment analysis, you can easily do so by opening the Inference section to analyze custom text input.
    """)

    st.markdown("---")
    st.caption("© 2025 Aceh Local Party Sentiment Analysis Project | Data collected ethically in accordance with platform policies")
elif selected == "Inference":
    st.markdown("# 🔍 Sentiment Analysis of Aceh Political Tweets")
    st.markdown("## Dataset: 8000+ Aceh Political Tweets (2019-2024)")
    columns = st.columns([1, 3])
    with columns[0]:
        st.selectbox(
            "Select number of rows to display:",
            options=[8, 10, 20, 50, "All"], index=0, key="num_row_shows",
            on_change=lambda: st.session_state.update({"dataframe_show": st.session_state.dataset.head(st.session_state.num_row_shows) if isinstance(st.session_state.num_row_shows, int) else st.session_state.dataset.head(0)})
        )

    st.dataframe(
        st.session_state.dataframe_show,
        use_container_width=True,
        hide_index=True,
        key="dataframe-value",
        selection_mode="multi-row",
        on_select=lambda: True,
        column_config={
            "text": st.column_config.TextColumn("Tweet", help="Content of Aceh political tweet"),
        }
    )
    

    if st.session_state.summary:
        st.text_area(
            "Summary of Selected Tweets",
            value=st.session_state.summary,
            key="summary",
            height=150
        )
        st.button("Send Summary to Input Text",
                  on_click=lambda: st.session_state.update({"input_text": st.session_state.summary}),
                  key="send_summary_btn")
    with st.expander("🔎 Search Tweet"):
        st.markdown("### Search Tweet")
        st.radio("Filter", options=["regex", "exact", "contains"], index=0, key="search_filter", horizontal=True)
        search_text = st.text_input(
            "Enter text to search Aceh political tweets:",
            placeholder="Example: pilkada, governor, political party",
            on_change=search_tweet,
            key="search_text"
        )
    replace_col, summary_col, _ = st.columns([2, 1, 1])
    with replace_col:
        if st.button("Replace Text Area with Selected Tweets"):
            selected_row = st.session_state.get("dataframe-value", None)
            if selected_row and "selection" in selected_row and "rows" in selected_row["selection"]:
                rows = selected_row["selection"]["rows"]
                tweets = st.session_state.dataframe_show.iloc[rows]["text"].tolist()
                joined_tweets = "\n".join([str(t) for t in tweets])
                st.session_state.input_text = joined_tweets
            else:
                st.warning("No rows selected to replace.")

    with summary_col:
        selected_row = st.session_state.get("dataframe-value", None)
        # Enable button only if there is at least one selected row
        if (
            selected_row and
            "selection" in selected_row and
            "rows" in selected_row["selection"] and
            selected_row["selection"]["rows"]
        ):
            st.button("Summary", on_click=summarize_tweet, key="summarize_btn")
        else:
            st.button("Summary", disabled=True, key="summarize_btn")


    # Inference Section
    st.markdown("## Sentiment Analysis Inference")
    st.selectbox(
        "Select model for sentiment analysis:",
        options=["Naive Bayes", "IndoBERT", "Compare"], index=0, key="model_choice"
    )
    st.text_area(
        "Enter Aceh political tweet text for analysis or select from dataset by clicking a row:",
        placeholder="Example: this candidate only makes promises and has no real proof",
        key="input_text"
    )
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        run_inference = st.button("Analyze Sentiment", key="run_inference_btn")
    with col2:
        st.button("Random Tweet", on_click=lambda: st.session_state.update({"input_text": st.session_state.dataset.get_random_row()}))

    if (
        st.session_state.get("input_text", "").strip() and
        (st.session_state.get("run_inference_btn") or ('run_inference' in locals() and run_inference))
    ):
        input_text = st.session_state.input_text
        nb_model = st.session_state.naive_bayes
        ib_model = st.session_state.indobert

        pre_nb = nb_model.preprocess(input_text)
        pre_ib = ib_model.preprocess(input_text)

        st.markdown("### 🔄 Preprocessing Results")
        st.write(f"**Naive Bayes:** `{pre_nb}`")
        st.write(f"**IndoBERT:** `{pre_ib}`")

        model_choice = st.session_state.model_choice

        if model_choice == "Compare":
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(
                    "<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa'>",
                    unsafe_allow_html=True)
                st.markdown("#### Naive Bayes")
                result_nb = nb_model.predict(input_text)
                st.write(f"**Sentiment:** {result_nb}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_right:
                st.markdown(
                    "<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa'>",
                    unsafe_allow_html=True)
                st.markdown("#### IndoBERT")
                result_ib = ib_model.predict(input_text)
                st.write(f"**Sentiment:** {result_ib['label']} (Score: {result_ib['score']})")
                st.markdown("</div>", unsafe_allow_html=True)
        elif model_choice == "Naive Bayes":
            st.markdown(
                "<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa; width:50%'>",
                unsafe_allow_html=True)
            st.markdown("#### Naive Bayes")
            result_nb = nb_model.predict(input_text)
            st.write(f"**Sentiment:** {result_nb}")
            st.markdown("</div>", unsafe_allow_html=True)
        elif model_choice == "IndoBERT":
            st.markdown(
                "<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa; width:50%'>",
                unsafe_allow_html=True)
            st.markdown("#### IndoBERT")
            result_ib = ib_model.predict(input_text)
            st.write(f"**Sentiment:** {result_ib['label']} (Score: {result_ib['score']})")
            st.markdown("</div>", unsafe_allow_html=True)
    elif not st.session_state.get("input_text", "").strip():
        st.warning("Please enter text first for sentiment analysis.")
    
elif selected == "Topic Modelling":
    st.header("Topic Modelling")
    st.markdown("""
    Topic analysis is performed to identify major themes in Aceh political conversations on Twitter using **Latent Dirichlet Allocation (LDA)**. Below is the workflow and insights from the topic modelling process on this dataset:
    """)

    # 1. Data After Preprocessing
    st.subheader("Data After Preprocessing")
    st.markdown("""
    After stemming and cleaning, the number of unique words remaining in the corpus:
    """)
    st.write("Count of unique words in corpus: 11,943")  # or the actual number from your notebook

    st.image("image/word_frequency.png", caption="Distribution of Most Frequent Words")

    # 2. Wordcloud and Initial Insights
    st.subheader("Wordcloud & Initial Insights")
    st.markdown("""
    Here is the wordcloud of frequently appearing words in Aceh political tweets after processing:
    """)
    st.image("image/wordcloud.png", caption="Wordcloud of Aceh Political Tweets")

    st.markdown("""
    **Wordcloud Insights:**
    1. Dominance of political topics: words like *political party*, *governor*, and *election* appear very frequently.
    2. Aceh regional focus, seen from frequent mentions of *Aceh* and *Banda Aceh*.
    3. Role of individuals: names of local political figures appear.
    4. Election issues & public participation: words like *election* and *KIP Aceh* stand out.
    5. Dynamics of local parties.
    """)

    # 3. LDA Application
    st.subheader("Topic Formation with LDA")
    st.markdown("""
    The **Latent Dirichlet Allocation (LDA)** model is used to discover main topic groups. Here are the keyword results for each topic:
    """)
    st.code("""
    Topic: 1
    aceh milu pilkada kip kabupaten suara laksana pilih banda kampanye aman kota
    Topic: 2
    aceh irwandi yusuf governor election people campaign former choose patrol free teungku
    Topic: 3
    aceh governor pj people yes person indonesia command parpol politics milu safrizal
    Topic: 4
    aceh people anies indonesia party win region pdip mayor politics prabowo jokowi
    Topic: 5
    aceh party governor candidate politics deputy parpol election local kip chairman support
    """, language="text")

    # 4. Topic Insights
    st.markdown("""
    **LDA Insights:**
    - **Topic 1:** Focus on the implementation of local elections in Aceh (elections, readiness, security, and execution in various districts/cities).
    - **Topic 2:** Campaigns & popular figures such as Irwandi Yusuf and public reactions to former officials and their campaigns.
    - **Topic 3:** Leadership & general politics (role of governor, local parties, and national relations).
    - **Topic 4:** National & local political dynamics, including discussion of national figures (Anies, Prabowo, Jokowi) & national parties in the Aceh context.
    - **Topic 5:** Candidates & political support, especially the contest for governor/deputy seats and local party dynamics.

    These topics show that Aceh Twitter conversations are very dynamic, covering local election issues, key figures, party interactions, and local-national political relations.
    """)

    st.markdown("---")
    st.caption("Images & insights on this page are automatically generated from the modelling process in the EDA LDA notebook. Please adjust image paths according to your visualization results.")

elif selected == "Live Crawling":
    st.header("Live Crawling of Aceh Political Tweets")
    st.markdown("""
    This section allows you to crawl live tweets related to Aceh political topics using the Twitter API. 
    You can specify keywords and the number of tweets to retrieve.
    """)

    keyword = st.text_input(
        "Enter keyword to search for Aceh political tweets:",
        placeholder="Example: pilkada, governor, political party",
        key="crawler_keyword"
    )
    num_tweets = st.number_input(
        "Number of tweets to retrieve:", min_value=1, max_value=20, value=10, step=1, key="num_tweets"
    )
    access_token = st.text_input(
        "Enter your Twitter Auth Token (optional):",
        placeholder="Your Twitter API Auth Token",
        key="access_token"
    )

    # Crawl tweets button
    if st.button("Crawl Tweets"):
        if not keyword.strip():
            st.warning("Please enter a keyword to search for tweets.")
        else:
            with st.spinner("Crawling tweets and summarizing..."):
                try:
                    st.session_state.crawler_dataframe = st.session_state.twitter_crawler.crawl_tweets(keyword, num_tweets)
                    if st.session_state.crawler_dataframe.empty:
                        st.warning("No tweets found for the given keyword.")
                    else:
                        # Precompute summary after crawling
                        st.session_state.crawler_summary = st.session_state.chat_agent.summarize_text(
                            "\n".join(st.session_state.crawler_dataframe["full_text"].tolist())
                        )
                except Exception as e:
                    st.error(f"Error while crawling tweets: {e}")

    if (
        "crawler_dataframe" in st.session_state 
        and st.session_state.crawler_dataframe is not None
    ):
        df = st.session_state.crawler_dataframe
        summary = st.session_state.get("crawler_summary", "")

        st.dataframe(
            df,
            use_container_width=True,
            key="crawler-dataframe-value",
            selection_mode="multi-row",
            on_select=lambda: True
        )
        
        st.text_area(
            "Summary of Crawled Tweets",
            value=summary,
            height=150,
            key="crawler_summary_area"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            # Button: Insert summary
            if st.button("Insert Summary to Analysis Text Area"):
                with st.spinner("Inserting summary..."):
                    st.session_state.crawler_input_text = summary
        
        with col2:
            # Insert selected tweets to text area
            if st.button("Insert Selected Tweets to Text Area"):
                with st.spinner("Inserting selected tweets..."):
                    selected_crawler_rows = st.session_state.get("crawler-dataframe-value", None)
                    if (
                        selected_crawler_rows
                        and "selection" in selected_crawler_rows
                        and "rows" in selected_crawler_rows["selection"]
                        and selected_crawler_rows["selection"]["rows"]
                    ):
                        rows = selected_crawler_rows["selection"]["rows"]
                        tweets = df.iloc[rows]["full_text"].tolist()
                        st.session_state.crawler_input_text = "\n".join(map(str, tweets))
                    else:
                        st.warning("No tweets selected to insert.")

        # Sentiment Analysis section
        st.markdown("## Sentiment Analysis on Crawled Tweets")

        st.text_area(
            "Tweets for analysis:",
            value=st.session_state.get("crawler_input_text", ""),
            key="crawler_input_text",
            height=150
        )

        # Model selection
        st.selectbox(
            "Select model for sentiment analysis:",
            options=["Naive Bayes", "IndoBERT", "Compare"],
            index=0,
            key="crawler_model_choice"
        )

        # Analyze button
        if st.button("Analyze Selected Tweets", key="crawler_analyze_btn"):
            with st.spinner("Analyzing..."):
                input_text = st.session_state.get("crawler_input_text", "").strip()
                if not input_text:
                    st.warning("Please select at least one tweet for analysis.")
                else:
                    nb_model = st.session_state.naive_bayes
                    ib_model = st.session_state.indobert

                    pre_nb = nb_model.preprocess(input_text)
                    pre_ib = ib_model.preprocess(input_text)

                    st.markdown("### 🔄 Preprocessing Results")
                    st.write(f"**Naive Bayes:** `{pre_nb}`")
                    st.write(f"**IndoBERT:** `{pre_ib}`")

                    model_choice = st.session_state.crawler_model_choice
                    if model_choice == "Compare":
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown("<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa'>", unsafe_allow_html=True)
                            st.markdown("#### Naive Bayes")
                            result_nb = nb_model.predict(input_text)
                            st.write(f"**Sentiment:** {result_nb}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col_right:
                            st.markdown("<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa'>", unsafe_allow_html=True)
                            st.markdown("#### IndoBERT")
                            result_ib = ib_model.predict(input_text)
                            st.write(f"**Sentiment:** {result_ib['label']} (Score: {result_ib['score']})")
                            st.markdown("</div>", unsafe_allow_html=True)
                    elif model_choice == "Naive Bayes":
                        st.markdown("<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa; width:50%'>", unsafe_allow_html=True)
                        st.markdown("#### Naive Bayes")
                        result_nb = nb_model.predict(input_text)
                        st.write(f"**Sentiment:** {result_nb}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif model_choice == "IndoBERT":
                        st.markdown("<div style='border:2px solid #e5e5e5; border-radius:8px; padding:16px; background:#fafafa; width:50%'>", unsafe_allow_html=True)
                        st.markdown("#### IndoBERT")
                        result_ib = ib_model.predict(input_text)
                        st.write(f"**Sentiment:** {result_ib['label']} (Score: {result_ib['score']})")
                        st.markdown("</div>", unsafe_allow_html=True)