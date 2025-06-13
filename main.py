import streamlit as st
from database import Database
from indobert import Indobert
from naivebayes import NaiveBayes

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
    
    if filter_type == "exact":
        st.session_state.dataframe_show = st.session_state.dataset.exact_filter(search_text).head(st.session_state.num_row_shows)
    elif filter_type == "contains":
        st.session_state.dataframe_show = st.session_state.dataset.contains_filter(search_text).head(st.session_state.num_row_shows)
    elif filter_type == "regex":
        st.session_state.dataframe_show = st.session_state.dataset.regex_filter(search_text).head(st.session_state.num_row_shows)
    else:
        st.session_state.dataframe_show = st.session_state.dataset.head(st.session_state.num_row_shows)

def select_text_from_row():
    selected_row = st.session_state.get("dataframe-value", None)
    if selected_row and "selection" in selected_row and "rows" in selected_row["selection"] and selected_row["selection"]["rows"]:
        row_idx = selected_row["selection"]["rows"][0]
        st.session_state.input_text = str(st.session_state.dataframe_show.iloc[row_idx].iloc[0])
    else:
        st.write("No row selected.")


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

# *************** SIDEBAR ***************
with st.sidebar:
    st.markdown("")
    with create_middle_part():
        st.markdown("# 🚀 Menu 🚀")
    st.markdown("---")
    sidebar_button("Home")
    sidebar_button("Inference")
    # sidebar_button("Topic Modelling")

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
            options=[8, 10, 20, 50], index=0, key="num_row_shows",
            on_change=lambda: st.session_state.update({"dataframe_show": st.session_state.dataset.head(st.session_state.num_row_shows)})
        )
    st.dataframe(
        st.session_state.dataframe_show,
        use_container_width=True,
        hide_index=True,
        key="dataframe-value",
        selection_mode="single-row",
        on_select=select_text_from_row,
        column_config={
            "text": st.column_config.TextColumn("Tweet", help="Content of Aceh political tweet"),
        }
    )
    with st.expander("🔎 Search Tweet"):
        st.markdown("### Search Tweet")
        st.radio("Filter", options=["regex", "exact", "contains"], index=0, key="search_filter", horizontal=True)
        search_text = st.text_input(
            "Enter text to search Aceh political tweets:",
            placeholder="Example: pilkada, governor, political party",
            on_change=search_tweet,
            key="search_text"
        )

    # Inference Section
    st.markdown("## Sentiment Analysis Inference")
    st.selectbox(
        "Select model for sentiment analysis:",
        options=["Naive Bayes", "IndoBERT", "Compare"], index=0, key="model_choice"
    )
    st.text_input(
        "Enter Aceh political tweet text for analysis or select from dataset by clicking a row:",
        placeholder="Example: this candidate only makes promises and has no real proof",
        key="input_text"
    )
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        run_inference = st.button("Analyze Sentiment", key="run_inference_btn")
    with col2:
        st.button("Random Tweet", on_click=lambda: st.session_state.update({"input_text": st.session_state.dataset.get_random_row()}))

    if st.session_state.get("input_text", "").strip() and st.session_state.get("run_inference_btn") or 'run_inference' in locals() and run_inference:
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
    Analisis topik dilakukan untuk mengidentifikasi tema besar dalam percakapan politik di Twitter Aceh menggunakan **Latent Dirichlet Allocation (LDA)**. Berikut alur dan insight dari proses topic modelling pada dataset ini:
    """)

    # 1. Data Setelah Preprocessing
    st.subheader("Data Setelah Preprocessing")
    st.markdown("""
    Setelah tahap stemming dan pembersihan, jumlah kata unik yang tersisa di korpus:
    """)
    st.write("Count of unique words in corpus: 11.943")  # atau angka sesuai output di notebook kamu

    st.image("image/word_frequency.png", caption="Distribusi Frekuensi Kata Terpopuler")

    # 2. Wordcloud dan Insight
    st.subheader("Wordcloud & Insight Awal")
    st.markdown("""
    Berikut adalah wordcloud dari kata yang sering muncul pada tweet politik Aceh setelah diproses:
    """)
    st.image("image/wordcloud.png", caption="Wordcloud Tweet Politik Aceh")

    st.markdown("""
    **Insight Wordcloud:**
    1. Dominasi topik politik: kata seperti *partai politik*, *gubernur*, dan *pilkada* sangat sering muncul.
    2. Fokus wilayah Aceh, terlihat dari sering munculnya *Aceh* dan *Banda Aceh*.
    3. Peran individu: munculnya nama tokoh-tokoh politik lokal.
    4. Isu pemilihan & partisipasi publik: kata *pilkada* dan *KIP Aceh* menonjol.
    5. Dinamika partai lokal.
    """)

    # 3. Penerapan LDA
    st.subheader("Pembentukan Topik dengan LDA")
    st.markdown("""
    Model **Latent Dirichlet Allocation (LDA)** digunakan untuk menemukan kelompok topik utama. Berikut hasil kata kunci per topik:
    """)
    st.code("""
    Topic: 1
    aceh milu pilkada kip kabupaten suara laksana pilih banda kampanye aman kota
    Topic: 2
    aceh irwandi yusuf gubernur pilkada rakyat kampanye mantan pilih patroli bebas teungku
    Topic: 3
    aceh gubernur pj rakyat ya orang indonesia perintah parpol politik milu safrizal
    Topic: 4
    aceh rakyat anies indonesia partai menang daerah pdip wali politik prabowo jokowi
    Topic: 5
    aceh partai gubernur calon politik wakil parpol milu lokal kip ketua dukung
    """, language="text")

    # 4. Insight Hasil Topik
    st.markdown("""
    **Insight dari LDA:**
    - **Topik 1:** Fokus pada pelaksanaan pemilu lokal di Aceh (pilkada, kesiapan, keamanan, dan pelaksanaan di berbagai kabupaten/kota).
    - **Topik 2:** Kampanye & figur populer seperti Irwandi Yusuf serta reaksi publik pada mantan pejabat dan kampanye mereka.
    - **Topik 3:** Kepemimpinan & politik umum (peran gubernur, partai lokal, dan hubungan ke nasional).
    - **Topik 4:** Dinamika politik nasional & lokal, termasuk pembahasan tokoh nasional (Anies, Prabowo, Jokowi) & partai nasional di konteks Aceh.
    - **Topik 5:** Calon & dukungan politik, terutama perebutan kursi gubernur/wakil dan dinamika partai lokal.

    Topik-topik ini membuktikan percakapan Twitter Aceh sangat dinamis, mencakup isu pemilu lokal, figur kunci, interaksi partai, dan hubungan politik lokal-nasional.
    """)

    st.markdown("---")
    st.caption("Gambar & insight pada halaman ini dihasilkan secara otomatis dari proses modelling di notebook EDA LDA. Silakan ganti path gambar sesuai hasil visualisasi kamu.")
