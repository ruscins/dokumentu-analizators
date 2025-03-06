import os
import streamlit as st
import pandas as pd
import plotly.express as px
import io
import base64
from pdfminer.high_level import extract_text
from typing import Optional
import threading
import queue
import tenacity
import nest_asyncio
import asyncio

# Slēdzis Google GenAI funkcionalitātei
USE_GENAI = True

# Pārbaude un ielāde no .env faila
if "GOOGLE_API_KEY" not in os.environ:
    if os.path.exists(".env"):
        with open(".env", "r") as env_file:
            for line in env_file:
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# Mēģinājums importēt google.genai
if USE_GENAI:
    try:
        from google import genai
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_AVAILABLE = False
        st.warning("Bibliotēka `google-genai` nav instalēta.  AI funkcionalitāte nebūs pieejama. Instalējiet ar `pip install google-genai`.")
else:
    GENAI_AVAILABLE = False
    st.info("Google GenAI funkcionalitāte ir atslēgta.  Lai to ieslēgtu, mainiet USE_GENAI uz True.")

# Streamlit konfigurācija
st.set_page_config(
    page_title="Dokumentu analīzes asistents",
    page_icon="📊",
    layout="wide"
)

# Stils
st.markdown("""
<style>
.main { padding: 1rem; }
.title-container { display: flex; align-items: center; margin-bottom: 1rem; }
.sidebar .sidebar-content { padding: 1rem; }
.download-link { padding: 0.5em 1em; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Virsraksts
st.title("📊 Dokumentu analīzes asistents")
st.markdown("*Augšupielādējiet CSV, Excel vai PDF failus analīzei*")

# Funkcijas

def save_uploaded_file(uploaded_file):
    """Saglabā augšupielādēto failu `data` mapē."""
    try:
        data_dir = "app/data" # Definē mapes nosaukumu vienā vietā
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Kļūda saglabājot failu: {e}")
        return None

def cleanup_user_data(data_dir="app/data"):  # Pievieno noklusējuma vērtību
    """Dzēš lietotāja datus."""
    try:
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            #Iekomentēts, lai atstātu tukšu mapi.
            #os.rmdir(data_dir)  # Dzēš mapi, ja tā ir tukša
            st.success("Visi pagaidu dati dzēsti")
    except Exception as e:
        st.error(f"Kļūda dzēšot datus: {e}")

def load_data(file):
    """Ielādē datus no CSV vai Excel faila, automātiski nosakot atdalītāju."""
    try:
        if file.name.endswith('.csv'):
            # Automātiska atdalītāja noteikšana
            try:
                df = pd.read_csv(file, sep=None, engine='python')
            except Exception as e:
                st.error(f"Kļūda ielādējot CSV failu: {e}. Lūdzu, pārbaudiet, vai atdalītājs ir pareizs.")
                return None
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Neatbalstīts faila formāts")
            return None
        return df
    except Exception as e:
        st.error(f"Kļūda ielādējot datus: {e}")
        return None

def analyze_pdf(file):
    """Analizē PDF failu un atgriež teksta saturu un statistiku."""
    try:
        text = extract_text(file)

        # Vienkārša teksta analīze
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # Biežāk sastopamie vārdi (izņemot stopvārdus)
        word_freq = {}
        for word in words:
            clean_word = ''.join(c for c in word.lower() if c.isalnum())
            if len(clean_word) > 3:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "text": text,
            "word_count": word_count,
            "char_count": char_count,
            "top_words": top_words
        }
    except Exception as e:
        st.error(f"Kļūda PDF apstrādē: {e}")
        return None

def get_csv_summary(df):
    """Iegūst datu kopsavilkumu no DataFrame."""
    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]
    summary["missing_values"] = df.isna().sum().sum()

    col_types = df.dtypes.value_counts().to_dict()
    summary["column_types"] = {str(k): v for k, v in col_types.items()}

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

    return summary

def generate_insights(df):
    """Ģenerē automātiskus secinājumus par datiem."""
    insights = []

    insights.append(f"Dati satur {df.shape[0]} ierakstus ar {df.shape[1]} atribūtiem.")

    missing = df.isna().sum().sum()
    if missing > 0:
        missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
        insights.append(f"Datos ir {missing} trūkstošās vērtības ({missing_pct:.2f}% no visiem datiem).")

    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:min(3, len(numeric_cols))]:  # Droši iterē caur skaitliskajām kolonnām
        mean_val = df[col].mean()
        median_val = df[col].median()

        skew = "augstāka" if mean_val > median_val else "zemāka"
        insights.append(f"Laukam '{col}' vidējā vērtība ({mean_val:.2f}) ir {skew} nekā mediāna ({median_val:.2f}), "
                      f"kas norāda uz datu sadalījuma {skew.replace('augstāka', 'pozitīvu').replace('zemāka', 'negatīvu')} nobīdi.")

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:min(3, len(cat_cols))]:  # Iterē droši caur kategoriskajām kolonnām
        value_counts = df[col].value_counts()
        if len(value_counts) < 10:
            most_common = value_counts.index[0]
            pct = (value_counts.iloc[0] / df.shape[0]) * 100
            insights.append(f"Laukā '{col}' visbiežāk sastopamā vērtība ir '{most_common}' ({pct:.2f}% no visiem ierakstiem).")

    return insights

# Globālie mainīgie Gemini klientam
gemini_client = None

def run_in_thread(func, *args, **kwargs):
    """Palaiž funkciju atsevišķā pavedienā un atgriež rezultātu"""
    q = queue.Queue()
    def worker():
        try:
            result = func(*args, **kwargs)
            q.put(result)
        except Exception as e:
            q.put(e)  # Ieliek kļūdu rindā
    t = threading.Thread(target=worker)
    t.start()
    t.join()  # Gaida, kamēr pavediens beigs darbu
    result = q.get()
    if isinstance(result, Exception):
        raise result  # Ja kļūda, paceļam to
    return result

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1))
def initialize_gemini():
    """Inicializē Google Gemini API."""
    global gemini_client

    if not GENAI_AVAILABLE:
        st.error("Google Generative AI bibliotēka nav instalēta vai GenAI ir atslēgts. AI funkcionalitāte nav pieejama.")
        return False

    try:
        # Izveido event loop, ja tāda vēl nav
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            nest_asyncio.apply() # Ieslēdz atbalstu nested event loops
            asyncio.set_event_loop(asyncio.new_event_loop())

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY vides mainīgais nav atrasts. Lūdzu, iestatiet to.")
            return False

        st.info(f"Atrasta API atslēga, kas sākas ar: {api_key[:5]}...")

        gemini_client = genai.Client(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Kļūda inicializējot Google Gemini API: {str(e)}", icon="🚨")
        st.warning("Pārliecinieties, vai GOOGLE_API_KEY ir iestatīts kā vides mainīgais un ir derīgs.", icon="⚠️")
        return False

def llm_analyze_data(df: pd.DataFrame, prompt: str = "") -> Optional[str]:
    """Analizē datus, izmantojot Google Gemini, un atgriež detalizētas atziņas."""

    if not GENAI_AVAILABLE:
        st.warning("GenAI nav pieejams. Funkcija nedarbosies.")
        return None
    try:
        if not initialize_gemini():
            return None

        data_description = []
        data_description.append(f"DataFrame ar {df.shape[0]} rindām un {df.shape[1]} kolonnām.")
        data_description.append(f"Kolonnu nosaukumi: {', '.join(df.columns.tolist())}")

        data_description.append("Kolonnu tipi:")
        for col, dtype in df.dtypes.items():
            data_description.append(f"- {col}: {dtype}")

        data_description.append("\nDatu statistikas kopsavilkums:")
        stats_str = df.describe().to_string()
        data_description.append(stats_str)

        data_description.append("\nDatu paraugs (pirmās 5 rindas):")
        sample_str = df.head(5).to_string()
        data_description.append(sample_str)

        if not prompt:
            prompt = """
            Lūdzu, analizē šos datus un sniedz detalizētas atziņas. Tavs uzdevums:
            1. Identificē galvenās tendences un iezīmes datos
            2. Norādi uz jebkurām anomālijām vai interesantiem atklājumiem
            3. Piedāvā 3-5 hipotēzes, kas varētu izskaidrot datu īpatnības
            4. Iesaki papildu analīzes virzienus, kas būtu vērtīgi šiem datiem

            Atbildi strukturētā veidā ar skaidriem apakšvirsrakstiem.
            """

        full_prompt = "\n".join(data_description) + "\n\n" + prompt

        if gemini_client:
            # Palaiž Gemini API pieprasījumu atsevišķā pavedienā
            try:
                response = run_in_thread(gemini_client.models.generate_content, contents=full_prompt, model="gemini-2.0-flash")
                return response.text
            except Exception as e:
                st.error(f"Kļūda saņemot Gemini atbildi: {e}")
                return None
        else:
            st.error("Gemini klients nav inicializēts. Mēģiniet vēlreiz.")
            return None

    except Exception as e:
        st.error(f"Kļūda izmantojot Google Gemini: {str(e)}")
        return None

def llm_answer_question(df: pd.DataFrame, question: str) -> Optional[str]:
    """Izmanto Google Gemini, lai atbildētu uz jautājumu par datiem."""
    if not GENAI_AVAILABLE:
        st.warning("GenAI nav pieejams. Funkcija nedarbosies.")
        return None
    try:
        if not initialize_gemini():
            return None

        data_description = []
        data_description.append(f"DataFrame ar {df.shape[0]} rindām un {df.shape[1]} kolonnām.")
        data_description.append(f"Kolonnu nosaukumi: {', '.join(df.columns.tolist())}")

        data_description.append("\nDatu paraugs (pirmās 5 rindas):")
        sample_str = df.head(5).to_string()
        data_description.append(sample_str)

        data_description.append("\nStatistikas pamatinformācija:")
        stats_str = df.describe().to_string()
        data_description.append(stats_str)

        prompt = f"""
        Tev sniegti šādi dati:

        {' '.join(data_description)}

        Lūdzu, atbildi uz šo jautājumu par dotajiem datiem:
        {question}

        Atbildi jautājumam pēc iespējas precīzāk, balstoties tikai uz sniegtajiem datiem.
        """
        if gemini_client:
            # Palaiž Gemini API pieprasījumu atsevišķā pavedienā
            try:
                response = run_in_thread(gemini_client.models.generate_content, model="gemini-2.0-flash", contents=prompt)
                return response.text
            except Exception as e:
                st.error(f"Kļūda saņemot Gemini atbildi: {e}")
                return None
        else:
            st.error("Gemini klients nav inicializēts. Mēģiniet vēlreiz.")
            return None

    except Exception as e:
        st.error(f"Kļūda izmantojot Google Gemini: {str(e)}")
        return None

def create_download_link(df):
    """Izveido lejupielādes saiti apstrādātajiem datiem."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="apstradati_dati.csv" class="download-link">Lejupielādēt apstrādātos datus</a>'
    return href

# Sānjosla
with st.sidebar:
    st.header("Opcijas")

    with st.expander("Drošības iestatījumi"):
        auto_delete = st.checkbox("Automātiski dzēst datus pēc sesijas", value=True)
        if st.button("Dzēst visus datus tagad"):
            cleanup_user_data()

        st.info("""
        Šī opcija nodrošina, ka jūsu augšupielādētie un analizētie dati
        tiek droši dzēsti pēc darba pabeigšanas vai sesijas beigām.
        """)

    uploaded_file = st.file_uploader("Augšupielādēt failu", type=["csv", "xlsx", "xls", "pdf"])

    if uploaded_file is not None:
        file_details = {
            "Faila nosaukums": uploaded_file.name,
            "Faila izmērs": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("Faila detaļas:")
        for key, value in file_details.items():
            st.write(f"{key}: {value}")

# Galvenais saturs
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        if uploaded_file.name.endswith((".csv", ".xlsx", ".xls")):
            df = load_data(uploaded_file)

            if df is not None:
                st.success("Dati veiksmīgi ielādēti!")

                tab1, tab2, tab3, tab4 = st.tabs(["Dati", "Vizualizācija", "Statistika", "Secinājumi"])

                with tab1:
                    st.header("Datu priekšskatījums")

                    col1, col2 = st.columns(2)
                    with col1:
                        show_rows = st.slider("Rādīt rindas", 5, 50, 10)
                    with col2:
                        if len(df.columns) > 0:
                            selected_columns = st.multiselect("Atlasīt kolonnas", df.columns.tolist(), df.columns.tolist()[:min(5, len(df.columns))])
                        else:
                            selected_columns = []

                    if selected_columns:
                        st.dataframe(df[selected_columns].head(show_rows))
                    else:
                        st.dataframe(df.head(show_rows))

                    st.markdown(create_download_link(df), unsafe_allow_html=True)

                with tab2:
                    st.header("Datu vizualizācija")

                    viz_type = st.selectbox(
                        "Izvēlieties diagrammas tipu",
                        ["Stabiņu diagramma", "Līniju grafiks", "Izkliedes diagramma", "Histogramma", "Kaste ar ūsām"]
                    )

                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

                    if len(numeric_cols) > 0:
                        if viz_type in ["Stabiņu diagramma", "Līniju grafiks"]:
                            col1, col2 = st.columns(2)
                            with col1:
                                x_axis = st.selectbox("X ass", df.columns.tolist(), index=0)
                            with col2:
                                y_axis = st.selectbox("Y ass", numeric_cols, index=0 if numeric_cols else None)

                            if viz_type == "Stabiņu diagramma":
                                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} pēc {x_axis}")
                            else:
                                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} pēc {x_axis}")

                        elif viz_type == "Izkliedes diagramma":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                x_axis = st.selectbox("X ass", numeric_cols, index=0)
                            with col2:
                                y_axis = st.selectbox("Y ass", numeric_cols, index=min(1, len(numeric_cols)-1))
                            with col3:
                                color_by = st.selectbox("Krāsot pēc", [""] + cat_cols, index=0)

                            if color_by:
                                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} pret {x_axis}")
                            else:
                                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} pret {x_axis}")

                        elif viz_type == "Histogramma":
                            x_axis = st.selectbox("Kolonna", numeric_cols, index=0)
                            bins = st.slider("Joslu skaits", 5, 100, 20)

                            fig = px.histogram(df, x=x_axis, nbins=bins, title=f"{x_axis} histogramma")

                        elif viz_type == "Kaste ar ūsām":
                            x_axis = st.selectbox("Grupēt pēc (neobligāti)", [""] + cat_cols, index=0)
                            y_axis = st.selectbox("Vērtību kolonna", numeric_cols, index=0)

                            if x_axis:
                                fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} sadalījums pēc {x_axis}")
                            else:
                                fig = px.box(df, y=y_axis, title=f"{y_axis} sadalījums")

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Nepieciešamas skaitliskas kolonnas, lai veidotu diagrammas.")

                with tab3:
                    st.header("Statistikas analīze")

                    summary = get_csv_summary(df)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rindu skaits", summary["rows"])
                    with col2:
                        st.metric("Kolonnu skaits", summary["columns"])

                    st.subheader("Trūkstošās vērtības")
                    st.write(f"Kopējais trūkstošo vērtību skaits: {summary['missing_values']}")

                    if summary["missing_values"] > 0:
                        missing_by_col = df.isna().sum().to_dict()
                        missing_data = pd.DataFrame({
                            'Kolonna': list(missing_by_col.keys()),
                            'Trūkstošo vērtību skaits': list(missing_by_col.values())
                        })
                        missing_data = missing_data[missing_data['Trūkstošo vērtību skaits'] > 0]

                        if len(missing_data) > 0:
                            fig = px.bar(
                                missing_data,
                                x='Kolonna',
                                y='Trūkstošo vērtību skaits',
                                title="Trūkstošās vērtības pa kolonnām"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Skaitliskā statistika")
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe())
                    else:
                        st.info("Datos nav skaitlisku kolonnu.")

                    st.subheader("Kategorisko kolonnu analīze")
                    cat_cols = df.select_dtypes(include=['object']).columns

                    if len(cat_cols) > 0:
                        selected_cat_col = st.selectbox("Izvēlieties kategorisko kolonnu", cat_cols)

                        value_counts = df[selected_cat_col].value_counts().reset_index()
                        value_counts.columns = [selected_cat_col, 'Skaits']

                        fig = px.bar(
                            value_counts.head(10),
                            x=selected_cat_col,
                            y='Skaits',
                            title=f"Top 10 vērtības kolonnā '{selected_cat_col}'"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Datos nav kategorisku kolonnu.")

                with tab4:
                    st.header("Automātiski secinājumi")

                    insights = generate_insights(df)
                    for i, insight in enumerate(insights):
                        st.write(f"💡 {insight}")

                    st.subheader("Korelācijas analīze")
                    numeric_cols = df.select_dtypes(include=['number']).columns

                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()

                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Korelācijas matrica"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_value = corr_matrix.iloc[i, j]
                                if abs(corr_value) > 0.5:
                                    corr_pairs.append({
                                        'Mainīgais 1': corr_matrix.columns[i],
                                        'Mainīgais 2': corr_matrix.columns[j],
                                        'Korelācija': corr_value
                                    })

                        if corr_pairs:
                            st.subheader("Spēcīgākās korelācijas")
                            corr_df = pd.DataFrame(corr_pairs)
                            st.dataframe(corr_df.sort_values('Korelācija', ascending=False))
                    else:
                        st.info("Nepietiek skaitlisku kolonnu korelācijas analīzei.")

                    st.subheader("AI asistenta analīze")

                    if GENAI_AVAILABLE:
                        if initialize_gemini():
                            with st.spinner("AI asistents analizē datus..."):
                                try:
                                    llm_insights = llm_analyze_data(df)
                                    if llm_insights:
                                        st.markdown(llm_insights)
                                    else:
                                        st.error("Neizdevās iegūt AI analīzi. Pārbaudiet, vai API atslēga ir pareiza un pieejama.")
                                except Exception as e:
                                    st.error(f"Kļūda AI analīzē: {e}")

                            st.subheader("Uzdod jautājumu par datiem")
                            user_question = st.text_input("Jautājums:", placeholder="Piemēram: Kādi faktori visvairāk ietekmē ienākumu līmeni?")

                            if user_question:
                                with st.spinner("AI asistents atbild..."):
                                    try:
                                        answer = llm_answer_question(df, user_question)
                                        if answer:
                                            st.markdown("### Atbilde")
                                            st.markdown(answer)
                                        else:
                                            st.error("Neizdevās iegūt atbildi. Lūdzu, mēģiniet vēlreiz vai pārfrāzējiet jautājumu.")
                                    except Exception as e:
                                        st.error(f"Kļūda AI atbildē: {e}")
                        else:
                            st.error("Neizdevās inicializēt Gemini. Pārbaudiet API atslēgu.")

                    else:
                        st.warning(
                            "AI analīze nav pieejama. Pārliecinieties, vai GOOGLE_API_KEY ir iestatīts kā vides mainīgais un bibliotēka google-genai ir instalēta."
                        )

        elif uploaded_file.name.endswith(".pdf"):
            pdf_bytes = uploaded_file.getvalue()
            pdf_stream = io.BytesIO(pdf_bytes)

            pdf_analysis = analyze_pdf(pdf_stream)

            if pdf_analysis:
                st.success("PDF veiksmīgi ielādēts!")

                tab1, tab2, tab3 = st.tabs(["Teksta saturs", "Teksta analīze", "AI analīze"])

                with tab1:
                    st.header("PDF teksta saturs")
                    st.text_area("Teksts", pdf_analysis["text"], height=400)

                with tab2:
                    st.header("PDF analīze")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vārdu skaits", pdf_analysis["word_count"])
                    with col2:
                        st.metric("Rakstzīmju skaits", pdf_analysis["char_count"])

                    st.subheader("Biežāk sastopamie vārdi")

                    top_words_df = pd.DataFrame(pdf_analysis["top_words"], columns=["Vārds", "Skaits"])

                    fig = px.bar(
                        top_words_df,
                        x="Vārds",
                        y="Skaits",
                        title="Top 10 biežāk sastopamie vārdi"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(top_words_df)

                with tab3:
                    st.header("AI asistenta analīze")
                    if GENAI_AVAILABLE:
                        if initialize_gemini():
                            st.subheader("Dokumenta kopsavilkums")
                            with st.spinner("AI asistents analizē dokumentu..."):
                                try:
                                    prompt = f"""
                                    Lūdzu, izveido šī dokumenta kopsavilkumu. Dokumenta teksts:

                                    {pdf_analysis["text"][:4000]}

                                    Lūdzu, sniedz:
                                    1. Īsu kopsavilkumu (3-5 teikumi)
                                    2. Galvenās tēmas un atziņas
                                    3. Dokumenta iespējamo mērķi un auditoriju
                                    """
                                    try:
                                        response = run_in_thread(gemini_client.models.generate_content, model="gemini-2.0-flash", contents=prompt)
                                        summary = response.text
                                        st.markdown(summary)
                                    except Exception as e:
                                        st.error(f"Kļūda saņemot Gemini atbildi: {e}")

                                except Exception as e:
                                    st.error(f"Kļūda Gemini kopsavilkumā: {str(e)}")

                            st.subheader("Uzdod jautājumu par dokumentu")
                            user_question = st.text_input("Jautājums:", placeholder="Piemēram: Kāda ir dokumenta galvenā tēma?", key="pdf_question")

                            if user_question:
                                with st.spinner("AI asistents atbild..."):
                                    try:
                                        prompt = f"""
                                        Lūdzu, atbildi uz jautājumu par šo dokumentu.

                                        Dokumenta teksts:
                                        {pdf_analysis["text"][:4000]}

                                        Jautājums: {user_question}
                                        """

                                        try:
                                            response = run_in_thread(gemini_client.models.generate_content, model="gemini-2.0-flash", contents=prompt)
                                            answer = response.text
                                            st.markdown("### Atbilde")
                                            st.markdown(answer)
                                        except Exception as e:
                                            st.error(f"Kļūda saņemot Gemini atbildi: {e}")

                                    except Exception as e:
                                        st.error(f"Kļūda Gemini atbildē: {str(e)}")
                        else:
                            st.error("Neizdevās inicializēt Gemini. Pārbaudiet API atslēgu.")

                    else:
                        st.warning(
                            "AI analīze nav pieejama. Pārliecinieties, vai GOOGLE_API_KEY ir iestatīts kā vides mainīgais un bibliotēka google-genai ir instalēta."
                        )

        else:
            st.error("Neatbalstīts faila formāts")
    #Automātiska datu dzēšana
    if auto_delete:
        cleanup_user_data()