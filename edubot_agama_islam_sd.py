# Import library
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
import requests

st.set_page_config(page_title="Sahabat Islam untuk SD", page_icon="📚", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #f0fff4;
    }
    .stChatMessage {
        border-radius: 20px;
        padding: 12px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e6f7ff;
    }
    .stChatMessage.assistant {
        background-color: #fffbe6;
    }
</style>
""", unsafe_allow_html=True)


st.image("header.png", use_container_width=True)
st.title("📚 Belajar Agama Islam untuk SD")
st.caption("Teman belajar Islami dengan bahasa santai, sopan, dan menyenangkan ✨")
st.info("👋 Assalamu'alaikum adik-adik, yuk belajar bersama dengan Sahabat Islamc 🌙")

with st.sidebar:
    st.subheader("🔧 Pengaturan")
    google_api_key = st.text_input("Google AI API Key", type="password")
    model_name = st.text_input("Model", value="gemini-2.0-flash")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    kelas = st.selectbox("Kelas SD", ["1","2","3","4","5","6"], index=3)
    tema = st.selectbox("Tema Pelajaran", [
        "Akhlak & Adab", "Sholat & Wudhu", "Sirah Nabi",
        "Bahasa Indonesia", "Matematika Dasar", "Sains Dasar"
    ])
    use_wiki = st.checkbox("Aktifkan Wikipedia (ID)", value=True)
    if st.button("🔁 Reset Percakapan"):
        for k in ("messages","score","agent","_last_key","_last_model","_last_temp"):
            if k in st.session_state: 
                del st.session_state[k]
        st.success("Percakapan direset.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "score" not in st.session_state:
    st.session_state.score = 0

@tool
def wikipedia_ringkas(query: str) -> str:
    """Ambil ringkasan singkat dari Wikipedia Indonesia."""
    try:
        url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return f"Tidak menemukan ringkasan untuk '{query}' di Wikipedia."
        data = r.json()
        extract = data.get("extract", "")
        parts = extract.split(". ")
        return ". ".join(parts[:4]).strip()
    except Exception as e:
        return f"Gagal memanggil Wikipedia: {e}"

tools = [wikipedia_ringkas] if use_wiki else []

if not google_api_key:
    st.info("Silakan masukkan Google AI API key di sidebar untuk mulai.", icon="🗝️")
    st.stop()

if ("agent" not in st.session_state) or \
   (getattr(st.session_state, "_last_key", None) != google_api_key) or \
   (getattr(st.session_state, "_last_model", None) != model_name) or \
   (getattr(st.session_state, "_last_temp", None) != temperature):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature
    )
    SYSTEM_PROMPT = f"""
    Kamu adalah Education Bot untuk siswa SD Islam kelas {kelas}.
    Gunakan bahasa santai tapi sopan, ramah, dan mudah dipahami anak-anak.
    - Tanamkan nilai adab, salam, akhlak mulia, tolong-menolong, dan kejujuran.
    - Hindari topik rumit/berkontroversi. Sarankan tanya guru/orang tua jika sulit.
    - Jawaban inti maksimal 5 kalimat, boleh dilanjut poin langkah sederhana.
    - Jika memakai Wikipedia, sebutkan sumber singkat.
    - Materi agama dasar: rukun iman & islam, wudhu, sholat, kisah nabi; hindari topik kontroversial/fiqih mendalam.
    - Jika pertanyaan sensitif/rumit: jelaskan singkat & sarankan bertanya kepada guru/orang tua.
    - Bila menggunakan tool, sebutkan sumbernya secara singkat (misal "Wikipedia (ID)").
    - Jangan menebak fakta sejarah/ayat/hadits jika tidak yakin; katakan tidak yakin atau gunakan tool.
    - Lindungi privasi: jangan minta data pribadi (nama lengkap, alamat, nomor identitas).
    Tema saat ini: {tema}.
    """
    st.session_state.agent = create_react_agent(
        model=llm, tools=tools, prompt=SYSTEM_PROMPT
    )
    st.session_state._last_key = google_api_key
    st.session_state._last_model = model_name
    st.session_state._last_temp = temperature

for msg in st.session_state.messages:
    avatar = "👦" if msg["role"] == "user" else "🕌"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Tulis pertanyaanmu di sini… (contoh: Bagaimana cara wudhu?)")

if prompt:
    st.session_state.score += 1
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👦"):
        st.markdown(prompt)
    
    messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    
    try:
        response = st.session_state.agent.invoke({"messages": messages})
        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "Maaf, aku belum bisa menjawabnya."
    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"
    
    with st.chat_message("assistant", avatar="🕌"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.success(f"🌟 Skor Belajar Kamu: {st.session_state.score}")
