
import streamlit as st

from document_pipeline import (
    process_document,
    search_document,
    search_multiple_documents,
    create_bm25_index
)

from rag import (
    generate_answer,
    summarize_document,
    extract_information,
    calculate_grounding_score
)

from embedding import generate_embeddings

from vector_store import (
    search_faiss,
    load_vector_database,
    load_multiple_documents,
    list_documents,
    create_document_id,
    delete_document
)


st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>
    /* ---------- Global ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(82, 68, 255, 0.10), transparent 28%),
            radial-gradient(circle at 95% 15%, rgba(0, 200, 170, 0.07), transparent 24%),
            #080b12;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 11, 18, 0.72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1320 0%, #0a0e17 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #8f7cff, #63d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #9aa6bd;
        font-size: 1rem;
        margin-top: 0.45rem;
        margin-bottom: 0;
    }

    .brand {
        padding: 0.25rem 0 1.15rem 0;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f5f7ff;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #7f8ba3;
        font-size: 0.72rem;
        margin-top: 0.28rem;
    }

    /* ---------- Cards ---------- */
    .feature-card {
        min-height: 108px;
        padding: 1.05rem 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(145deg, rgba(25,32,51,0.96), rgba(15,20,32,0.96));
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }

    .feature-icon {
        font-size: 1.55rem;
        margin-bottom: 0.4rem;
    }

    .feature-title {
        font-weight: 750;
        color: #f3f6ff;
        font-size: 0.98rem;
    }

    .feature-text {
        color: #8e9ab0;
        font-size: 0.76rem;
        margin-top: 0.25rem;
    }

    .section-card {
        padding: 1.05rem 1.15rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.075);
        background: rgba(13,18,29,0.78);
    }

    .active-doc {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(94, 113, 255, 0.28);
        background: linear-gradient(90deg, rgba(77,66,255,0.12), rgba(30,180,180,0.06));
    }

    .active-doc-title {
        font-weight: 700;
        color: #eef1ff;
    }

    .active-doc-meta {
        color: #8d99ae;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(255,255,255,0.08);
        background: #151c2b;
        color: #edf1ff;
        font-weight: 650;
        min-height: 42px;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: rgba(112, 103, 255, 0.65);
        background: #202942;
        color: white;
        transform: translateY(-1px);
    }

    /* ---------- Upload ---------- */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(112, 126, 255, 0.45);
        border-radius: 16px;
        padding: 0.35rem;
        background: rgba(20, 27, 44, 0.62);
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        padding: 0.85rem 0.95rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.065);
        background: rgba(17,23,36,0.78);
    }

    [data-testid="stMetricLabel"] {
        color: #8793aa;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,0.055);
        border-radius: 16px;
        margin-bottom: 0.65rem;
        background: rgba(15,20,31,0.58);
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* ---------- Sidebar controls ---------- */
    [data-testid="stSidebar"] .stMultiSelect,
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.35rem;
    }

    .model-pill {
        padding: 0.7rem 0.85rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        background: rgba(19,25,39,0.9);
        color: #cdd5e7;
        font-size: 0.82rem;
    }

    .model-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6cf0bd;
        margin-right: 8px;
        box-shadow: 0 0 10px rgba(108,240,189,0.55);
    }

    /* ---------- Sources ---------- */
    .source-chip {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(79, 91, 255, 0.13);
        color: #aeb5ff;
        font-size: 0.72rem;
        margin-right: 0.25rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}


    /* ---------- Premium workspace ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(12,17,28,0.65);
        padding: 0.35rem;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.55rem 0.9rem;
        color: #8e9ab0;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(76, 87, 255, 0.18);
        color: #eef1ff;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "document_processed": False,
    "chunks": None,
    "embeddings": None,
    "index": None,
    "bm25_index": None,
    "tables": [],
    "chat_history": [],
    "file_name": None,
    "document_id": None,
    "summary": None,
    "extracted": None,
    "database_loaded": False,
    "selected_document_ids": [],
    "multi_documents": [],
    "multi_mode": False,
    "pending_question": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def render_sources(results):
    with st.expander("📚 View Sources"):
        for i, result in enumerate(results):
            st.markdown(f"### Source {i + 1}")

            document_name = result.get(
                "document_name",
                st.session_state.file_name or "Document"
            )

            page = result.get("page", "Unknown")

            st.write(
                f"📄 **{document_name} — Page {page}**"
            )

            if result.get("type") == "table":
                st.write("📊 **Retrieved table**")

            st.write(result.get("text", ""))

            st.caption(
                f"🔀 Hybrid Score: "
                f"{result.get('hybrid_score', 0):.4f}"
            )

            st.caption(
                f"🧠 Semantic Score: "
                f"{result.get('semantic_score', 0):.4f}"
            )

            st.caption(
                f"🔤 Keyword Score: "
                f"{result.get('keyword_score', 0):.4f}"
            )

            st.divider()


def render_grounding(grounding):
    st.markdown("### 🎯 Answer Grounding")

    score = grounding["score"]

    if score >= 75:
        st.success(
            f"🟢 Strongly Supported — {score}%"
        )
    elif score >= 50:
        st.warning(
            f"🟡 Moderately Supported — {score}%"
        )
    else:
        st.error(
            f"🔴 Weakly Supported — {score}%"
        )

    st.progress(int(score))
    st.caption(grounding["explanation"])


def reset_active_document():
    st.session_state.document_processed = False
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.bm25_index = None
    st.session_state.tables = []
    st.session_state.file_name = None
    st.session_state.document_id = None
    st.session_state.summary = None
    st.session_state.extracted = None
    st.session_state.database_loaded = False
    st.session_state.multi_mode = False


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div style="padding: 0.6rem 0 1.1rem 0;">
            <div class="hero-title">AI Document Intelligence</div>
            <div class="hero-subtitle">
                Your personal AI workspace for searching, understanding,
                summarizing and extracting information from documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.summary = None
        st.session_state.extracted = None
        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div style="font-size:2rem;">📄</div>
            <div class="brand-title">AI Document<br>Intelligence</div>
            <div class="brand-subtitle">Your personal document assistant</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-pill"><span class="model-dot"></span>'
        '<b>Local AI</b><br><span style="margin-left:16px;color:#8e9ab0;">'
        'llama3.2:3b</span></div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📚 Document Library")

    saved_documents = list_documents()

    if saved_documents:

        document_options = {
            document.get(
                "document_id"
            ): document.get(
                "file_name",
                "Unknown document"
            )
            for document in saved_documents
        }

        selected_ids = st.multiselect(
            "Select documents for cross-document RAG",
            options=list(document_options.keys()),
            default=[
                doc_id
                for doc_id in st.session_state.selected_document_ids
                if doc_id in document_options
            ],
            format_func=lambda x:
                document_options.get(
                    x,
                    "Unknown document"
                )
        )

        st.session_state.selected_document_ids = (
            selected_ids
        )

        if selected_ids:

            if st.button(
                "🔎 Activate Selected Documents",
                use_container_width=True
            ):

                loaded = load_multiple_documents(
                    selected_ids
                )

                if loaded:

                    st.session_state.multi_documents = (
                        loaded
                    )

                    st.session_state.multi_mode = True

                    st.session_state.document_processed = True

                    st.session_state.file_name = (
                        "Multiple selected documents"
                    )

                    st.session_state.chat_history = []

                    st.session_state.summary = None
                    st.session_state.extracted = None

                    st.success(
                        f"✅ {len(loaded)} documents activated."
                    )

                    st.rerun()

        st.divider()

        st.subheader("🗑️ Document Management")

        delete_id = st.selectbox(
            "Select document to delete",
            options=list(document_options.keys()),
            format_func=lambda x:
                document_options.get(
                    x,
                    "Unknown document"
                ),
            key="delete_document_selector"
        )

        if st.button(
            "🗑️ Delete Selected Document",
            use_container_width=True
        ):

            if delete_document(delete_id):

                st.session_state.selected_document_ids = [
                    doc_id
                    for doc_id in
                    st.session_state.selected_document_ids
                    if doc_id != delete_id
                ]

                if (
                    st.session_state.document_id
                    == delete_id
                ):
                    reset_active_document()

                st.success("Document deleted.")
                st.rerun()

    else:
        st.info(
            "No saved documents yet."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.success(
            "Conversation cleared!"
        )


# ============================================================
# UPLOAD
# ============================================================

st.markdown("### 📤 Add Documents")
st.caption("Upload a PDF to process it with OCR, semantic retrieval and RAG.")

uploaded_file = st.file_uploader(
    "Drop a PDF here or browse your computer",
    type=["pdf"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()

    uploaded_document_id = create_document_id(
        uploaded_file.name,
        uploaded_bytes
    )

    if (
        st.session_state.document_id
        != uploaded_document_id
    ):

        with st.spinner(
            "🔄 Processing document..."
        ):

            try:

                (
                    chunks,
                    embeddings,
                    index,
                    bm25_index,
                    tables
                ) = process_document(
                    uploaded_bytes,
                    uploaded_file.name
                )

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.index = index
                st.session_state.bm25_index = bm25_index
                st.session_state.tables = tables

                st.session_state.document_processed = True
                st.session_state.database_loaded = False

                st.session_state.file_name = uploaded_file.name
                st.session_state.document_id = uploaded_document_id

                st.session_state.chat_history = []
                st.session_state.summary = None
                st.session_state.extracted = None

                st.session_state.multi_mode = False

                st.success(
                    f"✅ Processed: {uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    f"❌ Processing failed: {e}"
                )

                st.stop()


# ============================================================
# MULTI DOCUMENT MODE
# ============================================================

if (
    st.session_state.multi_mode
    and st.session_state.multi_documents
):

    documents = st.session_state.multi_documents

    st.markdown(
        f"""
        <div class="active-doc">
            <div class="active-doc-title">📚 Cross-document workspace</div>
            <div class="active-doc-meta">
                {len(documents)} documents are active. Ask one question
                and retrieve evidence across the selected files.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📚 Documents",
            len(documents)
        )

    with col2:
        total_chunks = sum(
            len(document["chunks"])
            for document in documents
        )

        st.metric(
            "✂️ Total Chunks",
            total_chunks
        )

    with col3:
        st.metric(
            "💬 Questions",
            len(st.session_state.chat_history)
        )

    st.divider()

    st.markdown("### 💬 Ask Across Your Documents")
    st.caption("Compare information and find supporting evidence across multiple files.")

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):

            st.write(chat["answer"])

            if "grounding" in chat:
                render_grounding(
                    chat["grounding"]
                )

            render_sources(
                chat["sources"]
            )

    question = st.chat_input(
        "Ask a question across the selected documents..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "🔎 Searching across documents..."
                ):

                    results = search_multiple_documents(
                        question,
                        documents,
                        top_k=6
                    )

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        results
                    )

                st.write(answer)

                grounding = calculate_grounding_score(
                    question,
                    answer,
                    results
                )

                render_grounding(
                    grounding
                )

                render_sources(
                    results
                )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": results,
                    "grounding": grounding
                })

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )

    st.stop()


# ============================================================
# NO DOCUMENT
# ============================================================

if not st.session_state.document_processed:

    st.markdown(
        """
        <div style="padding:1.1rem 0 0.8rem 0;">
            <h2 style="margin-bottom:0.2rem;">Welcome to your document workspace 👋</h2>
            <p style="color:#8e9ab0;">
                Upload a PDF above or activate documents from your library to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    feature_cols = st.columns(4)

    features = [
        ("📤", "Upload & Process", "Extract text from digital and scanned PDFs."),
        ("💬", "Ask Questions", "Get grounded answers with document citations."),
        ("📝", "Summarize", "Generate concise summaries from long documents."),
        ("📊", "Extract Information", "Find tables, fields and structured content.")
    ]

    for column, (icon, title, description) in zip(feature_cols, features):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    st.markdown("### ✨ What you can do")
    capabilities = st.columns(3)

    capability_groups = [
        ("🔎 Retrieval", "Semantic search, BM25 keyword search, hybrid retrieval and cross-document RAG."),
        ("📄 Document Understanding", "OCR, table extraction, intelligent chunking and page-level evidence."),
        ("🧠 AI Analysis", "Local Llama answers, fast summaries, structured extraction and grounding scores.")
    ]

    for column, (title, description) in zip(capabilities, capability_groups):
        with column:
            st.markdown(
                f"""
                <div class="section-card">
                    <b>{title}</b>
                    <div style="color:#8e9ab0;font-size:0.82rem;margin-top:0.35rem;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# SINGLE DOCUMENT MODE
# ============================================================

document_name = (
    st.session_state.file_name
    or "Unknown document"
)

# ---------- Workspace header ----------
st.markdown(
    f"""
    <div class="active-doc">
        <div class="active-doc-title">📄 {document_name}</div>
        <div class="active-doc-meta">
            Active document · Search, summarize, extract and chat with your document
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------- Metrics ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Chunks", len(st.session_state.chunks or []))

with col2:
    if st.session_state.embeddings is not None:
        dimension = st.session_state.embeddings.shape[1]
    elif st.session_state.index is not None:
        dimension = st.session_state.index.d
    else:
        dimension = 0
    st.metric("🧠 Embedding", dimension)

with col3:
    st.metric("📊 Tables", len(st.session_state.tables or []))

with col4:
    st.metric("💬 Questions", len(st.session_state.chat_history))

st.write("")

# ---------- Workspace tabs ----------
tab_chat, tab_summary, tab_extract = st.tabs([
    "💬 Ask & Chat",
    "📝 Summarize",
    "📋 Extract Information"
])

# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:
    st.markdown("### Ask your document")
    st.caption("Get grounded answers with page-level evidence from the active document.")

    # Suggested questions make the empty state feel like a real product.
    suggested_questions = [
        "What are the main points of this document?",
        "What are the key findings or conclusions?",
        "Summarize the most important information.",
        "What data or tables are most relevant?"
    ]

    if not st.session_state.chat_history:
        st.markdown("**✨ Try a suggested question**")
        qcols = st.columns(2)
        for i, suggestion in enumerate(suggested_questions):
            with qcols[i % 2]:
                if st.button(
                    suggestion,
                    key=f"suggested_single_{i}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(chat["question"])

        with st.chat_message("assistant", avatar="🧠"):
            st.write(chat["answer"])
            if "grounding" in chat:
                render_grounding(chat["grounding"])
            render_sources(chat["sources"])

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask something about the document...")
    active_question = question or pending

    if active_question:
        with st.chat_message("user", avatar="👤"):
            st.write(active_question)

        with st.chat_message("assistant", avatar="🧠"):
            try:
                with st.spinner("🔎 Searching document..."):
                    results = search_document(
                        active_question,
                        st.session_state.index,
                        st.session_state.bm25_index,
                        st.session_state.chunks,
                        top_k=3
                    )

                for result in results:
                    result["document_id"] = st.session_state.document_id
                    result["document_name"] = st.session_state.file_name

                with st.spinner("🤖 Generating grounded answer..."):
                    answer = generate_answer(active_question, results)

                st.write(answer)

                grounding = calculate_grounding_score(
                    active_question,
                    answer,
                    results
                )

                render_grounding(grounding)
                render_sources(results)

                st.session_state.chat_history.append({
                    "question": active_question,
                    "answer": answer,
                    "sources": results,
                    "grounding": grounding
                })

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")

# ============================================================
# SUMMARY TAB
# ============================================================

with tab_summary:
    st.markdown("### 📝 Document Summary")
    st.caption("Generate a concise AI summary from the processed document chunks.")

    if st.button(
        "✨ Generate Summary",
        use_container_width=True,
        key="generate_summary_professional"
    ):
        with st.spinner("📝 Creating summary..."):
            try:
                st.session_state.summary = summarize_document(
                    st.session_state.chunks
                )
            except Exception as e:
                st.error(f"❌ Summary failed: {e}")

    if st.session_state.summary:
        st.markdown(
            "<div class='section-card'>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No summary generated yet. Click **Generate Summary** to analyze this document.")

# ============================================================
# EXTRACTION TAB
# ============================================================

with tab_extract:
    st.markdown("### 📋 Structured Information")
    st.caption("Extract useful fields and structured information using the document context.")

    if st.button(
        "✨ Extract Information",
        use_container_width=True,
        key="extract_information_professional"
    ):
        with st.spinner("📋 Extracting information..."):
            try:
                st.session_state.extracted = extract_information(
                    st.session_state.chunks,
                    st.session_state.index,
                    generate_embeddings,
                    search_faiss
                )
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")

    if st.session_state.extracted:
        st.markdown(
            "<div class='section-card'>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.extracted)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No structured information extracted yet. Click **Extract Information** to begin.")

# ============================================================
# EVIDENCE CENTER
# ============================================================

if st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]
    sources = latest.get("sources", [])
    grounding = latest.get("grounding")

    st.divider()
    st.markdown("### 🔎 Evidence Center")
    st.caption("The latest answer is grounded in the following retrieved document evidence.")

    evidence_cols = st.columns([1, 2])

    with evidence_cols[0]:
        if grounding:
            score = grounding.get("score", 0)
            st.metric("🎯 Grounding", f"{score}%")
            st.progress(max(0, min(100, int(score))))
            st.caption(grounding.get("explanation", ""))
        st.markdown(
            f"**{len(sources)}** evidence passages retrieved"
        )

    with evidence_cols[1]:
        for i, source in enumerate(sources[:4]):
            name = source.get("document_name", document_name)
            page = source.get("page", "Unknown")
            score = source.get("hybrid_score", 0)
            text = source.get("text", "")
            preview = text[:260].replace("\n", " ")
            if len(text) > 260:
                preview += "..."

            st.markdown(
                f"""
                <div class="section-card" style="margin-bottom:0.55rem;">
                    <div style="font-weight:700;color:#eef1ff;">📄 {name}</div>
                    <div style="color:#8e9ab0;font-size:0.76rem;margin:0.2rem 0 0.45rem;">
                        Page {page} · Hybrid relevance {score:.3f}
                    </div>
                    <div style="color:#c1c9d8;font-size:0.84rem;line-height:1.45;">
                        {preview}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
