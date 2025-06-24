import streamlit as st
from app import (
    extract_text_from_pdf, split_text, retrieve_with_cosine_similarity,
    generate_response, listen_with_deepgram, speak_text,
    embedding_model, gemini_api_key
)

st.set_page_config(
    page_title="🎤 EchoLens - Voice PDF Assistant", layout="centered")

st.title("🎤 EchoLens")
st.markdown(
    "Upload a PDF and then ask questions by speaking. The assistant will listen, understand, and respond!")

# === PDF UPLOAD ===
pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if pdf_file is not None and "pdf_name" not in st.session_state:
    # Save uploaded file
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Extract and embed
    pdf_text = extract_text_from_pdf("temp_uploaded.pdf")
    chunks = split_text(pdf_text)
    with st.spinner("📚 Generating embeddings..."):
        document_embeddings = embedding_model.encode(
            chunks, show_progress_bar=True)

    # Store in session
    st.session_state["pdf_name"] = pdf_file.name
    st.session_state["chunks"] = chunks
    st.session_state["document_embeddings"] = document_embeddings
    st.success("✅ PDF loaded and embeddings ready!")

# === Once PDF is loaded ===
if "chunks" in st.session_state and "document_embeddings" in st.session_state:
    if "listening" not in st.session_state:
        st.session_state["listening"] = False

    st.markdown("### 🎙️ Voice Control")
    stt_output = st.empty()
    answer_output = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎧 Start Listening", disabled=st.session_state["listening"]):
            st.session_state["listening"] = True

    with col2:
        if st.button("🛑 Stop Listening", disabled=not st.session_state["listening"]):
            st.session_state["listening"] = False

    if st.session_state["listening"]:
        stt_output.info("🎤 Listening...")
        query = listen_with_deepgram()
        if query.strip():
            stt_output.success(f"📝 You said: `{query}`")

            if query.strip().lower() in ["exit", "quit"]:
                st.stop()

            relevant_chunks = retrieve_with_cosine_similarity(
                query, embedding_model, st.session_state["document_embeddings"], st.session_state["chunks"])
            context = "\n".join(relevant_chunks)

            prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information above I want you to think step by step to answer the query in a crisp manner. In case you don't know the answer, say 'I don't know!'.
Query: {query}
Answer: """

            try:
                response = generate_response(prompt, gemini_api_key)
                answer = response['candidates'][0]['content']['parts'][0]['text']
                answer_output.success(f"🤖 Bot: {answer}")
                speak_text(answer)
            except Exception as e:
                answer_output.error("❌ Error generating response.")
                speak_text("Sorry, I ran into an error.")
        else:
            stt_output.warning("⚠️ Nothing was transcribed. Try again.")
