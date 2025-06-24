import streamlit as st
import base64
from app import (
    extract_text_from_pdf, split_text, retrieve_with_cosine_similarity,
    generate_response, listen_with_deepgram, speak_text,
    embedding_model, gemini_api_key
)

st.set_page_config(page_title="EchoLens - Voice PDF Assistant", layout="wide")

st.title("EchoLens")
st.markdown(
    "_Your AI study partner. Upload a PDF and ask questions by speaking._")

# === Create a 2-column layout, 60% left for controls, 40% right for PDF ===
left_col, right_col = st.columns([3, 2])

with left_col:
    # Upload PDF
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # Show progress, process PDF, voice assistant UI
    if pdf_file is not None and "pdf_name" not in st.session_state:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(pdf_file.read())

        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf("temp_uploaded.pdf")

        with st.spinner("Splitting into chunks..."):
            chunks = split_text(pdf_text)

        with st.spinner("Generating embeddings..."):
            document_embeddings = embedding_model.encode(
                chunks, show_progress_bar=True)

        st.session_state["pdf_name"] = pdf_file.name
        st.session_state["chunks"] = chunks
        st.session_state["document_embeddings"] = document_embeddings
        st.success("PDF processed and ready!")

    if "chunks" in st.session_state and "document_embeddings" in st.session_state:
        if "listening" not in st.session_state:
            st.session_state["listening"] = False

        st.markdown("### Voice Assistant")
        stt_output = st.empty()
        answer_output = st.empty()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Listening", disabled=st.session_state["listening"]):
                st.session_state["listening"] = True

        with col2:
            if st.button("Mute Microphone", disabled=not st.session_state["listening"]):
                st.session_state["listening"] = False

        with col3:
            if st.button("Reset"):
                st.session_state.clear()
                st.experimental_rerun()

        if st.session_state["listening"]:
            stt_output.info("🎤 Listening...")
            query = listen_with_deepgram()
            if query.strip():
                stt_output.success(f"You said: `{query}`")

                if query.strip().lower() in ["exit", "quit"]:
                    st.stop()

                relevant_chunks = retrieve_with_cosine_similarity(
                    query, embedding_model, st.session_state["document_embeddings"], st.session_state["chunks"]
                )
                context = "\n".join(relevant_chunks)

                prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information above, think step-by-step to answer the query concisely. If the answer is unknown, say "I don't know!".
Query: {query}
Answer:"""

                try:
                    response = generate_response(prompt, gemini_api_key)
                    answer = response['candidates'][0]['content']['parts'][0]['text']
                    answer_output.success(f"Bot: {answer}")
                    speak_text(answer)
                except Exception:
                    answer_output.error("Error generating response.")
                    speak_text("Sorry, I ran into an error.")
            else:
                stt_output.warning("Nothing was transcribed. Try again.")

with right_col:
    # Show the uploaded PDF in an iframe fixed height and width
    if pdf_file is not None:
        pdf_bytes = pdf_file.getvalue()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="700px"
            style="border: 1px solid #555;"
        ></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
