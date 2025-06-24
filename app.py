import os
import sys
import wave
import sounddevice as sd
import requests
import PyPDF2
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deepgram import Deepgram

load_dotenv()
deepgram_api_key = os.getenv("deepgram_api_key")
cartesia_api_key = os.getenv("cartesia_api_key")
gemini_api_key = os.getenv("gemini_api_key")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# === TEXT TO SPEECH (CARTESIA) ===


def speak_text(text):
    try:
        print("🗣️ Synthesizing speech with Cartesia...")

        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "Authorization": f"Bearer {os.getenv('cartesia_api_key')}",
            "Cartesia-Version": "2025-04-16",
            "Content-Type": "application/json"
        }

        payload = {
            "transcript": text,
            "model_id": "sonic-2",
            "voice": {
                "mode": "id",
                # replace if you use a different voice
                "id": "694f9389-aac1-45b6-b726-9d9369183238"
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100
            }
        }

        response = requests.post(url, headers=headers,
                                 json=payload, stream=True)
        response.raise_for_status()

        with open("response.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        audio = AudioSegment.from_file("response.wav", format="wav")
        play(audio)
        os.remove("response.wav")

    except Exception as e:
        print(f"❌ TTS Error: {e}")

# === SPEECH TO TEXT (DEEPGRAM) ===


def listen_with_deepgram(duration=5, fs=16000):
    print("🎙️ Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs,
                   channels=1, dtype='int16')
    sd.wait()

    filename = "recording.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

    print("🔁 Transcribing via Deepgram...")
    with open(filename, 'rb') as audio_file:
        dg = Deepgram(deepgram_api_key)
        response = dg.transcription.sync_prerecorded(
            {'buffer': audio_file, 'mimetype': 'audio/wav'})
        transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
        print(f"📝 You said: {transcription}")
        return transcription

# === PDF & EMBEDDINGS ===


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def retrieve_with_cosine_similarity(query, model, document_embeddings, chunks, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def generate_response(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={
        api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


# === MAIN LOOP ===

def main(pdf_path):
    print(f"📄 Extracting and embedding content from: {pdf_path}")
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(pdf_text)
    document_embeddings = embedding_model.encode(
        chunks, show_progress_bar=True)

    print("✅ Ready to go! Ask a question by speaking.")

    while True:
        print("\n🎤 Speak now (say 'exit' to quit)...")
        query = listen_with_deepgram()

        if not query.strip():
            print("⚠️ No speech detected. Try again.")
            continue

        if query.strip().lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        relevant_chunks = retrieve_with_cosine_similarity(
            query, embedding_model, document_embeddings, chunks)
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
            print(f"\n🤖 Bot: {answer}")
            speak_text(answer)
        except Exception as e:
            print("❌ Error during response:", e)
            speak_text("Sorry, I ran into an error.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a PDF path.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
