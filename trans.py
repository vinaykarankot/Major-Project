from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from googletrans import Translator
import re

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

def summarize_text(text, model_name='facebook/bart-large-cnn', chunk_size=1000):
    summarizer = pipeline("summarization", model=model_name, device=0)  # Use GPU if available
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def summarize_chunk(chunk):
        input_length = len(chunk.split())
        max_length = min(200, int(input_length * 0.5))
        summary = summarizer(chunk, max_length=max_length, min_length=int(max_length * 0.5), do_sample=False)
        return summary[0]['summary_text']
    
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_chunk, chunks))

    return " ".join(summaries)

def clean_translation(text):
    # Basic text normalization (add more rules as needed)
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    text = text.replace(' ,', ',').replace(' .', '.')  # Fix spacing around punctuation
    return text.strip()

def translate_summary(summary, target_language='te'):
    translator = Translator()
    translation = translator.translate(summary, dest=target_language).text
    translation = clean_translation(translation)
    return translation

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=__EoOvVkEMo"  # Replace with your video URL
    video_id = url.split("v=")[-1]
    transcript = get_transcript(video_id)
    
    if transcript:
        print("Transcript:")
        print(transcript)
        
        summary = summarize_text(transcript)
        print("\nSummary:")
        print(summary)
        
        translated_summary = translate_summary(summary, target_language='te')  # Change 'te' to any other language code as needed
        print("\nTranslated Summary:")
        print(translated_summary)
    else:
        print("Transcript could not be retrieved.")
