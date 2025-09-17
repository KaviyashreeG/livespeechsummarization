import nltk
from speech_recognition import Recognizer, Microphone
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt_tab')  # Even though this isn't commonly required


def transcribe_speech():
    recognizer = Recognizer()
    mic = Microphone()

    with mic as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise for 1 second
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print("Error:", e)
        return ""

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stop_words]
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(5)]  # Return top 5 most frequent words

def summarize_text(text):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:20])  # Summarize by taking the first 20 sentences

def main():
    while True:
        speech_text = transcribe_speech()
        if speech_text:
            print("Transcribed text:", speech_text)
            keywords = extract_keywords(speech_text)
            print("Keywords:", keywords)
            summary = summarize_text(speech_text)
            print("Summary:", summary)

if __name__ == "__main__":
    main()
