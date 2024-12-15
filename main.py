from modules.MeloTTS.melo.api import TTS, os
from PIL import ImageGrab
from ollama import chat
import whisper
import pyaudio
import wave
import numpy as np

speed = 1.0

device = "cpu"


def synthesize(text, speed):
    output_path = f"audio.wav"
    model = TTS(language="EN", device=device)
    speaker_ids = model.hps.data.spk2id
    model.tts_to_file(
        text, speaker_ids["EN-US"], output_path, speed=speed, format="wav"
    )

    return output_path


def get_bot_response(conversation_history, image=False):

    if image:
        resp = chat(
            model="llama3.2-vision",
            messages=conversation_history,
            keep_alive=True,
        )

        return resp.message.content

    resp = chat(model="llava", messages=conversation_history, keep_alive=True)

    return resp.message.content


def play_audio(file_path):
    """
    Plays an audio file using the pyaudio library.

    Parameters:
        file_path (str): The path to the audio file (.wav format) to be played.
    """
    try:
        # Open the audio file
        wf = wave.open(file_path, "rb")

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open a stream
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        # Read and play audio in chunks
        chunk = 1024
        data = wf.readframes(chunk)
        print("Playing audio...")
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Terminate PyAudio
        p.terminate()
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
audio_file = "audio.wav"  # Replace with the actual file path
# play_audio(audio_file)


def record_with_silence_detection(
    output_filename,
    silence_threshold=500,
    silence_duration=2,
    sample_rate=44100,
    channels=1,
    format=pyaudio.paInt16,
    chunk=1024,
):
    """
    Record audio with silence detection.

    Parameters:
    - output_filename: Name of the output WAV file
    - silence_threshold: Maximum amplitude considered as silence
    - silence_duration: Seconds of silence to end recording
    - sample_rate: Audio sample rate (default 44.1kHz)
    - channels: Number of audio channels (default mono)
    - format: PyAudio audio format
    - chunk: Number of frames per buffer

    Returns:
    - Duration of the recorded audio
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Recording... Speak now.")

    # Lists to store audio frames and silence tracking
    frames = []
    silence_frames = 0
    total_frames = 0

    try:
        while True:
            # Read audio data
            data = stream.read(chunk)

            # Convert to numpy array for amplitude calculation
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if current chunk is silence
            if np.abs(audio_data).max() < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            # Store frames
            frames.append(data)
            total_frames += 1

            # Check for silence duration
            if silence_frames > (silence_duration * sample_rate / chunk):
                print("\nSilence detected. Stopping recording.")
                break

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded audio
        wf = wave.open(output_filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

        # Calculate and return recording duration
        # recording_duration = total_frames * chunk / sample_rate
        # print(f"Recorded {recording_duration:.2f} seconds to {output_filename}")

        # return recording_duration


def transcribe_audio(file_path):
    """
    Transcribes audio using OpenAI Whisper.

    Parameters:
        file_path (str): Path to the audio file.
    """
    try:
        print("Transcribing audio...")
        model = whisper.load_model(
            "base.en"
        )  # Choose "base", "small", "medium", or "large"
        result = model.transcribe(file_path)
        print("Transcription completed.")
        return result["text"]
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None


def user_chatbot_conversation():
    chat_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant. ANSWER ALL THE QUESTIONS IN SHORT",
        }
    ]

    i = 0

    while True:
        i += 1
        audio_file = "temp_audio.wav"
        record_with_silence_detection(audio_file)
        transcription = transcribe_audio(audio_file)
        print("User:", transcription)
        os.remove(audio_file)
        user_input = transcription
        if user_input.lower() == "exit.":
            break

        isImage = False

        if "screen" in user_input.lower() or "image" in user_input.lower():
            isImage = True
            screenshot = ImageGrab.grab()
            screenshot.save(f"{i}screenshot.png")
            print("Screenshot saved to screenshot.png")
            chat_history.append(
                {
                    "role": "user",
                    "content": user_input,
                    "images": [f"./{i}screenshot.png"],
                }
            )

        else:
            chat_history.append({"role": "user", "content": user_input})
        response = (
            get_bot_response(chat_history, isImage)
            if isImage
            else get_bot_response(chat_history)
        )

        chat_history.append({"role": "assistant", "content": response})
        print("ChatBot:", response)
        bot_audio_file = synthesize(response, 1.2)
        play_audio(bot_audio_file)

    return chat_history


user_chatbot_conversation()
