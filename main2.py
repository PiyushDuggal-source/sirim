import os
import base64
import wave
import numpy as np
import pyaudio
import whisper
import concurrent.futures
from PIL import ImageGrab
from ollama import chat
from modules.MeloTTS.melo.api import TTS


class OptimizedChatBot:

    def __init__(
        self,
        tts_device: str = "mps",
        ollama_model: str = "llama3.2",
        sample_rate: str = 16000,
        whisper_model="base.en",
        silence_threshold=500,
        silence_duration=2,
    ):

        self.tts_device = tts_device
        self.whisper_model = whisper.load_model(whisper_model)
        self.ollama_model = ollama_model
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        # preload model

        self.tts_model = TTS(language="EN", device=self.tts_device)
        self.speaker_ids = self.tts_model.hps.data.spk2id

    def synthesize(self, text, speed=1.2):
        output_path = "output.wav"
        self.tts_model.tts_to_file(
            text, self.speaker_ids["EN-US"], output_path, speed=speed, format="wav"
        )

        return output_path

    def record_audio(self, output_path):

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
        )

        frames = []
        silence_frames = 0
        total_frames = 0

        print("Recording... Speak now.")
        try:
            while True:
                data = stream.read(1024)
                audio_data = np.frombuffer(data, dtype=np.int16)

                if np.abs(audio_data).max() < self.silence_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                frames.append(data)
                total_frames += 1

                if silence_frames > (self.silence_duration * self.sample_rate / 1024):
                    print("\nSilence detected. Stopping recording.")
                    break

        except KeyboardInterrupt:
            print("\nRecording interrupted by user.")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(frames))

    def transcribe_audio(self, file_path):
        try:
            print("Transcribing audio...")
            result = self.whisper_model.transcribe(file_path)
            print("Transcription completed.")
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def get_bot_response(self, conversation_history):
        try:
            resp = chat(
                model=self.ollama_model, messages=conversation_history, keep_alive=True
            )
            return resp.message.content
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I'm sorry, I couldn't generate a response."

    def play_audio(self, file_path):
        try:
            with wave.open(file_path, "rb") as wf:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)

                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            print(f"Audio playback error: {e}")

    def run_conversation(self):
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

            # Use concurrent execution for recording and potential screenshot
            with concurrent.futures.ThreadPoolExecutor() as executor:
                record_future = executor.submit(self.record_audio, audio_file)
                record_future.result()  # Wait for recording to complete

                transcription = self.transcribe_audio(audio_file)
                os.remove(audio_file)

            print("User:", transcription)

            if transcription.lower() == "exit.":
                break

            is_image = (
                "screen" in transcription.lower() or "image" in transcription.lower()
            )

            if is_image:
                screenshot = ImageGrab.grab()
                screenshot_file = "screenshot.png"
                screenshot.save(screenshot_file)
                with open(screenshot_file, "rb") as image_file:
                    base64_string = base64.b64encode(image_file.read()).decode("utf-8")
                chat_history.append(
                    {
                        "role": "user",
                        "content": transcription,
                        "images": [base64_string],
                    }
                )

                print("Screenshot saved to screenshot.png")
            else:
                chat_history.append({"role": "user", "content": transcription})

            response = self.get_bot_response(chat_history)
            chat_history.append({"role": "assistant", "content": response})

            print("ChatBot:", response)

            bot_audio_file = self.synthesize(response)
            self.play_audio(bot_audio_file)

        return chat_history


chatbot = OptimizedChatBot()
chatbot.run_conversation()
