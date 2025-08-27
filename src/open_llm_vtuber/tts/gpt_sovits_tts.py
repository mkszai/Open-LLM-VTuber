####
# change from xTTS.py
####

import re
import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000/infer_single",
        dl_url: str = "http://127.0.0.1:8000",
        version: str = "v4",
        model_name: str = "",
        prompt_text_lang: str = "中文",
        emotion: str = "默认",
        text_lang: str = "中文",
        ref_audio_path: str = "",
        prompt_lang: str = "中文",
        prompt_text: str = "",
        text_split_method: str = "按标点符号切",
        batch_size: int = 10,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        media_type: str = "wav",
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        seed: int = -1,
        sample_steps: int = 16,
        if_sr: bool = False,
        top_k: int = 10,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        self.api_url = api_url
        self.dl_url = dl_url
        self.version = version
        self.model_name = model_name
        self.prompt_text_lang = prompt_text_lang
        self.emotion = emotion
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.batch_threshold = batch_threshold
        self.split_bucket = split_bucket
        self.speed_factor = speed_factor
        self.fragment_interval = fragment_interval
        self.media_type = media_type
        self.parallel_infer = parallel_infer
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.sample_steps = sample_steps
        self.if_sr = if_sr
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)
        
        # Prepare the data for the POST request
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "prompt_text_lang": self.prompt_text_lang,
            "emotion": self.emotion,
            "model_name": self.model_name,
            "version": self.version,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "split_bucket": self.split_bucket,
            "speed_factor": self.speed_factor,
            "fragment_interval": self.fragment_interval,
            "media_type": self.media_type,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "sample_steps": self.sample_steps,
            "if_sr": self.if_sr,
        }

        # Send POST request to the TTS API
        try:
            response = requests.post(f"{self.api_url}", json=data, timeout=120)
            
            # Check if the POST request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Check if synthesis was successful
                if result.get("msg") == "合成成功" and "audio_url" in result:
                    # Extract the audio URL
                    audio_url = result["audio_url"]
                    
                    # Replace 0.0.0.0 or 127.0.0.1 in the audio URL with the actual server IP from dl_url
                    if audio_url.startswith("http://0.0.0.0") or audio_url.startswith("http://127.0.0.1"):
                        # Extract the path and port from the audio_url
                        path = audio_url.split("/", 3)[3] if len(audio_url.split("/")) > 3 else ""
                        # Use the dl_url's host and port but with the path from audio_url
                        audio_url = f"{self.dl_url.rstrip('/')}/{path}"
                    
                    # Download the audio file
                    audio_response = requests.get(audio_url, timeout=120)
                    
                    # Check if the GET request was successful
                    if audio_response.status_code == 200:
                        # Save the audio content to a file
                        with open(file_name, "wb") as audio_file:
                            audio_file.write(audio_response.content)
                        return file_name
                    else:
                        logger.critical(
                            f"Error: Failed to download audio. Status code: {audio_response.status_code}"
                        )
                        return None
                else:
                    return None
            else:
                # Handle errors or unsuccessful requests
                logger.critical(
                    f"Error: Failed to generate audio. Status code: {response.status_code}, Response: {response.text}"
                )
                return None
        except Exception as e:
            logger.critical(f"Error: Failed to generate audio. Exception: {str(e)}")
            return None
