from groq import Groq
import logging

class GroqClientWrapper:
    def __init__(self, api_key, model_name, temperature=0.7, max_tokens=8192, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

    def __call__(self, prompt, **kwargs):
        # Merge the default kwargs with any additional kwargs provided
        params = {**self.kwargs, **kwargs}
        logging.debug(f"Calling GroqClientWrapper with params: {params}")
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"]
        )
        return response.choices[0].message.content.strip()