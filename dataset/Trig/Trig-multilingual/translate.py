from google import genai
from google.genai import types

system_instruction = """
You are a professional translation assistant.  
- Your task is to provide accurate, fluent, and context-aware translations from English to {target_language}.  
- Preserve the meaning, tone, and style of the original text.  
- The translation should be natural and idiomatic, without omitting any information, and should maintain the original structure.
I will provide you with one sentence each time, and you should return only the translation in {target_language}, without any additional explanations or responses.
"""
target_language = "Chinese"
client = genai.Client(api_key="AIzaSyALD7dr1v_p37295DYo0UhcECCpVCxIqcw")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction.format(target_language=target_language)),
    contents="Hello there"
)

print(response.text)