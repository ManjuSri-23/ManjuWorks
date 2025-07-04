


from pydantic import BaseModel 
from groq import Groq
import google.generativeai as genai

class user(BaseModel):
        content:str
        
def ai_gemini(prompt):

    genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")

    model=genai.GenerativeModel(model_name="gemini-2.0-flash",generation_config={"response_mime_type":"application/json","response_schema":user})
    response=model.generate_content(prompt)
    result=response.text
    print(f"response from gemini : {result}")

def ai_groq(prompt):
 
    client = Groq(api_key="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")  
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",  
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    )

    result1=chat_completion.choices[0].message.content
    groqtext=user(content=result1)
    print(f"response from groq :{groqtext.content}")

prompt='tell me a astronomy fact'

ai_groq(prompt)
ai_gemini(prompt)