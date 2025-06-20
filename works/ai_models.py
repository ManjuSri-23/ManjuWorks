# prompt='''
# tell me about 2004 tsunami at chennai in four lines
# '''

# import google.generativeai as genai
# genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")

# model=genai.GenerativeModel(model_name="gemini-2.0-flash")
# response=model.generate_content(prompt)
# result=response.text
# print(f"response from gemini : {result}")


# from groq import Groq
# client = Groq(api_key="gsk_ajYwkhUYXc6ieObWbOWGdyb3FYQIVNl1xPzMQrt4MvIRqjrqru")  
# chat_completion = client.chat.completions.create(
#     model="llama3-70b-8192",  
#     messages=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
   
# )

# result1=chat_completion.choices[0].message.content
# print(f"response from groq :{result1}")


# prompt='''
# "Explain black holes in simple words. return the answer in json format'''

# import google.generativeai as genai
# genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")

# model=genai.GenerativeModel(model_name="gemini-2.0-flash")
# response=model.generate_content(prompt)
# result=response.text
# print(f"response from gemini : {result}")


# from groq import Groq
# client = Groq(api_key="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")  
# chat_completion = client.chat.completions.create(
#     model="llama3-70b-8192",  
#     messages=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
   
# )

# result1=chat_completion.choices[0].message.content
# print(f"response from groq :{result1}")





# from pydantic import BaseModel
# import google.generativeai as genai
# from groq import Groq

# class LLMResponse(BaseModel):
#     provider: str
#     content: str


# genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")

# gemini_model =genai.GenerativeModel(model_name="gemini-2.0-flash")

# groq_client = Groq(api_key="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")

# prompt = 'explain what is coment in 4 lines'

# gemini_response = gemini_model.generate_content(prompt)
# gemini_text = gemini_response.text
# gemini_structured = LLMResponse(provider="Gemini", content=gemini_text)


# groq_response = groq_client.chat.completions.create(
#     model="llama3-70b-8192",
#     messages=[{"role": "user", "content": prompt}]
# )
# groq_text = groq_response.choices[0].message.content
# groq_structured = LLMResponse(provider="Groq (LLaMA3)", content=groq_text)


# for response in [gemini_structured, groq_structured]:
#     print(f" {response.provider} Response:{response.content}")


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