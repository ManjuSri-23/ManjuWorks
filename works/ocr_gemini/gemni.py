from pydantic import BaseModel
from datetime import date
import re
import google.generativeai as genai
from PIL import Image

class User(BaseModel):
    name:str
    dob:str
    gender:str


img=Image.open(r"C:\Users\MANJUSREE\OneDrive\Pictures\Screenshots\Screenshot 2025-06-13 114110.png")
genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")

model=genai.GenerativeModel(model_name="gemini-2.0-flash",generation_config={
    'response_mime_type':"application/json",
    'response_schema':User
})
prompt='''
from the image extract the  information  such as name of the candidate , date of birth  , gender and return all the three in format specified.
 
'''

response=model.generate_content([img,prompt])
result=response.text
print(result)

# import google.generativeai as genai
# from PIL import Image
# from pydantic import BaseModel

# genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")
# img=Image.open(r"C:\Users\MANJUSREE\OneDrive\Pictures\Screenshots\Screenshot 2024-11-24 164530.png")
# class example(BaseModel):
#     text_content:str
# prompt='''
# from the image extract the text content and return in the format specified
# '''
# model=genai.GenerativeModel(model_name="gemini-2.0-flash",generation_config={"response_mime_type":"application/json",
#                                                                             "response_schema":example} )
# response=model.generate_content([img,prompt])
# result=response.text
# print(result)

# name = re.search(r'name:\s*(.+)', result).group(1).strip()
# dob = re.search(r'dob:\s*(.+)', result).group(1).strip()
# gender = re.search(r'gender:\s*(.+)', result).group(1).strip()

# user = User(name=name, dob=dob, gender=gender)
# print(user)

