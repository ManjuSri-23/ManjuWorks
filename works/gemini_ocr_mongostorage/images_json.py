
import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel
import json
from pymongo import MongoClient

genai.configure(api_key="AIzaSyAYjDDkAN9CmQKitUBhtfIrg8Amk1C1mMc")
img=Image.open(r"C:\Users\MANJUSREE\OneDrive\Pictures\Screenshots\Screenshot 2024-10-25 173243.png")


class example(BaseModel):
    parent:str
    technology:str
    
prompt='''
from the image extract the group all the parent sent and group all  Technology audience received and return it in specified format
'''
model=genai.GenerativeModel(model_name="gemini-2.0-flash",generation_config={"response_mime_type":"application/json",
                                                                            "response_schema":example})
                            
response=model.generate_content([img,prompt])
result=response.text
print(result)

client=MongoClient('mongodb+srv://manjusri2306:thakalimanju@cluster0.alpj0mm.mongodb.net/')
result_dict = json.loads(result)
db=client['mydatabase']
collection=db['gemni']
prompted={
    "prompt_input":prompt
}

inserted_document=collection.insert_one(result_dict)
inserted_document1=collection.insert_one(prompted)
print(inserted_document.inserted_id)
print(inserted_document1.inserted_id)
client.close()

