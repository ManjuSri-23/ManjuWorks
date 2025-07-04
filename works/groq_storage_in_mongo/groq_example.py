
from groq import Groq
import json
from pymongo import MongoClient


client = Groq(api_key="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")  


chat_completion = client.chat.completions.create(
    model="llama3-70b-8192",  
    messages=[
        {
            "role": "user",
            "content": "crack me a joke  "
        }
    ]
   
)

result={"string":chat_completion.choices[0].message.content}

client=MongoClient('mongodb+srv://manjusri2306:limanju@cluster0.alpj0mm.mongodb.net/')
db=client['database']
collection=db['groq']
inserted_doc=collection.insert_one(result)
print(inserted_doc.inserted_id)
client.close()

