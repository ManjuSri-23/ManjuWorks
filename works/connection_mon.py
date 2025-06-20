from pymongo import MongoClient
client=MongoClient('mongodb+srv://manjusri2306:thakalimanju@cluster0.alpj0mm.mongodb.net/')
db=client['mydatabase']
collection=db['mycollection']
document={'name':'manju','city':'banglore'}
inserted_document=collection.insert_one(document)
print(inserted_document.inserted_id)
client.close()