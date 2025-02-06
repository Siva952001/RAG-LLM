from pymongo import MongoClient

def get_mongo_client(uri="mongodb://localhost:27017/"):
    return MongoClient(uri)

def get_database(client, db_name="RAG-STORE"):
    return client[db_name]

def get_collection1(db, collection_name="llama"):
    return db[collection_name]

def get_collection2(db, collection_name="credentials"):
    return db[collection_name]

def get_collection3(db, collection_name = "knowledge_base"):
    return db[collection_name]

# def get_collection_chat_process(db, collection_name="process_files"):
#     return db[collection_name]

def save_chat_to_db(collection, chat_data):
    collection.insert_one(chat_data)
    

