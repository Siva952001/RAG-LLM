import chromadb
import pandas as pd 
import streamlit as st

pd.set_option('display.max_columns', None)  
def view_collections(db_path):
    st.markdown("### ChromaDB Path: %s" % db_path)

    try:
        
        client = chromadb.PersistentClient(path=db_path)
        
        collections = client.list_collections()
        if not collections:
            st.write("No collections found in the database.")
            return

        # st.header("Collections")

        for collection in collections:
            st.markdown("### Collection Name: **%s**" % collection.name)

            
            data = collection.get()
            
            
            df = pd.DataFrame({
                'ids': data.get('ids', []),
                'embeddings': data.get('embeddings', [""] * len(data.get('ids', []))),
                # 'metadata': data.get('metadatas', [""] * len(data.get('ids', []))),
                'documents': data.get('documents', [])
            })

            st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred while accessing the database: {str(e)}")

if __name__ == "__main__":
    
    db_path = st.text_input("Enter the Chroma DB Path:", "")
    
    if db_path:
        view_collections(db_path)
    else:
        st.write("Please provide a valid database path.")
