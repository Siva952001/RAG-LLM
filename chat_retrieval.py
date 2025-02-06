import streamlit as st
from database import get_collection1,get_collection2,get_collection3,get_database,get_mongo_client

# Mongo setup
client = get_mongo_client()
db = get_database(client)
collection = get_collection1(db)
credentials_collection = get_collection2(db)
knowledge_base = get_collection3(db)

# Get Knowledge basis from db    
def get_categories_from_db():
    categories = [doc['category_name'] for doc in db['knowledge_base'].find()]
    return categories



def load_chat_retrieval():
        if 'is_admin' not in st.session_state or not st.session_state['is_admin']:
            st.write("🔐  You do not have permission to retrieve chat history. Only admins can access this feature.")
            return
        
        def recheck_admin_status():
            if not st.session_state.get('is_admin', False):
                st.write("🔐  Admin permissions required.")
                return False
            return True
        
        if not recheck_admin_status():
            return
    
        users = list(credentials_collection.find({}))
        usernames = [user["username"] for user in users]
        
        usernames.insert(0, "Click here to view users")
        selected_user = st.sidebar.selectbox("Select User", options=usernames, index=0)
        
        categories = get_categories_from_db()
        categories.insert(0, "Click here to view categories")
        
        selected_category = st.sidebar.selectbox("Select Category", options=categories, index=0)

        if selected_user and selected_category:
            start_date = st.sidebar.date_input("📅 Start Date", min_value=None)
            end_date = st.sidebar.date_input("📅 End Date", min_value=start_date)

            def retrieve_data_by_user_and_date_and_category(user,start_date,end_date, category):
                start_date_str = start_date.strftime('%d-%m-%Y')
                end_date_str = end_date.strftime('%d-%m-%Y')
                
                query = {"username": user, "timestamp": {"$gte": start_date_str, "$lte": end_date_str}, "category": category}
                return list(collection.find(query))

            if st.sidebar.button("History", key="submit_button", use_container_width=True):
                if start_date and end_date:
                    
                    data = retrieve_data_by_user_and_date_and_category(selected_user,start_date,end_date, selected_category)

                    if data:
                        st.subheader(f"Chats of {selected_user} in {selected_category} from {start_date} to {end_date}")
                        for record in data:
                            st.write(f"**💬 User Query:** {record['user_query']}")
                            st.write(f"**🤖 Assistant Response:** {record['assistant_response']}")
                            # st.write(f"**🕒 Timestamp:** {record['timestamp']}")
                            st.write(f"**⚙️ Model:** {record['model_name']}")
                            st.write("---")
                    else:
                        st.write(f"No records found for {selected_user} in {selected_category} from {start_date} to {end_date}")
                else:
                    st.write("Please select a date.")
    
        