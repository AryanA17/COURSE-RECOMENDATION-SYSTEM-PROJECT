import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os  # For file handling

# Correct dataset file path
DATASET_PATH = r"C:\Users\Aryan\Desktop\Project\course_dataset.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        return df
    else:
        st.error(f"⚠️ Dataset file not found at '{DATASET_PATH}'. Please check the file location!")
        return None

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

@st.cache_data
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    try:
        course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
        idx = course_indices[title]

        sim_scores = list(enumerate(cosine_sim_mat[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_of_rec+1]

        selected_course_indices = [i[0] for i in sim_scores]
        selected_course_scores = [i[1] for i in sim_scores]

        result_df = df.iloc[selected_course_indices].copy()
        result_df['similarity_score'] = selected_course_scores
        return result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers', 'content_duration']]
    
    except KeyError:
        return None

@st.cache_data
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False, na=False)]
    return result_df[['course_title', 'url', 'price', 'num_subscribers', 'content_duration']]

def main():
    st.title("📚 Course Recommendation App")

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load dataset from fixed path
    df = load_data()
    if df is None:
        return  # Stop execution if dataset is missing

    if choice == "Home":
        st.subheader("🏠 Home")
        st.write("Displaying a preview of the dataset:")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("📌 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])

        search_term = st.text_input("Enter course title:")
        num_of_rec = st.sidebar.slider("Number of recommendations", 4, 20, 7)

        if st.button("Get Recommendations"):
            if search_term:
                results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                if results is not None:
                    st.success(f"🎯 Top {num_of_rec} recommendations for '{search_term}':")
                    for _, row in results.iterrows():
                        st.subheader(f"📌 {row['course_title']}")
                        st.write(f"**⏳ Duration:** {row['content_duration']} hours")
                        st.write(f"**🔢 Similarity Score:** {row['similarity_score']:.2f}")
                        st.write(f"💲 **Price:** {row['price']} | 🧑‍🎓 **Students:** {row['num_subscribers']}")
                        st.markdown(f"[🔗 View Course]({row['url']})", unsafe_allow_html=True)
                        st.divider()

                else:
                    st.warning(f"⚠️ No exact match found for '{search_term}'. Showing similar courses:")
                    similar_results = search_term_if_not_found(search_term, df)

                    if not similar_results.empty:
                        for _, row in similar_results.iterrows():
                            st.subheader(f"📌 {row['course_title']}")
                            st.write(f"**⏳ Duration:** {row['content_duration']} hours")
                            st.write(f"₹ **Price:** {row['price']} | 🧑‍🎓 **Students:** {row['num_subscribers']}")
                            st.markdown(f"[🔗 View Course]({row['url']})", unsafe_allow_html=True)
                            st.divider()
                    else:
                        st.error("No similar courses found!")

            else:
                st.error("Please enter a course title to search!")

    else:
        st.subheader("ℹ️ About")
        st.write("Built using **Streamlit** & **Pandas** for course recommendations.")

# Run the app
if __name__ == '__main__':
    main()
