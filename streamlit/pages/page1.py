import streamlit as st


st.title('Solarmente Berlin App')

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.selectbox(
    'Please pick and address',
    ('Email', 'Home phone', 'Mobile phone')
)

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")