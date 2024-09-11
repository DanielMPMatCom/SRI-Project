import streamlit as st



def main():
    st.markdown("# hello there")
    st.sidebar.info("This is a sidebar")

if __name__ == '__main__':
    st.set_page_config(page_title="Informe", page_icon=":bar_chart:", layout="wide")
    main()
