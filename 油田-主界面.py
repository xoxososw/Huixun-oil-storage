import streamlit as st
from utils import *
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    # layout="wide"
)
logo()
st.write("# 欢迎来到慧寻油储! 👋")
st.image('./images/系统框架.png', caption='慧寻油储系统设计框架')
st.sidebar.success("请选择一个模块.")