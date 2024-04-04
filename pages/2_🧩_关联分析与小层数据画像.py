import subprocess
from utils import *

import pandas as pd
import streamlit as st
import os
st.set_page_config(
    page_title="关联分析",
    page_icon="📊",
    layout="wide"
)
# custom_css = """
# <style>
# .stApp {
#     margin-top: -55px; /* 负数值用于减少顶部空间 */
# }
# </style>
# """
logo()
# 插入自定义 CSS
# st.markdown(custom_css, unsafe_allow_html=True)
st.write("#### 这里是关联分析与小层数据画像页面 👋")



data_folder = "./关联分析表"
# 存储读取的所有表格数据
tables = {}
st.write("##### 1. 关联分析结果：")
for file in os.listdir(data_folder):
    if file.endswith(".xlsx"):
        file_path = os.path.join(data_folder, file)
        table_name = os.path.splitext(file)[0]  # 使用文件名作为表格名
        tables[table_name] = pd.read_excel(file_path)
# 让用户选择要显示的表格
selected_tables = st.selectbox('选择要显示的表格：', list(tables.keys()),index=None)
# 根据用户的选择，展示对应的表格
# for table_name in selected_tables:
c1, c2= st.columns([1, 1])
if selected_tables == "地质参数参数关联分析结果":
    with c1:
        st.write(f'#### {selected_tables}')
        st.dataframe(tables[selected_tables])
    with c2:
        st.write("")
        st.write("")
        st.write("")

        st.image('./关联分析表/地质/小层参数分布情况.png', caption='小层参数分布情况', width=500)
a1, a2= st.columns([1.2, 1.5])
if selected_tables == "小层测井参数分布集中区间":
    with a1:
        st.write(f'#### {selected_tables}')
        st.dataframe(tables[selected_tables])
    with a2:
        st.write("")
        st.write("")
        st.write("")
        st.image('./关联分析表/测井/参数分布统计.png', caption='参数分布统计', width=500)
b1, b2= st.columns([1, 1.2])
if selected_tables == "测井参数含油性关系分析":
    with b1:
        st.write(f'#### {selected_tables}')
        st.dataframe(tables[selected_tables])
    with b2:
        st.write("")
        st.write("")
        st.write("")
        st.image('./关联分析表/测井/关联分析结果.png', caption='关联分析结果', width=500)
# data = st.selectbox(
#     label='选择数据集',
#     options=("测井数据","地质数据","生产数据"),
#     index=None,
#     format_func=str,
#     help='选择您训练模型准备使用的数据集'
# )
e1, e2, e3 = st.columns([1.1, 1, 2])
with e1:
    st.write("##### 2. 聚类分析：")
# with e2:
    # julei_button = st.button("生成聚类结果", key="julei_button")
# if julei_button:
d1, d2, d3 = st.columns([1.1, 1, 1.2])
with d1:
    st.image('./聚类分析结果/1.聚类结果图.png', caption='聚类结果图', width=320)
with d2:
    st.image('./聚类分析结果/2.各类小层数据均值分析.png', caption='各类小层数据均值分析', width=320)
with d3:
    st.write("")
    st.write("")
    st.image('./聚类分析结果/3.聚类结果分析图.png', caption='聚类结果分析图', width=320)
f1, f2, f3 = st.columns([1.1, 1, 2])
with f1:
    st.write("##### 3. 小层数据画像：")
# with f2:
#     huaxiang_button = st.button("生成潜力小层数据画像", key="huaxiang_button")
# if huaxiang_button:
st.image('./聚类分析结果/小层数据画像1.png', caption='小层数据画像1', width=800)
st.image('./聚类分析结果/小层数据画像2.png', caption='小层数据画像2', width=800)
