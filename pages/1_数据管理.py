import os
import subprocess
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="数据管理",
    page_icon="graph-up-arrow",
    layout="wide"
)

st.write("#### 这里是数据管理页面 👋")
file_path=[]
def get_file_list(suffix, path):
    input_template_all =[]
    input_template_all_path =[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == suffix:
                input_template_all.append(name)
                input_template_all_path.append(os.path.join(root, name))
    return input_template_all, input_template_all_path

upload_dir = "./data1"
os.makedirs(upload_dir, exist_ok=True)

uploaded_file = st.sidebar.file_uploader("Upload a zip file", type="zip")

def support_gbk(zip_file: ZipFile):
    name_to_info = zip_file.NameToInfo
    # copy map first
    for name, info in name_to_info.copy().items():
        real_name = name.encode('cp437').decode('gbk')
        if real_name != name:
            info.filename = real_name
            del name_to_info[name]
            name_to_info[real_name] = info
    return zip_file



page_options = ["#### 数据查看", "#### 多源数据融合"]
selected_page = st.sidebar.radio(
    "#### 请选择功能",
    page_options,
    index=0
)
if selected_page== "#### 多源数据融合":
    # st.write(file_path)
    st.write("###### 选择建模所用特征参数")
    method_choice = st.radio("选择方法", ["方法一：选择一项默认参数组合", "方法二：自定义参数组合建模"])
    if method_choice == "方法二：自定义参数组合建模":
        search = st.multiselect(
            label='自定义参数组合建模',
            options=('GR', 'AC', 'CNL', 'CAL', 'DEN', 'RA25', 'RA45', 'SP', 'RT'),
            default=None,
            format_func=str,
            help='请仔细检查所选参数组合是否已研究过',
            placeholder='请选择您自定义的参数（可多选）'
        )

        sorting = st.text_input(
            "参数组合命名", placeholder="请为所选参数组合命名"
        )

    elif method_choice == "方法一：选择一项默认参数组合":
        st.write("""
        <span style='font-size: 14.7px;'>选择一项默认参数组合</span>
        """, unsafe_allow_html=True)

        st.checkbox("6RA : [ AC, CNL, DEN, GR, SP, RA25 ]", value=False)
        st.checkbox("6RT : [ AC, CNL, DEN, GR, SP, RT ]", value=False)
        st.checkbox("5RA : [ AC, CNL, GR, SP, RA25 ]", value=False)

    datapro_button = st.button("数据预处理", key = "datapro_button", type="primary")
    if datapro_button:
        # 在按钮被点击时执行后端脚本，并捕获输出内容
        output = subprocess.check_output(['python', 'main2lsf.py'], encoding='gbk')
        # 将输出内容显示在前端页面中的文本框中

        # st.text_area("数据处理过程：", value=output, height=400)
        # 显示进度条
        progress_bar = st.progress(0)
        # 模拟训练过程
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        st.empty()
        # 创建新的列容器
        col, coll, colll = st.columns([3, 0.2, 1.2])
        # 显示成功提示框
        # tsfresh_results = "这里是tsfresh提取的特征结果"
        text = f'所选特征数据集处理完毕！'
        col.success(text)
    else:
        text = f'请选择特征，点击“数据预处理”按钮'
        st.write("###### "+text)
        st.header(" ", divider='rainbow')
    data_folder = "数据预处理后的训练集和测试集"
        # 存储读取的所有表格数据
    tables = {}
    # 获取数据文件夹中的所有CSV文件
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            table_name = os.path.splitext(file)[0]  # 使用文件名作为表格名
            tables[table_name] = pd.read_csv(file_path)
    # 让用户选择要显示的表格
    selected_tables = st.multiselect('选择要显示的表格：', list(tables.keys()))
    # 根据用户的选择，展示对应的表格
    for table_name in selected_tables:
        st.write(f'## {table_name}')
        st.dataframe(tables[table_name])
if selected_page== "#### 数据查看":
    if uploaded_file:
        # 解压文件
        with support_gbk(ZipFile(uploaded_file, "r")) as zip_ref:
            zip_ref.extractall(upload_dir)

        # 获取解压后的文件路径
        extracted_files = os.listdir(upload_dir)
        for extracted_file in extracted_files:
            extracted_file_path = os.path.join(upload_dir, extracted_file)
            file_path.append(extracted_file_path)
        st.write("数据已经成功上传！")
        # 创建两行两列的容器
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # 在第一行第一列中添加选择器和数据展示
        with row1_col1:
            st.header("小层数据")
            selected_file_1 = st.selectbox("选择文件", os.listdir(file_path[0]))
            if selected_file_1:
                df = pd.read_csv(os.path.join(file_path[0], selected_file_1))
                st.dataframe(df)

        # 在第一行第二列中添加选择器和数据展示
        with row1_col2:
            st.header("录井数据")
            selected_file_2 = st.selectbox("选择文件", os.listdir(file_path[1]))
            if selected_file_2:
                df = pd.read_csv(os.path.join(file_path[1], selected_file_2))
                st.dataframe(df)

        # 在第二行第一列中添加选择器和数据展示
        with row2_col1:
            st.header("测井数据")
            selected_file_3 = st.selectbox("选择文件", os.listdir(file_path[2]))
            if selected_file_3:
                df = pd.read_csv(os.path.join(file_path[2], selected_file_3))
                st.dataframe(df)

        # 在第二行第二列中添加选择器和数据展示
        with row2_col2:
            st.header("生产数据")
            selected_file_4 = st.selectbox("选择文件", os.listdir(file_path[3]))
            if selected_file_4:
                df = pd.read_csv(os.path.join(file_path[3], selected_file_4))
                st.dataframe(df)
    else:
        st.write('没有上传文件，请选择文件进行上传')
