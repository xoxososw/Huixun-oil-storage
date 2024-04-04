import subprocess
import streamlit as st
import os
st.set_page_config(
    page_title="关联分析",
    page_icon="graph-up-arrow",
    layout="wide"
)
from utils import *
logo()
# custom_css = """
# <style>
# .stApp {
#     margin-top: -55px; /* 负数值用于减少顶部空间 */
# }
# </style>
# """
# # 插入自定义 CSS
# st.markdown(custom_css, unsafe_allow_html=True)
st.write("#### 这里是关联分析与特征提取页面 👋")
st.write("###### one：选择建模所用数据")

# 获取指定目录下的所有文件
data_folder = "./数据预处理后的训练集和测试集"
all_files = os.listdir(data_folder)

# 筛选出所有以 ".csv" 结尾的文件
csv_files = [file for file in all_files if file.endswith('.csv')]
data = st.multiselect(
    label='选择数据集',
    options=csv_files,
    default=None,
    format_func=str,
    help='选择您训练模型准备使用的数据集'
)

# st.write("###### two：选择建模所用特征参数")
# method_choice = st.radio("选择方法", ["方法一：自定义参数组合建模", "方法二：选择一项默认参数组合"])
# if method_choice == "方法一：自定义参数组合建模":
#     search = st.multiselect(
#         label='自定义参数组合建模',
#         options=('GR', 'AC', 'CNL', 'CAL', 'DEN', 'RA25', 'RA45', 'SP', 'RT'),
#         default=None,
#         format_func=str,
#         help='请仔细检查所选参数组合是否已研究过',
#         placeholder='请选择您自定义的参数（可多选）'
#     )
#
#     sorting = st.text_input(
#         "参数组合命名", placeholder="请为所选参数组合命名"
#     )
#
# elif method_choice == "方法二：选择一项默认参数组合":
#     st.write("""
#     <span style='font-size: 14.7px;'>选择一项默认参数组合</span>
#     """, unsafe_allow_html=True)
#
#     st.checkbox("6RA : [ AC, CNL, DEN, GR, SP, RA25 ]", value=False)
#     st.checkbox("6RT : [ AC, CNL, DEN, GR, SP, RT ]", value=False)
#     st.checkbox("5RA : [ AC, CNL, GR, SP, RA25 ]", value=False)
# if data is None:
#     text = f'未选择数据集'
# 初始化结果变量
tsfresh_results = None
rf_selected_features = None
pca_results = None
st.write("###### two：特征提取与特征选择")
ts_button = st.button("tsfresh特征提取", key = "ts_button")
if ts_button:
    # 显示进度条
    progress_bar = st.progress(0)
    # 模拟训练过程
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    st.empty()
    # 创建新的列容器
    col, coll, colll = st.columns([3, 0.2, 1.2])
    # 显示成功提示框
    tsfresh_results = "这里是tsfresh提取的特征结果"
    text = f'数据集： :orange[{data}]已完成tsfresh特征提取！'
    col.success(text)

else:
    text = f'数据集：:red[{data}]'
    st.write("###### "+text)
    st.header(" ", divider='rainbow')

rf_button = st.button("随机森林特征选择",key = "rf_button")
if rf_button:
    # 显示进度条
    progress_bar = st.progress(0)
    # 模拟训练过程
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    st.empty()
    # 创建新的列容器
    col, coll, colll = st.columns([3, 0.2, 1.2])
    # 显示成功提示框
    text = f'数据集： :orange[{data}]已完成随机森林特征选择！'
    col.success(text)
    # col.success("已完成tsfresh特征选择！")
    # text = f'所选择的数据集为： :orange[{data}]'
else:
    text = f'数据集：:red[{data}]'
    st.write("###### "+text)
    st.header(" ", divider='rainbow')

pca_button = st.button("PCA降维融合特征",key = "pca_button")
if pca_button:
    # # 在按钮被点击时执行后端脚本，并捕获输出内容
    subprocess.check_output(['python', '训练集-提取特征-选特征-降维1.py'], encoding='gbk')
    # # 将输出内容显示在前端页面中的文本框中
    # st.text_area("特征工程：", value=output, height=400)
    # 显示进度条
    progress_bar = st.progress(0)
    # 模拟训练过程
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    st.empty()
    # 创建新的列容器
    col, coll, colll = st.columns([3, 0.2, 1.2])
    # 显示成功提示框
    text = f'数据集： :orange[{data}]已完成PCA特征融合！'
    col.success(text)
    # col.success("已完成PCA特征融合！")
    # text = f'所选择的数据集为： :orange[{data}]'
else:
    text = f'数据集：:red[{data}]'
    st.write("###### "+text)
    st.header(" ", divider='rainbow')
