import streamlit as st
import pandas as pd
import os
import subprocess
from utils import *
# 页面配置和自定义 CSS
st.set_page_config(page_title="模型建立与预测", page_icon="💻", layout="wide")
# custom_css = """
# <style>
# .stApp {margin-top: -55px;} /* 负数值用于减少顶部空间 */
# </style>
# """
logo()
def main_page():
    st.write("#### 这里是模型建立与预测页面 👋")

    # 选择建模所用数据
    st.write("###### one：选择建模所用数据")
    folder_path = os.path.join(os.getcwd(), "..", "数据预处理后的训练集和测试集")
    st.session_state.data = st.selectbox(
        label='选择数据集',
        options=('小层+测井数据+录井+训练+5RA25-addShui.csv', '小层+测井数据+录井+训练+6RA25-addShui.csv', '小层+测井数据+录井+训练+6RT-addShui.csv'),
        index=None,
        format_func=str,
        help='选择您训练模型准备使用的数据集'
    )

    # 选择模型训练过程所用参数
    st.write("###### two：选择模型训练过程所用参数")
    st.session_state.iterations = st.number_input(
    '模型迭代次数',
    min_value=0,
    max_value=100,
    value=st.session_state.get('iterations', 10),  # 使用get从session_state获取迭代次数
    step=1,
    help='请输入您想要训练模型的迭代次数'
)

    # 模型命名与训练
    st.write("###### three：模型命名与训练")
    st.session_state.model_name = st.text_input(
    '模型名称',
    value=st.session_state.get('model_name', ""),  # 使用get从session_state获取模型名称
    max_chars=100,
    placeholder='当使用同样的参数设置进行建模时，建议各算法的模型按同样的名字命名'
)

    col1, col2, col3 = st.columns([1.5, 1, 1])
    # 按钮触发训练

    if col2.button("开始训练", key="train_button", type="primary"):
        # 更新训练记录
        subprocess.check_output(['python', 'lsf.py'], encoding='gbk')
        if 'training_records' not in st.session_state:
            st.session_state.training_records = []
        st.session_state.training_records.append(f"{st.session_state.model_selection}训练已完成: {st.session_state.model_name}")
        # 显示进度条
        progress_bar = st.progress(0)
        # 模拟训练过程
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        st.empty()
        # 创建新的列容器
        col, coll, colll = st.columns([3, 0.2, 1.2])
        # 显示成功提示框
        col.success(f"{st.session_state.model_selection}已训练完成！")

        # 创建一个下载按钮
        # colll.button('下载已训练的模型pkl文件', key=f"download_button")



# st.markdown(custom_css, unsafe_allow_html=True)

# 侧边栏模型选择
model_options = ["#### AdaBoost模型", "#### GBDT模型","#### 随机森林模型", "#### XGBoost模型"]
selected_model = st.sidebar.radio(
    "请选择模型",
    model_options,
    index=0
)

# 检测模型选择变化并重置状态
if selected_model != st.session_state.get('model_selection', ''):
    # 保存训练记录
    training_records = st.session_state.get('training_records', [])
    # 重置session_state，除了训练记录
    st.session_state.clear()
    st.session_state.model_selection = selected_model
    st.session_state.training_records = training_records
    # 触发页面重载
    st.experimental_rerun()


# 调用主页面渲染函数
main_page()

# 侧边栏显示训练记录
if 'training_records' in st.session_state:
    st.sidebar.markdown("#### 训练记录:")
    for record in st.session_state.training_records:
        st.sidebar.write(record)
