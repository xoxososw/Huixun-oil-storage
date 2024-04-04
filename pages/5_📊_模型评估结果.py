import streamlit as st
from utils import *

st.set_page_config(
    page_title="潜力层识别",
    page_icon="shield-check",
    layout="wide"
)
logo()
# custom_css = """
# <style>
# .stApp {
#     margin-top: -55px; /* 负数值用于减少顶部空间 */
# }
# </style>
# """
# 设置文本框的样式
text_area_style = {
    'font-size': '20px'  # 设置字体大小为20像素
}
# 插入自定义 CSS
# st.markdown(custom_css, unsafe_allow_html=True)

st.write("#### 这里是潜力层识别页面 👋")

# 定义用于存储结果的变量
ada_result = None
xgb_result = None
gbdt_result = None
rf_result = None
# 侧边栏模型选择
model_options = ["#### AdaBoost模型", "#### GBDT模型","#### 随机森林模型", "#### XGBoost模型"]
selected_model = st.sidebar.radio(
    "#### 请选择模型",
    model_options,
    index=0
)
if selected_model == "#### AdaBoost模型":
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    # 第一列：显示图片
    with col1:
        st.write("#### 1. 5RA25:")
        st.image('./四个模型/ADA/5RA25/ADA-混淆矩阵.png', caption='Adaboost-5RA25-混淆矩阵', width=400)
        st.write("#### 2. 6RA25:")
        st.image('./四个模型/ADA/6RA25/ADA-混淆矩阵.png', caption='Adaboost-6RA25-混淆矩阵', width=400)
        st.write("#### 3. 6RT:")
        st.image('./四个模型/ADA/6RT/ADA-混淆矩阵.png', caption='Adaboost-6RT-混淆矩阵', width=400)
    # 第二列：显示文本内容
    with col2:
        # 使用HTML标签设置文本为标题1大小
        with open("./四个模型/ADA/5RA25/ADA-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.header(" ", divider='rainbow')
        st.write("#### Adaboost-5RA25-评分结果：")
        st.write(txt_content)
        # st.text_area("", value=txt_content, height=500)
        with open("./四个模型/ADA/6RA25/ADA-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### Adaboost-6RA25-评分结果：")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
        with open("./四个模型/ADA/6RT/ADA-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### Adaboost-6RT-评分结果： ")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)


if selected_model == "#### XGBoost模型":
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    # 第一列：显示图片
    with col1:
        st.write("#### 1. 5RA25:")
        st.image('./四个模型/XGBOOST/5RA25/XGB-混淆矩阵.png', caption='XGBOOST-5RA25-混淆矩阵', width=400)
        st.write("#### 2. 6RA25:")
        st.image('./四个模型/XGBOOST/6RA25/XGB-混淆矩阵.png', caption='XGBOOST-6RA25-混淆矩阵', width=400)
        st.write("#### 3. 6RT:")
        st.image('./四个模型/XGBOOST/6RT/XGB-混淆矩阵.png', caption='XGBOOST-6RT-混淆矩阵', width=400)
    # 第二列：显示文本内容
    with col2:
        # 使用HTML标签设置文本为标题1大小
        with open("./四个模型/XGBOOST/5RA25/XGB-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### XGBOOST-5RA25-评分结果：")
        st.write(txt_content)
        # st.text_area("", value=txt_content, height=500)
        with open("./四个模型/XGBOOST/6RA25/XGB-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### XGBOOST-6RA25-评分结果：")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
        with open("./四个模型/XGBOOST/6RT/XGB-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### XGBOOST-6RT-评分结果： ")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
if selected_model == "#### GBDT模型":
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    # 第一列：显示图片
    with col1:
        st.write("#### 1. 5RA25:")
        st.image('./四个模型/GBDT/5RA25/GBDT-混淆矩阵.png', caption='GBDT-5RA25-混淆矩阵', width=400)
        st.write("#### 2. 6RA25:")
        st.image('./四个模型/GBDT/6RA25/GBDT-混淆矩阵.png', caption='GBDT-6RA25-混淆矩阵', width=400)
        st.write("#### 3. 6RT:")
        st.image('./四个模型/GBDT/6RT/GBDT-混淆矩阵.png', caption='GBDT-6RT-混淆矩阵', width=400)
    # 第二列：显示文本内容
    with col2:
        # 使用HTML标签设置文本为标题1大小
        with open("./四个模型/GBDT/5RA25/评分结果.txt", "r") as f:
            txt_content = f.read()
        st.header(" ", divider='rainbow')
        st.write("#### GBDT-5RA25-评分结果：")
        st.write(txt_content)
        # st.text_area("", value=txt_content, height=500)
        with open("./四个模型/GBDT/6RA25/评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.header(" ", divider='rainbow')
        st.write("#### GBDT-6RA25-评分结果：")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
        with open("./四个模型/GBDT/6RT/评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.write(" ")
        st.write(" ")

        st.header(" ", divider='rainbow')
        st.write("#### GBDT-6RT-评分结果： ")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
if selected_model == "#### 随机森林模型":
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    # 第一列：显示图片
    with col1:
        st.write("#### 1. 5RA25:")
        st.image('./四个模型/RF/5RA25/RF-混淆矩阵.png', caption='RF-5RA25-混淆矩阵', width=400)
        st.write("#### 2. 6RA25:")
        st.image('./四个模型/RF/6RA25/RF-混淆矩阵.png', caption='RF-6RA25-混淆矩阵', width=400)
        st.write("#### 3. 6RT:")
        st.image('./四个模型/RF/6RT/RF-混淆矩阵.png', caption='RF-6RT-混淆矩阵', width=400)
    # 第二列：显示文本内容
    with col2:
        # 使用HTML标签设置文本为标题1大小
        with open("./四个模型/RF/5RA25/RF-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.header(" ", divider='rainbow')
        st.write("#### RF-5RA25-评分结果：")
        st.write(txt_content)
        # st.text_area("", value=txt_content, height=500)
        with open("./四个模型/RF/6RA25/RF-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### GBDT-RF-评分结果：")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
        with open("./四个模型/RF/6RT/RF-评分结果.txt", "r") as f:
            txt_content = f.read()
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.header(" ", divider='rainbow')
        st.write("#### RF-6RT-评分结果： ")
        st.write(txt_content)
        # st.text_area(" ", value=txt_content, height=500)
# # AdaBoost模型预测结果选择框和按钮
# option1 = st.selectbox(
#     "**1.请选择Adaboost模型预测结果**",
#     ("AdaBoost-006-5RA", "AdaBoost-008-6RA", "..."),
#     index=None,
#     placeholder=" -请选择预测结果-",
# )
# ada_button_key = 'ada_button'  # 唯一的key参数
# ada_button = st.button("AdaBoost展示预测结果", key=ada_button_key)
# def show_ada_result():
#     global ada_result
#     ada_result = option1
#     st.success(f'已选择AdaBoost模型预测结果为：{ada_result}')
# if ada_button and option1 is not None:
#     show_ada_result()
# if option1 is not None:
#     text = f'已选择AdaBoost模型预测结果为： :orange[{option1}]'
# else:
#     text = '暂时未选择模型预测结果...'
# st.write("###### "+text)
# st.header(" ", divider='rainbow')
#
# # XGBoost模型预测结果选择框和按钮
# option2 = st.selectbox(
#     "**2.请选择XGBoost模型预测结果**",
#     ("XGBoost-006-5RA", "XGBoost-008-6RA", "..."),
#     index=None,
#     placeholder=" -请选择预测结果-",
# )
# xgb_button_key = 'xgb_button'  # 唯一的key参数
# xgb_button = st.button("XGBoost展示预测结果", key=xgb_button_key)
# def show_xgb_result():
#     global xgb_result
#     xgb_result = option2
#     st.success(f'已选择XGBoost模型预测结果为：{xgb_result}')
# if xgb_button and option2 is not None:
#     show_xgb_result()
# if option2 is not None:
#     text = f'# 已选择XGBoost模型预测结果为： :orange[{option2}]'
# else:
#     text = '暂时未选择模型预测结果...'
# st.write("###### "+text)
# st.header(" ", divider='rainbow')
#
# # GBDT模型预测结果选择框和按钮
# option3 = st.selectbox(
#     "**3.请选择GBDT模型预测结果**",
#     ("GBDT-006-5RA", "GBDT-008-6RA", "..."),
#     index=None,
#     placeholder=" -请选择预测结果-",
# )
# gbdt_button_key = 'gbdt_button'  # 唯一的key参数
# gbdt_button = st.button("GBDT展示预测结果", key=gbdt_button_key)
# def show_gbdt_result():
#     global gbdt_result
#     gbdt_result = option3
#     st.success(f'已选择GBDT模型预测结果为：{gbdt_result}')
# if gbdt_button and option3 is not None:
#     show_gbdt_result()
# if option3 is not None:
#     text = f'已选择GBDT模型预测结果为： :orange[{option3}]'
# else:
#     text = '暂时未选择模型预测结果...'
# st.write("###### "+text)
# st.header(" ", divider='rainbow')
#
# # 随机森林模型预测结果选择框和按钮
# option4 = st.selectbox(
#     "**4.请选择随机森林模型预测结果**",
#     ("随机森林-006-5RA", "随机森林-008-6RA", "..."),
#     index=None,
#     placeholder=" -请选择预测结果-",
# )
# rf_button_key = 'rf_button'  # 唯一的key参数
# rf_button = st.button("随机森林展示预测结果", key=rf_button_key)
# def show_rf_result():
#     global rf_result
#     rf_result = option4
#     st.success(f'已选择随机森林模型预测结果为：{rf_result}')
# if rf_button and option4 is not None:
#     show_rf_result()
# if option4 is not None:
#     text = f'已选择随机森林模型预测结果为： :orange[{option4}]'
# else:
#     text = '暂时未选择模型预测结果...'
# st.write("###### "+text)
# st.header(" ", divider='rainbow')
