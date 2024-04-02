import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="融合模型预测结果",
    page_icon="shield-check",
    layout="wide"
)

# 初始化 session_state，如果已经初始化过，则不再重复初始化
if 'model_selection' not in st.session_state:
    st.session_state['model_selection'] = None
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = None
custom_css = """
<style>
.stApp {
    margin-top: -55px; /* 负数值用于减少顶部空间 */
}
</style>
"""

# 插入自定义 CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.write("#### 这里是融合模型预测结果页面 👋")

# 定义用于存储结果的变量
ronghe_result = None
daochu_result = None


# AdaBoost模型预测结果选择框和按钮
option1 = st.selectbox(
    "**请选择测试数据集**",
    ("小层+测井数据+录井+测试+5RA25.csv", "小层+测井数据+录井+测试+6RA25.csv", "小层+测井数据+录井+测试+6RT.csv"),
    index=None,
    placeholder=" -请选择特定参数数据集的预测结果-",
    help='选择您测试模型准备使用的数据集'
)
def show_ronghe_result():
    global ronghe_result
    ronghe_result = option1
def show_daochu_result():
    global daochu_result
    ronghe_result = option1
    st.success(f'已选择融合模型预测结果导出为：:red[融合结果_{ronghe_result}.csv]')

if option1 is not None:
    show_ronghe_result()
if option1 is not None:
    text = f'已选择融合模型预测的测试数据集为： :orange[{option1}]'
else:
    text = '暂时未选择数据集预测结果...'
st.write("###### "+text)
# mean = st.checkbox('求预测概率平均数')
# sort = st.checkbox('根据概率排序')
# st.write('求预测概率平均数 , 展示结果根据概率排序')
col1, col3,col2= st.columns([7,0.3,1])
with col1:
    model = st.multiselect(
        label='**请选择模型**',
        options=('AdaBoost模型-5RA25', 'Adaboost模型-6RA25','Adaboost模型-6RT',
                 'GBDT模型-5RA25', 'GBDT模型-6RA25','GBDT模型-6RT',
                 '随机森林模型-5RA25', '随机森林模型-6RA25','随机森林模型-6RT',
                 'XGBoost模型-5RA25', 'XGBoost模型-6RA25','XGBoost模型-6RT'),
        default=None,
        format_func=str,
        placeholder='-选择您训练模型准备使用的模型-',
        help='选择您训练模型准备使用的模型'
    )
# 按钮触发训练
with col2:
    st.write("")
    st.write("")
    test_button=st.button("开始预测", key="test_button", type="primary")
if test_button:
    # subprocess.check_output(['python', '模型汇总.py'], encoding='gbk')
    # 更新训练记录
    # if 'training_records' not in st.session_state:
    #     st.session_state.training_records = []
    # st.session_state.training_records.append(f"{st.session_state.model_selection}预测已完成: {st.session_state.model_name}")
    # 显示进度条
    progress_bar = st.progress(0)
    # 模拟训练过程
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    st.empty()
    # 创建新的列容器

    # 显示成功提示框
    st.success(f'''结果：\n
            数据集：{option1}\n
    模型：{model}\n
    已预测完成！
    '''
        )


daochu_button_key = 'daochu_button'  # 唯一的key参数
daochu_button = st.button("融合模型展示预测结果", key=daochu_button_key)
if daochu_button and option1 is not None:
    show_daochu_result()
    data_path = 'modelce/6RT/merge-result/合并预测结果.csv'
    # 读取 CSV 文件为 DataFrame
    data = pd.read_csv(data_path)


    # 自定义样式函数，用于同时加粗最后一列并突出显示每列的最大值
    def style_dataframe(data):
        def bold_last_column_and_highlight_max(cell):
            styles = ['background-color: yellow' if cell.name == data.columns[-1] else '' for i in range(len(cell))]
            return [f'{styles[i]}' for i in range(len(cell))]

        return data.style.apply(bold_last_column_and_highlight_max, axis=0)
    # 在Streamlit应用中展示样式处理后的DataFrame
    st.dataframe(style_dataframe(data))
st.header(" ", divider='rainbow')