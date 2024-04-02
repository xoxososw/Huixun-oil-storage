from PIL import Image
import streamlit as st
from streamlit_extras.app_logo import add_logo

# 打开图片
image = Image.open("images/logo.png")

# 设置要调整的大小
new_width = 215  # 设置新的宽度
new_height = 100  # 设置新的高度

# 调整图片大小
resized_image = image.resize((new_width, new_height))

# 保存调整大小后的图片
resized_image.save("images/resized_logo.png")

# 添加 Logo
def logo():
    add_logo("images/resized_logo.png")

# 调用 logo 函数
logo()
