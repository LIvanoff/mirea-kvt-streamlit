import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from utils import learn_model, linreg_gd

VIDEO_DIR = "video"


def run():
    st.set_page_config(layout="wide")
    st.sidebar.title("Профиль ИСППР")
    st.sidebar.markdown("Студенты данного профиля изучают теорические и практические основы разработки и проектирования интеллектуальных систем.")
    st.sidebar.markdown("Во время обучения читаются курсы, посвященные машинному обучению, созданию экспертных систем, а также обучению нейронных сетей.")
    st.sidebar.divider()
    st.sidebar.title("Что нужно знать сегодняшнему специалисту в области Data Science?")
    st.sidebar.markdown("В рамках обучения студентами будет изучен стек знаний нужный дата-сайентисту и ML-разработчик, в который входят методы классического обучения "
                        "линейная и логистическая регрессия, дерево решений, ансамблевые методы. Также сегодняшнему специалисту в данной области очень важно обладать знаниями"
                        "в области глубокого обучения, необходимые знания в данной области даются во время прохождения курса по Проектированию и обучению нейронных сетей, "
                        "в частности изучается архитектуры трансформенных, сверточных и рекуррентных нейронных сетей.")

    st.sidebar.text(" ")
    st.sidebar.text(" ")

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title("Искусственный интеллект от ИИТ")
    with row0_2:
        st.text("")
        st.subheader('[Ivanov Leonid](https://github.com/LIvanoff/)')

    st.subheader("Обучи свою модель машинного обучения!")
    st.markdown("Начнем с линейной регресии. Обучим модель для задачи..")
    task_linreg = st.selectbox(label='Задачи:',options=('Предсказание успеваемости', '', ''))
    st.markdown("Для начала попробуй сам составить модель, подобрав нужные коэффициенты для нее, k - коэффициент угла наклона линейной функции, а также  b - точку пересечения с осью ординат.")
    st.markdown("")
    learn_model(task_linreg)
    st.divider()
    st.subheader("Линейная регрессия с SGD!")
    st.markdown("Теперь дадим обучиться нашей модели самостоятельно с помощью градиентного спуска.")


    lr = st.slider(label='Скорость обучения', min_value=0.001, max_value=0.99,
                            value=0.1, step=0.001)
    if st.button("Обучить модель", type="primary"):
        linreg_gd(task_linreg, lr)

    st.divider()
    st.subheader("Классифицируем!")
    st.markdown("Задача классификации представляет собой создание модели способной разбить объекты, исходя из их признаков, на заранее определенные классы.")
    st.image('img/yandex_handbook.png')
    st.markdown(
        "https://education.yandex.ru/handbook/ml/article/linear-models")

    st.divider()
    st.title("Прикладные задачи")
    st.markdown("К прикладным задачам относяться нетолько задачи основанные на табличных данных, но и более сложные задачи, которые сегодня решаются с помощью методов глубокого обучения, "
                " то есть с помощью глубоких нейронных сетей."
                " В частности задачи компьютерного зрения (CV) и задачи относящиеся к области обработки естественного языка (NLP).")

    st.subheader("Семантическая сегментация")
    video_file = open('video/3D LiDAR Semantic Segmentation (Experimental Version).mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.divider()
    st.subheader("Обнаружение объектов")
    video_file = open("video/3D Object Detection for Autonomous Driving using Deep Learning (Master's Thesis Project).mp4", 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


    st.divider()
    st.title("Мэтчинг данных")
    col1, col2 = st.columns(2)
    col1.subheader('Исходное изображение')
    col1.image('img/img_src.PNG')
    col2.subheader('Найденные похожие изображение')
    col2.image('img/img_match0.PNG')
    col2.image('img/img_match1.PNG')
    col2.image('img/img_match2.PNG')

    # vids = os.listdir(VIDEO_DIR)
    # print("Availible videos: ", *vids)

    # count = 0
    # stframe1 = st.empty()
    # stframe2 = st.empty()
    # vid_capture1 = load_video(VIDEO_DIR, count, vids)
    # vid_capture2 = load_video(VIDEO_DIR, count+1, vids)
    # count = 2
    # # stframe1, stframe2 = st.columns(2)
    #
    # while True:
    #     success1, frame = vid_capture1.read()
    #     if success1:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         stframe1.image(frame)
    #
    #     success2, frame = vid_capture2.read()
    #     if success2:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         stframe2.image(frame)
    #
    #     if success1 is False:
    #         vid_capture1 = load_video(VIDEO_DIR, count, vids)
    #         count += count_update(count, len(vids))
    #     if success2 is False:
    #         vid_capture2 = load_video(VIDEO_DIR, count, vids)
    #         count = count_update(count, len(vids))


if __name__ == '__main__':
    run()

