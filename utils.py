import cv2
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class Regression(nn.Module):

    def __init__(self, n_dim, lr=1e-1):
        super().__init__()
        self.lr = lr
        self.fc = nn.Linear(n_dim, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

    # эпоха обучения
    def train_epoch(self, X, y, optimizer, loss):
        pred = self.forward(X)
        loss = loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    # инициализация оптимизатора и лосса
    def configure(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        return optimizer, loss

    # валидация
    def eval(self, X, y, loss):
        pred = self.forward(X)
        loss = loss(pred, y)
        return loss

    #функция обучения
    def fit(self, epochs, X_train, y_train, X_val, y_val, X, y, epsilon=1e-5):
        last_loss = 1e+6
        history = {'train': [], 'val': []}
        optimizer, loss_fn = self.configure()
        figure = st.empty()
        for epoch in range(epochs):
            tr_loss = 0.
            val_loss = 0.
            loss = self.train_epoch(X_train, y_train, optimizer, loss_fn)
            tr_loss += loss / len(X_train)
            history['train'].append(tr_loss.item())

            loss = self.eval(X_val, y_val, loss_fn)
            val_loss += loss / len(X_val)
            history['val'].append(val_loss.item())
            if epoch % 100== 0:
                print(f'epoch: {epoch+1} tr_loss: {tr_loss} val_loss: {val_loss}')

            # критерий остановы
            if abs(last_loss - history['val'][-1]) < epsilon:
                return self, history
            last_loss = history['val'][-1]

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].scatter(X, y, label='average marks', c=X)
            axes[0].plot(X, self.forward(X).detach().numpy(), c='r', label='predict')
            axes[0].set_title(f'y = x * {self.fc.weight.item()} + {self.fc.bias.item()} + {history["train"][-1]}')
            axes[0].set_ylabel('Средняя оценка')
            axes[0].set_xlabel(
                f'Количество долгов \n')
            axes[0].legend()
            axes[1].plot(np.arange(1, len(history['train']) + 1), history['train'], label='train')
            axes[1].plot(np.arange(1, len(history['val']) + 1), history['val'], label='val')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            figure.pyplot(fig, use_container_width=False)
        return self, history

def learn_model(task_linreg):
    data = None
    if task_linreg == 'Предсказание успеваемости':
        data = {
            'x': [0, 0, 1, 0, 1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 2, 4, 4, 5, 7, 5, 6, 8, 9, 7, 10],
            'y': [5, 4.95, 5, 4.9, 4.8, 4.7, 4.5, 4.45, 4.43, 4.41, 4.3, 4.1, 4.05, 4.0, 3.9, 3.7, 3.59, 3.55,
                    3.51, 3.4, 3.38, 3.35, 3.2, 3.09, 3.0, 2.8],}
    elif task_linreg == '':
        pass
    elif task_linreg == '':
        pass

    df = pd.DataFrame(data)

    col0_0, col0_1 = st.columns([3, 1])
    k = col0_0.slider(label='Коэффициент угла наклона линейной функции?', min_value=-10., max_value=10., value=0., step=0.001)
    b = col0_0.slider('Точка пересечения с осью ординат?', -100., 100., 0., step=0.1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    df['предсказание'] = k * df['x'] + b
    loss = np.mean(np.sum(df['предсказание'] - df['y']) ** 2)
    ax.set_title(f"Линейная регрессия: {k} * x + {b} + {loss}    Ошибка = {loss}", fontsize=8)
    ax.scatter(df['x'], df['y'], label='average marks')
    ax.plot(df['x'], df['предсказание'], label='average marks', c='r')
    ax.set_ylabel('Средняя оценка', fontsize=6)
    ax.set_xlabel(f'Количество долгов', fontsize=6)
    # ax[0].hist(hypot.work_days)
    # ax[1].set_title("Возраст", fontsize=6)
    # ax[1].hist(hypot.age)
    # ax[2].set_title("Пол", fontsize=6)
    # ax[2].pie(list(hypot.sex.value_counts()), explode=(0, 0.1), labels=['Мужчины', 'Женщины'], autopct='%1.1f%%',
    #           shadow=True, startangle=90,
    #           textprops={'fontsize': 5})
    col0_0.pyplot(fig, use_container_width=False)
    if loss < 1.:
        st.balloons()


    # отрисовка датафрейма
    col0_1.subheader(f"Датафрейм")
    col0_1.write(df)


def linreg_gd(task_linreg, lr):
    data = None
    if task_linreg == 'Предсказание успеваемости':
        data = {
            'x': [5, 4.95, 5, 4.9, 4.8, 4.7, 4.5, 4.45, 4.43, 4.41, 4.3, 4.1, 4.05, 4.0, 3.9, 3.7, 3.59, 3.55,
                    3.51, 3.4, 3.38, 3.35, 3.2, 3.09, 3.0, 2.8],
            'y': [0, 0, 1, 0, 1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 2, 4, 4, 5, 7, 5, 6, 8, 9, 7, 10]}
    elif task_linreg == '':
        pass
    elif task_linreg == '':
        pass

    df = pd.DataFrame(data)

    X = df['x'].values
    y = df['y'].values

    X = torch.Tensor(X).unsqueeze(-1)
    y = torch.Tensor(y).unsqueeze(-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Regression(n_dim=1, lr=lr)
    model.fit(epochs=10000, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, X=X, y=y,
                               epsilon=1e-6)


def cls(task_linreg):
    data = {
        'x': [5, 4.95, 5, 4.9, 4.8, 4.7, 4.5, 4.45, 4.43, 4.41, 4.3, 4.1, 4.05, 4.0, 3.9, 3.7, 3.59, 3.55,
                3.51, 3.4, 3.38, 3.35, 3.2, 3.09, 3.0, 2.8],
        'y': [0, 0, 1, 0, 1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 2, 4, 4, 5, 7, 5, 6, 8, 9, 7, 10]}

    df = pd.DataFrame(data)

    col0_0, col0_1 = st.columns([3, 1])
    k = col0_0.slider(label='Коэффициент w0', min_value=-10., max_value=10., value=0., step=0.001)
    b = col0_0.slider('Коэффициент w1', -100., 100., 0., step=0.1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    df['предсказание'] = k * df['x'] + b
    loss = np.mean(np.sum(df['предсказание'] - df['y']) ** 2)
    ax.set_title(f"Ошибка = {loss}", fontsize=8)
    ax.scatter(df['x'], df['y'], label='average marks')
    ax.plot(df['x'], df['предсказание'], label='average marks', c='r')
    ax.set_ylabel('x1', fontsize=6)
    ax.set_xlabel(f'x2', fontsize=6)
    col0_0.pyplot(fig, use_container_width=False)
    if loss < 1.:
        st.balloons()

    # отрисовка датафрейма
    col0_1.subheader(f"Датафрейм")
    col0_1.write(df)
