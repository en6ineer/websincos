from flask import Flask, request, render_template
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

app = Flask(__name__)

# Папка для хранения загруженных изображений
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    # Получаем загруженное изображение
    period = float(request.form['period'])
    function_type = request.form['function_type']
    direction = request.form['direction']
    file = request.files['image']

    # Сохраняем изображение
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Загружаем изображение
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Получаем размеры изображения
    height, width, _ = img.shape
    # Создаем периодическую функцию
    if direction == 'horizontal':
        x = np.linspace(0, 2 * np.pi, width)
        if function_type == 'sin':
            wave = (np.sin(x / period) + 1) / 2  # Нормировка
        else:
            wave = (np.cos(x / period) + 1) / 2  # Нормировка

        wave = np.tile(wave, (height, 1))  # Повторяем по высоте
        wave = np.stack([wave] * 3, axis=-1)  # Создаем 3 канала
    else:
        y = np.linspace(0, 2 * np.pi, height)
        if function_type == 'sin':
            wave = (np.sin(y / period) + 1) / 2  # Нормировка
        else:
            wave = (np.cos(y / period) + 1) / 2  # Нормировка

        wave = np.tile(wave[:, np.newaxis], (1, width))  # Повторяем по ширине
        wave = np.stack([wave] * 3, axis=-1)  # Создаем 3 канала

    # Умножаем изображение на периодическую функцию
    new_img = img * wave
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # Сохраняем новое изображение
    new_image_path = os.path.join(UPLOAD_FOLDER, 'new_' + file.filename)
    cv2.imwrite(new_image_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    # Создаем график распределения цветов
    plt.figure(figsize=(12, 6))
    # Распределение цветов исходного изображения
    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), bins=256, color='blue', alpha=0.7, label='Original Image')
    plt.title('Color Distribution - Original')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Распределение цветов нового изображения plt.subplot(1, 2, 2)
    plt.hist(new_img.ravel(), bins=256, color='red', alpha=0.7, label='Processed Image')
    plt.title('Color Distribution - Processed')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Сохраняем график
    graph_path = os.path.join(UPLOAD_FOLDER, 'color_distribution.png')
    plt.savefig(graph_path)
    plt.close()

    # Перенаправляем на страницу с результатами
    return render_template('result.html', new_image=new_image_path, graph=graph_path)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
