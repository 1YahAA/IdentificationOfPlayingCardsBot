import os
import telebot
import tensorflow as tf
import numpy as np
from telebot import types

# Инициализация бота
bot = telebot.TeleBot("7163848746:AAHwZfQV6wfqZGy9H-Hmkzf05iEknwzrQUY") #токен

# Загрузка модели
model = tf.keras.models.load_model('card_classification_model.h5')

# Функция для предсказания карты
def predict_card(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # создаем батч из одного изображения
    img_array = img_array / 255.0  # масштабирование

    # Предсказание
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = [
        'туз крести', 'туз бубен', 'туз червей', 'туз пик',
        'восьмёрка крести', 'восьмёрка бубен', 'восьмёрка червей', 'восьмёрка пик',
        'пятёрка крести', 'пятёрка бубен', 'пятёрка червей', 'пятёрка пик',
        'четвёрка крести', 'четвёрка бубен', 'четвёрка червей', 'четвёрка пик',
        'крестовый валет', 'бубновый валет', 'червовый валет', 'пиковый валет',
        'джокер', 'крестовый король', 'бубновый король', 'червовый король', 'пиковый король',
        'девятка крести', 'девятка бубен', 'девятка червей', 'девятка пик',
        'крестовая дама', 'бубовая дама', 'червовая дама', 'пиковая дама',
        'семерка крести', 'семерка бубен', 'семерка червей', 'семерка пик',
        'шестерка крести', 'шестерка бубен', 'шестерка червей', 'шестерка пик',
        'десятка крести', 'десятка бубен', 'десятка червей', 'десятка пик',
        'тройка крести', 'тройка бубен', 'тройка червей', 'тройка пик',
        'двойка крести', 'двойка бубен', 'двойка червей', 'двойка пик'
    ]
    predicted_class = class_names[np.argmax(score)]
    confidence = np.max(score) * 2000

    return predicted_class, confidence

# Функция для отправки главного меню
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Что умеет этот бот")
    item2 = types.KeyboardButton("Распознать карту")
    markup.add(item1, item2)
    bot.send_message(message.chat.id, "Добро пожаловать! Выберите действие:", reply_markup=markup)

# Обработчик старта
@bot.message_handler(commands=['start'])
def start(message):
    send_welcome(message)

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text(message):
    if message.text == "Что умеет этот бот":
        bot.send_message(message.chat.id, "Этот бот может распознавать игральные карты.")
    elif message.text == "Распознать карту":
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        item = types.KeyboardButton("Загрузить изображение")
        markup.add(item)
        bot.send_message(message.chat.id, "Пожалуйста, загрузите изображение карты.", reply_markup=markup)
    else:
        bot.send_message(message.chat.id, "Пожалуйста, используйте кнопки для взаимодействия с ботом.")

# Обработчик сообщений с изображениями
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)

    with open('temp.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    predicted_card, confidence = predict_card('temp.jpg')

    bot.reply_to(message, f"На изображении скорее всего {predicted_card} ({confidence:.2f}% вероятность)")

# Ограничиваем возможность отправки текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_all_message(message):
    if message.content_type != 'photo':
        send_welcome(message)

bot.polling()