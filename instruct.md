Инструкция по установке весов:

1. Скачать веса по ссылке

2. Если установлен Cmake выполнить скрипт install.sh
  
3. Запустить скрипт для конвертации весов weight.sh

4. Из llama.cpp (корневой папки) запустить основную команду
   `make -j & ./main -m ./models/7B/ggml-model-f16.gguf -p "Building a website is not hard because" -n 500 -с 1000`

   Если медленно работает, можно попробовать запустить quantized версию 
   `make -j & ./main -m ./models/7B/ggml-model-q4_0.gguf -p "Building a website is not hard because" -n 500 -с 1000` 


