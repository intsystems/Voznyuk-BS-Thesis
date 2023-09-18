Инструкция по установке весов:

1. Скачать веса по ссылке

2. Если установлен Cmake выполнить скрипт install.sh

   2a. Если нет CMake, то можно попробовать просто запустить `make`, как написано [тут](https://github.com/ggerganov/llama.cpp#build)
  
3. Запустить скрипт для конвертации весов convert_weights.sh

4. Из llama.cpp (корневой папки) запустить основную команду
   `make -j & ./main -m ./models/7B/ggml-model-f16.gguf -p "Building a website is not hard because" -n 500 -с 1000`

   Если медленно работает, можно попробовать запустить quantized версию 
   `make -j & ./main -m ./models/7B/ggml-model-q4_0.gguf -p "Building a website is not hard because" -n 500 -с 1000` 

5. Скачать датасет medium.csv [отсюда](https://github.com/intsystems/2023-Project-126/blob/master/medium.csv)

5. Запустить [cкрипт](https://github.com/intsystems/2023-Project-126/blob/master/llama_replace_new.py). Там в конце закомментированы три типа генерации (когда генерируется два параграфа, три и четыре, примерно одинаковые доли текстов)
