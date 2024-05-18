# Детекция машинно-сгенерированных фрагментов на базе анализа смены стиля текста

Тип научной работы

**Автор:** Вознюк Анастасия

**Консультант/эксперт:** к.ф.-м.н. Грабовой Андрей Валерьевич

## Постановка задачи

Предложить модель для определения границы между частью текста, написанной человеком, и продолжением этой части, сгенерированной языковой моделью. Данная граница может быть в любой части текста, но она проходит по словам. 

Дополнительно изучались возможности решать задачу для гипотезы, когда авторы меняются по параграфам.

Предлагается использовать трансформерные архитектуры в качестве решения, так как на данный момент именно они показывают наилучшие результаты~\footnote{Macko et al., 2023, MULTITuDE: Large-Scale Multilingual Machine-Generated Text Detection Benchmark}

## Как запускать

Running experiment with [mldev](https://gitlab.com/mlrep/mldev) involves the following steps.

Install the ``mldev`` by executing

```bash
$ git clone https://gitlab.com/mlrep/mldev 
$ cd ./mldev && git checkout -b 79-fixes-for-0-3-dev1-exploreparams --track origin/79-fixes-for-0-3-dev1-exploreparams
$ ./install_reqs.sh core
$ python setup.py clean build install
``` 
Then get the repo
```bash
$ git clone <this repo>
$ cd <this repo folder>
```

Then initialize the experiment, this will install required dependencies

```bash
$ mldev init -p venv .
```
Now install mldev into this venv as follows (need this to run sub-experiment)

```bash
$ /bin/bash -c "source venv/bin/activate; cd ../mldev && python setup.py clean build install"
```

Detailed description of the experiment can be found in [experiment.yml](./experiment.yml). See docs for [mldev](https://gitlab.com/mlrep/mldev) for details.

Run simple experiment for a specific set of params

```bash
$ mldev run pipeline
```

And now, run the full experiment with params grid explored. See [explore_params.yml](./explore_params.yml) for details.

```bash
$ mldev run run_grid
```

Results will be placed into [./results](./results) folder.

## Проведение полного эксперимента 

Скрипт [./runs.sh](./run_experiment.sh) запускает эксперимент для модели DeBERTav3-large

## Исходный код


Пример.

Исходники кода находятся в [./src](./code) .  [main.py](./code/main.py) содержит запуск экспериментов.
[experiment.py](./code/experiment.py) содержит реализацию шаблона проведения эксперимента.
Данные сохраняются при помощи [results.py](./code/results.py) для каждого проведённого эксперимента.
[mathmodel.py](./code/mathmodel.py) cодержит основные компоненты для провдения экспериментов. .

## Что осталось сделать

TODO Указать, если что-то из задуманного пока не реализованного

Пример.

 - [ ] add a sample iPython notebook 
 - [ ] make the template support arbitrary experiment parameters without rewriting [main.py](./src/main.py)

## Как процитировать работу

TODO Указать ссылку на публикацию или arxiv. Если пока нет публикации дать ссылку на этот репозиторий в формате Bibtex

## Лицензия
