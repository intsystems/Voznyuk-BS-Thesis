|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Detection of machine-generated fragments in text
    :Тип научной работы: M1P
    :Автор: Анастасия Вознюк
    :Научный руководитель: к.ф.-м.н. Грабовой Андрей
    
Baseline Experiment:
========

Assuming each parsgrsph has only author we can try to represent paragrpahs. For that we fine-tuned RoBERTa for classification of paragraphs then used embedders for [CLS] token to cluster vector respresentations of paragraphs.

`Link to code <https://github.com/intsystems/2023-Project-126/blob/master/code/fine_tuning_vectorizers_paragraph_level.ipynb>`_


Datasets
========

✅ CEFR Levelled English Texts - texts of different complexity - https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts

❓ CLiPS Stylometry Investigation (CSI) Corpus: a yearly expanded corpus of student texts in two genres: essays and reviews. The purpose of this corpus lies primarily in stylometric research, but other applications are possible. (on request)

❓ Gutenberg Standardized Corpus: Standardized Project Gutenberg Corpus, 55905 books (3GB counts + 18GB tokens)

❓ Wesbury Lab Wikipedia Corpus - Snapshot of all the articles in the English part of the Wikipedia that was taken in April 2010. It was processed, as described in detail below, to remove all links and irrelevant material (navigation text, etc) The corpus is untagged, raw text. Used by Stanford NLP (1.8 GB).

❓ Wikipedia Extraction (WEX): a processed dump of english language wikipedia (66 GB)


❓ Wikipedia Dataset - Created by Facebook Research at 2016, the Wikipedia The 2016-12-21 dump of English Wikipedia., in English language. Containing 5,075,182 in SQL file format.

❓ SFU Opinion and Comments Corpus (SOCC) Dataset https://metatext.io/datasets/sfu-opinion-and-comments-corpus-(socc)

✅ All the News 2.0 Dataset -  contains 2.7 million articles from 26 different publications from January 2016 to April 1, 2020., in English language. Containing 2.7M in CSV file format. https://metatext.io/datasets/all-the-news-2.0

❓ NELA-GT-2019 Dataset - contains 1.12M news articles from 260 sources collected between January 1st 2019 and December 31st 2019. Included are source-level ground truth labels from 7 different assessment sites. , in English language. Containing 1.12M in JSON file format. https://metatext.io/datasets/nela-gt-2019


Comparsion of Different Open LLM Models
========

The prompt is: "The main reasons why it's so hard to launch a rocketship are"


.. list-table:: 
   :widths: 25 700 50 50
   :header-rows: 1

   * - LLM
     - Example text
     - Number of parameters
     - Link
   * - LLaMA
     - The main reasons why it's so hard to launch a rocketship are: There is a lot of gravity between the two launch sites - especially if they are in polar orbit, which is where you want to go for cheaper rockets. You have to make sure your engine can overcome this and that you have enough fuel to get out of Earth's gravity well.Launching from near the equator maximizes the amount of available rocketry (thrust) per kg of payload, which means it costs less. In order to launch a satellite in polar orbit and not waste all the thrust you got for free on your ascent, you need to have some crazy amount of fuel - at least 2/3rds the mass of the entire rocketship (assuming an ideal engine).
     - 7B
     - `LLaMA <https://github.com/ggerganov/llama.cpp>`_
   * - Alpaca
     - Rocketships are incredibly difficult to launch due to their complexity and cost, as well as the need for specialized facilities. Firstly, rockets require complex engineering in order to be able to withstand takeoff and flight conditions while also being powerful enough to reach orbit or a desired destination. Secondly, they can often cost hundreds of millions if not billions of dollars due to their complexity and number of components involved. Last but certainly not least, launching rockets requires access to specialized facilities such as spaceports which are usually limited in numbers around the world. All these factors make rocketships a challenging endeavor for any company or organization looking to reach orbit or beyond.
     - 7B
     - `Alpaca <https://github.com/tatsu-lab/stanford_alpaca`_
   * - Cerebras-GPT
     - The main reasons why it's so hard to launch a rocketship are the following: The rocket is too big and heavy for its own weight, which makes launching from orbit impossible (it would be very difficult).  This problem can also occur with other types of satellites such as balloons or even small drones that have been launched by humans in space before they reach their destination - this means there will always some risk involved when trying out new technologies like these on Earth; but if you're going into an area where people might not want your technology then don't do anything risky until after all launches were completed successfully!
     - 1.3B
     - -
   * - OpenLLaMA
     - The main reasons why it is so hard to launch a rocketship are:\1. It’s not easy for the average person (or even an expert) to understand how rocket engines work, and what they do in spaceflight operations; 2. There aren't many people who have experience with building or operating them on Earth - which means that there isn't much of any practical knowledge about their operation available anywhere else but NASA itself! This makes things very difficult when trying to figure out exactly where something went wrong during testing at one point because you don&apos; t know if your assumptions were correct enough before starting up again after fixing whatever problem occurred last time around... Or worse yet- If someone has already tried this exact same thing multiple times without success then chances increase exponentially as more attempts occur over longer periods until eventually failure occurs due lack Of sufficient understanding between all parties involved regarding proper procedure
     - 3B
     - -
   * - ChatGPT
     - Launching rockets is immensely challenging due to a convergence of factors. Designing a craft that endures extreme launch conditions demands cutting-edge engineering and materials. The propulsion system must generate sufficient thrust for escape velocity. Safety is paramount, necessitating meticulous testing and redundancy. Overcoming aerodynamic forces and managing intense heat is complex. Precise navigation amid Earth's rotation and debris is critical. Compliance with intricate regulations, substantial costs, and funding hurdles add to the complexity. Despite these obstacles, progress in technology and innovation continues to push the boundaries of space exploration.
     - -
     - -
   * - GPT-2
     - The main reasons why it is so hard to launch a rocketship are because the engines are so heavy. The engine is so heavy that when it is under a full pressure of water it will explode at about 100 knots and burn up in about 1.2 minutes, according to the International Space Station.However, it is possible to launch an Ariane 5 rocket in such a short time, said NASA spokesman Scott Wiebe. The spacecraft is equipped with a new generation of engines to support it, which could be installed in any of NASA's existing rockets.
     - 345M
     - -
   * - GPT-J
     - The rocket has no engine, only fins and wings that can be used for steering purposes but not thrusting (like in airplanes).  So you need some kind of propulsion system which will provide both lift AND forward motion at once! This means your vehicle needs two engines or thrusters - one providing vertical force/lift like with conventional aircrafts while another provides horizontal movement as well by using its own propellant(rocket fuel) instead from external sources such us jet-fuel etc...
     - 6B
     - -

Example
========
Sabine is a lunar impact crater that forms a nearly matching pair with Ritter to the northwest. The two rims are separated by a distance of only a couple of kilometers. To the west is the bowl-shaped crater Schmidt, and farther to the north are Manners and Arago. Its diameter is 300km. It was named after Irish physicist and astronomer Edward Sabine.The outer rim of this crater is roughly circular and relatively featureless. The interior floor has a pair of small craterlets and a central rise. 

There is a ridge at the western edge of the floor that is concentric with the inner wall. **This part of the wall has been eaten away by impacts, but it remains as a broad raised ring along the southern side. The remaining interior floor is rough in appearance.The floor has been partially flooded by lava, forming small craterlets and ridges on its surface. There are no other features of interest on this floor, although there appear to be some small faults that might run through the interior floor.**

In To A Rocky Moon, lunar geologist Don E. Wilhelms summarized: They are identical twins in morphology and size (29-30 km).  They lack radial rim ejecta and secondary craters despite their apparent youth.  They are positioned at the presumably active edge of a mare.  They are even aligned along graben, the Hypatia rilles.  Most significant, they lack deep floors recognized since the days of Gilbert as diagnostic of impacts.  However, **after the Apollo landings were complete, it was realized that" the absence of impact-shaped features was an artifact of sampling bias: their rays reach up to 35 km from each crater and would have been sampled by the lunar orbiters. These craters are now thought to be volcanic in origin, as evidenced by their proximity to the Rimae Hypatia. The most probable scenario is that they were created by a fissure eruption of lava with high viscosity which then cooled rapidly and was thus hardened before it could drain away from the crater walls.↑ Wilhelms, Don E. (April 1987). To A Rocky Moon: A History of Lunar Exploration. University of Arizona Press. ISBN 0-8165-1121-4.**


Notes
========

During development of the system code from 
`this repository <https://github.com/Coolcumber/inpladesys>`_ was used
