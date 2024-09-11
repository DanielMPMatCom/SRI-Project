# Hibridación de Técnicas en Sistemas de Recomendación: Ventajas del Enfoque Probabilístico y Expansión del Algoritmo NBCF

![Logo](report/assets/logo.jpeg)

## Autores

- **Daniel Machado Pérez** - [@DanielMPMatCom](https://github.com/DanielMPMatCom)
- **Osvaldo R. Moreno Prieto** - [@Val020213](https://github.com/Val020213)
- **Daniel Toledo Martínez** - [@Phann020126](https://github.com/Phann020126)

## Descripción del Problema

En la era digital contemporánea, la sobrecarga de información es uno de los principales desafíos a los que se enfrentan los usuarios al interactuar con plataformas que ofrecen una gran cantidad de contenidos, como servicios de streaming, bibliotecas digitales, y tiendas de comercio electrónico. Los sistemas de recomendación han surgido como una solución eficaz a este problema, permitiendo a los usuarios descubrir y acceder a productos que se alinean con sus intereses y preferencias.

Dentro de los enfoques más prominentes en el desarrollo de sistemas de recomendación, el **filtrado colaborativo** se ha consolidado como uno de los más efectivos. Sin embargo, los métodos tradicionales, como la **factorización matricial**, aunque altamente precisos, presentan limitaciones en términos de interpretabilidad, lo que dificulta la explicación de las recomendaciones a los usuarios.

Para superar estas limitaciones, se ha propuesto el uso de enfoques **probabilísticos** como el **Naive Bayes Collaborative Filtering (NBCF)**, que no solo iguala o supera la precisión de la factorización matricial, sino que también mejora significativamente la capacidad del sistema para explicar las recomendaciones generadas. Además, la expansión del algoritmo NBCF para realizar recomendaciones a grupos de usuarios representa un avance significativo en la personalización colectiva de contenidos, un área que aún requiere un mayor desarrollo dentro del campo.

Este proyecto tiene como objetivo la implementación y evaluación de un sistema de recomendación híbrido que combina técnicas probabilísticas y de filtrado colaborativo, con un enfoque especial en la adaptabilidad y explicabilidad de las recomendaciones. Se basa en el trabajo previo desarrollado en varias tesis doctorales y artículos académicos, que exploran las ventajas de los enfoques probabilísticos , proporcionando una base sólida para la investigación y desarrollo de sistemas de recomendación más robustos y adaptativos. Siguiendo la sugerencia de una de las tesis analizadas, se propondrá un método para expandir el NBCF y realizar recomendaciones a grupos de usuarios combinándolo con una técnica existente en la literatura (NBP).

Se utilizará el dataset **FilmTrust**, ampliamente reconocido en la comunidad de investigación, para validar el rendimiento del sistema propuesto y compararlo con enfoques tradicionales.

## Requerimientos

Para ejecutar este proyecto, se requiere tener instalado **Python 3.8** o superior y las siguientes librerías:

- [streamlit](https://streamlit.io/): `pip install streamlit`
- [numpy](https://numpy.org/) : `pip install numpy`
- [pandas](https://pandas.pydata.org/) : `pip install pandas`
- [matplotlib](https://matplotlib.org/) : `pip install matplotlib`
- [time](https://docs.python.org/3/library/time.html) : `pip install time`
- [random](https://docs.python.org/3/library/random.html) : `pip install random`
  
