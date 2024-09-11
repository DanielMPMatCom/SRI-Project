import streamlit as st


def main():

    st.write(
        "# Hibridación de Técnicas en Sistemas de Recomendación: Ventajas del Enfoque Probabilístico y Expansión del Algoritmo NBCF"
    )

    st.sidebar.info("Seleccione una sección para continuar")
    st.sidebar.write(
        """
        ## Tabla de Contenidos
        1. [Descripción del problema](#f640d096)
        2. [Requerimientos](#requerimientos)
        3. [APIs utilizadas](#apis-utilizadas)
        4. [Modo de uso](#modo-de-uso)
        """
    )

    left_co, right_co = st.columns(2)
    with left_co:
        st.image(
            "./report/assets/logo.jpeg",
            caption="Proyecto Investigativo de SRI. Facultad de Matemática y Ciencia de la Computación. Universidad de La Habana. 2024",
            use_column_width=True,
        )
    with right_co:

        st.divider()
        st.write(
            "## Visite el repositorio en [GitHub](https://github.com/DanielMPMatCom/SRI-Project.git)"
        )
        st.code(
            "git clone https://github.com/DanielMPMatCom/SRI-Project.git",
            language="shell",
        )
        st.divider()
        st.markdown(
            """          
        ## Autores

        - **Daniel Machado Pérez** - [@DanielMPMatCom](https://github.com/DanielMPMatCom)
        - **Osvaldo R. Moreno Prieto** - [@Val020213](https://github.com/Val020213)
        - **Daniel Toledo Martínez** - [@Phann020126](https://github.com/Phann020126)
        """
        )

        st.divider()
        st.write("## Tabla de Contenidos")
        st.write(
            """
        1. [Descripción del problema](#sobre-el-problema)
        2. [Requerimientos](#requerimientos)
        3. [APIs utilizadas](#apis-utilizadas)
        4. [Modo de uso](#modo-de-uso)
        """
        )

        st.divider()

    st.write("# Sobre el problema")
    st.write(
        """
        En la era digital contemporánea, la sobrecarga de información es uno de los principales desafíos a los que se enfrentan los usuarios al interactuar con plataformas que ofrecen una gran cantidad de contenidos, como servicios de streaming, bibliotecas digitales, y tiendas de comercio electrónico. Los sistemas de recomendación han surgido como una solución eficaz a este problema, permitiendo a los usuarios descubrir y acceder a productos que se alinean con sus intereses y preferencias.

        Dentro de los enfoques más prominentes en el desarrollo de sistemas de recomendación, el **filtrado colaborativo** se ha consolidado como uno de los más efectivos. Sin embargo, los métodos tradicionales, como la **factorización matricial**, aunque altamente precisos, presentan limitaciones en términos de interpretabilidad, lo que dificulta la explicación de las recomendaciones a los usuarios.

        Para superar estas limitaciones, se ha propuesto el uso de enfoques **probabilísticos** como el **Naive Bayes Collaborative Filtering (NBCF)**, que no solo iguala o supera la precisión de la factorización matricial, sino que también mejora significativamente la capacidad del sistema para explicar las recomendaciones generadas. Además, la expansión del algoritmo NBCF para realizar recomendaciones a grupos de usuarios representa un avance significativo en la personalización colectiva de contenidos, un área que aún requiere un mayor desarrollo dentro del campo.

        Este proyecto tiene como objetivo la implementación y evaluación de un sistema de recomendación híbrido que combina técnicas probabilísticas y de filtrado colaborativo, con un enfoque especial en la adaptabilidad y explicabilidad de las recomendaciones. Se basa en el trabajo previo desarrollado en varias tesis doctorales y artículos académicos, que exploran las ventajas de los enfoques probabilísticos , proporcionando una base sólida para la investigación y desarrollo de sistemas de recomendación más robustos y adaptativos. Siguiendo la sugerencia de una de las tesis analizadas, se propondrá un método para expandir el NBCF y realizar recomendaciones a grupos de usuarios combinándolo con una técnica existente en la literatura (NBP).

        Se utilizará el dataset **FilmTrust**, ampliamente reconocido en la comunidad de investigación, para validar el rendimiento del sistema propuesto y compararlo con enfoques tradicionales.
        """
    )

    st.divider()
    st.write("# Requerimientos")
    st.write(
        """
        Para ejecutar este proyecto, se requiere tener instalado **Python 3.8** o superior y las siguientes librerías:
        - [streamlit](https://streamlit.io/): `pip install streamlit`
        - [numpy](https://numpy.org/) : `pip install numpy`
        - [pandas](https://pandas.pydata.org/) : `pip install pandas`
        - [matplotlib](https://matplotlib.org/) : `pip install matplotlib`
        - [time](https://docs.python.org/3/library/time.html) : `pip install time`
        - [random](https://docs.python.org/3/library/random.html) : `pip install random`
        """
    )
    st.divider()
    st.write("# APIs utilizadas")
    st.write(
        """
        No se utilizan APIs en este proyecto. En cambio se utiliza el framework de Streamlit para la creación de la interfaz gráfica.
        """
    )
    st.divider()
    st.write("# Modo de uso")
    st.write(
        """
        Para ejecutar la aplicación, se debe abrir una terminal en la carpeta raíz del proyecto y ejecutar el siguiente comando:
        ```bash startup.sh```
        """
    )
    st.code(
        "bash startup.sh",
        language="shell",
    )
    st.write(
        """
        Esto iniciará el servidor de Streamlit y abrirá la aplicación en el navegador web predeterminado. A partir de ahí, se puede interactuar con la aplicación y explorar las diferentes funcionalidades y visualizaciones proporcionadas.
        """
    )


if __name__ == "__main__":

    st.set_page_config(
        page_title="SRI",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
