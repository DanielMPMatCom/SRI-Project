import streamlit as st


def main():
    st.sidebar.info("Seleccione una sección para continuar")
    
    with st.sidebar:
        st.write("## Tabla de Contenidos")
        st.write(
            """
            1. [**Introducción**](#1-introduction)
            2. [**Estado del Arte**](#2-estado-del-arte)
            3. [**Algoritmo NBCF**](#3-algoritmo-nbcf)
            4. [**Expansión del Algoritmo NBCF**](#1aefdd70)
            5. [**Evaluación de los Resultados**](##9a2ccae1)
            6. [**Conclusiones**](#6-conclusiones)
            7. [**Bibliografía**](#287afcd5)
            """
        )

    st.write(
        "## Hibridación de técnicas de Sistemas de Recomendación. Ventajas del enfoque probabilístico en comparación con la factorización matricial. Implementación del algoritmo Naive Bayes Collaborative Filtering y su expansión para realizar recomendaciones a grupos de usuarios."
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.write("## Resumen")
        st.write(
            """El presente trabajo explora la hibridación de 
        técnicas en sistemas de recomendación, centrándose 
        en las ventajas del enfoque probabilístico frente a 
        la factorización matricial en cuanto a filtrado colaborativo. Se destaca cómo el 
        enfoque probabilístico, al proporcionar una 
        representación explícita de las incertidumbres, 
        mejora la interpretabilidad y explicación de las 
        recomendaciones generadas. En particular, 
        se implementa el algoritmo **Naive Bayes Collaborative Filtering** (NBCF), 
        que combina la simplicidad del modelo **Naive Bayes** 
        con el poder del filtrado colaborativo, 
        permitiendo recomendaciones precisas y explicativas. 
        Además, se expande este algoritmo para adaptarse 
        a la recomendación a grupos de usuarios, 
        abordando un área clave en la personalización 
        colectiva de contenidos. Los resultados demuestran 
        que el enfoque probabilístico no solo ofrece una 
        alternativa robusta a la factorización matricial, 
        sino que también potencia la capacidad del sistema 
        para ofrecer recomendaciones personalizadas y 
        comprensibles, tanto a individuos como a grupos."""
        )
       
        st.caption(
            """**Palabras claves:** **Sistemas de Recomendación** (RS), **Filtrado Colaborativo** (CF), **Enfoque Probabilístico**, **Factorización Matricial**, **Naive Bayes Collaborative Filtering** (NBCF), **Naive Pooling** (NBP)."""
        )
    with right_column:
        st.info("Nota: Consulte el informe en formato pdf, para todos los detalles.")
        st.write("## Tabla de Contenidos")
        st.write(
            """
            1. [**Introducción**](#1-introduction)
            2. [**Estado del Arte**](#2-estado-del-arte)
            3. [**Algoritmo NBCF**](#3-algoritmo-nbcf)
            4. [**Expansión del Algoritmo NBCF**](#1aefdd70)
            5. [**Evaluación de los Resultados**](##9a2ccae1)
            6. [**Conclusiones**](#6-conclusiones)
            7. [**Bibliografía**](#287afcd5)
            """
        )

    st.divider()

    st.write("## 1. Introduction")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write("### Descripción del tema y Técnicas de Recomendación")
        st.write(
            """
            Los sistemas de recomendación se han consolidado como 
            herramientas esenciales en la personalización de 
            contenidos en diversas plataformas digitales, 
            desde servicios de **streaming** hasta comercio electrónico. 
            Estos sistemas tienen como objetivo filtrar grandes 
            volúmenes de información y presentar a los usuarios 
            elementos relevantes según sus preferencias. 
            Entre las técnicas de recomendación más utilizadas, 
            se destacan el filtrado colaborativo, el filtrado 
            basado en contenido, el filtrado demográfico y los enfoques híbridos que 
            combinan estos métodos.
            \n

            El filtrado colaborativo, en particular, 
            ha sido ampliamente adoptado debido a su capacidad 
            para identificar patrones de comportamiento entre 
            usuarios y ofrecer recomendaciones basadas en 
            similitudes en sus interacciones previas. 
            Este enfoque se puede implementar mediante técnicas 
            basadas en memoria, que utilizan directamente las 
            interacciones pasadas de los usuarios, o mediante 
            técnicas basadas en modelos, que crean representaciones 
            abstractas de las relaciones entre usuarios e ítems.
                 """
        )
        st.write("### Enfoques de Filtrado Colaborativo Basado en Modelos")
        st.write(
            """
            Dentro del filtrado colaborativo basado en modelos, 
            dos enfoques destacan por su eficacia y popularidad: 
            la factorización matricial y los modelos probabilísticos. 
            La factorización matricial, como lo demuestra el 
            algoritmo de descomposición en valores singulares (SVD), 
            es una técnica poderosa para descomponer la matriz de 
            interacciones usuario-ítem en factores latentes, 
            permitiendo predicciones precisas de las preferencias 
            de los usuarios. No obstante, su principal limitación 
            radica en la falta de interpretabilidad de los factores 
            latentes, lo que dificulta la explicación de las 
            recomendaciones generadas.\n

            En contraste, los modelos probabilísticos, 
            como el **Naive Bayes Collaborative Filtering** (NBCF), 
            ofrecen una alternativa que, si bien puede alcanzar 
            niveles de precisión similares a los de la 
            factorización matricial, presenta la ventaja adicional 
            de proporcionar interpretaciones más claras de las 
            recomendaciones. El enfoque probabilístico permite 
            modelar explícitamente la incertidumbre en las 
            preferencias de los usuarios, lo que facilita la 
            explicación del porqué de cada recomendación."""
        )
    with col_2:
        st.write("### Antecedentes y Justificación")
        st.write(
            """
            La elección del enfoque probabilístico como base de 
            esta investigación se sustenta en los hallazgos 
            presentados en la tesis doctoral titulada 
            "Sistema recomendador híbrido basado en modelos 
            probabilísticos". Esta tesis profundiza en las ventajas 
            de utilizar modelos probabilísticos en sistemas de 
            recomendación, destacando su capacidad para superar 
            las limitaciones de los enfoques tradicionales de 
            factorización matricial. Además, se presenta una 
            implementación del algoritmo NBCF, que ha mostrado 
            resultados prometedores en términos de precisión y 
            explicabilidad.\n

            Sin embargo, un área poco explorada en esta tesis es 
            la capacidad de estos modelos para realizar 
            recomendaciones a grupos de usuarios, una 
            característica esencial en contextos como la 
            recomendación de contenido para familias, grupos de 
            amigos o equipos de trabajo. Esta investigación se 
            propone expandir el algoritmo NBCF, siguiendo las 
            recomendaciones de la tesis doctoral, para adaptarlo 
            a la recomendación grupal, un desafío significativo 
            en la personalización colectiva.\n

            Para extender el algoritmo NBCF a la recomendación de 
            grupos, se adoptará la idea del **Naive Pooling** (NBP) 
            propuesta en el artículo **"Extended Naïve Bayes for 
            Group Based Classification"**[5]. El método NBP utiliza 
            las probabilidades calculadas (en nuestro caso las ya 
            con NBCF) y las 
            combina de manera que se maximice la probabilidad 
            conjunta para un grupo de usuarios, permitiendo así 
            una clasificación efectiva de grupos con una alta 
            coherencia en las recomendaciones. Este enfoque se 
            considera particularmente adecuado para garantizar 
            que todos los miembros del grupo reciban 
            recomendaciones que reflejen tanto las preferencias 
            individuales como las del colectivo.
            """
        )

        st.write("### Dataset Seleccionado")
        st.write(
            """
            Para la evaluación de la implementación y expansión 
            del algoritmo NBCF, se ha seleccionado el dataset 
            **filmtrust**[6], un conjunto de datos ampliamente utilizado 
            en la investigación de sistemas de recomendación. 
            FilmTrust contiene miles de calificaciones de 
            películas proporcionadas por usuarios, lo que lo 
            convierte en un recurso valioso para el análisis y 
            desarrollo de modelos de recomendación. 
            La riqueza y diversidad del dataset permiten probar 
            la eficacia de los algoritmos en un entorno 
            cercano a escenarios del mundo real. 
            Este dataset fue uno de los utilizados en la tesis antes mencionada.
            """
        )

    st.divider()
    col_3, col_4 = st.columns(2)

    with col_3:
        st.write("### Estructura del Trabajo")
        st.write(
            """
            El presente informe se estructura en varias secciones 
            que desarrollan en detalle los diferentes aspectos de 
            la investigación:
            """
        )
        st.write(
            "1. **Estado del Arte**: Se revisa la literatura existente sobre técnicas de recomendación, con un enfoque en el filtrado colaborativo basado en modelos."
        )
        st.write(
            "2. **Algoritmo NBCF**: Se describe la implementación del algoritmo NBCF y su funcionamiento."
        )
        st.write(
            "3. **Expansión del Algoritmo NBCF**: Se presenta la adaptación del NBCF para realizar recomendaciones a grupos de usuarios, detallando las modificaciones realizadas."
        )
        st.write(
            "4. **Evaluación de los Resultados**: Se analizan los resultados obtenidos tras la implementación y se comparan con enfoques tradicionales."
        )
        st.write(
            "5. **Conclusiones**: Se resumen los hallazgos más relevantes de la investigación y se sugieren posibles direcciones futuras."
        )

        st.write("## 2. Estado del Arte")
        st.write(
            """
            Los sistemas de recomendación se han vuelto 
            indispensables en la era de la información, 
            donde los usuarios requieren herramientas que les 
            permitan descubrir contenidos relevantes de manera 
            eficiente. Existen diversas técnicas para abordar 
            este problema, cada una con sus propias ventajas y 
            limitaciones. Entre las más destacadas están el 
            filtrado colaborativo, el filtrado basado en contenido, 
            el filtrado demográfico y los enfoques híbridos. 
            En esta sección, se revisarán las principales técnicas 
            de recomendación, con un enfoque particular en el 
            filtrado colaborativo y sus variantes basadas en 
            modelos probabilísticos.
            """
        )
        st.write("### Técnicas de Recomendación")
        st.write(
            """
            * **Filtrado Colaborativo**: Este enfoque se basa en la idea de que los usuarios que han compartido preferencias similares en el pasado probablemente coincidan en sus elecciones futuras. El filtrado colaborativo puede implementarse a través de dos métodos: basado en memoria y basado en modelos. Los enfoques basados en memoria, como el algoritmo de k vecinos más cercanos (k-NN), utilizan directamente la matriz de interacciones usuario-ítem para realizar recomendaciones. Por otro lado, los enfoques basados en modelos, que incluyen técnicas como la factorización matricial y los modelos probabilísticos, construyen un modelo predictivo a partir de los datos disponibles, ofreciendo recomendaciones más precisas y escalables. [2]
            """
        )
        st.write(
            """
            * **Filtrado Basado en Contenido**: Este método recomienda ítems a un usuario en función de la similitud entre los ítems que ha consumido previamente y otros ítems disponibles. A diferencia del filtrado colaborativo, se basa en las características de los ítems, como el género, el director o los actores en el caso de películas. [2]
            """
        )
        st.write(
            """
            * **Filtrado Demográfico**: Aunque menos utilizado en comparación con los métodos anteriores, el filtrado demográfico se basa en las características personales de los usuarios, tales como su edad, género o ubicación. Este enfoque supone que usuarios con características demográficas similares tienden a compartir preferencias similares. Si bien puede ser útil para ciertos contextos, su efectividad suele ser menor, ya que no tiene en cuenta las interacciones individuales entre usuarios e ítems. [2]
            """
        )
        st.write(
            """
            * **Enfoques Híbridos**: Estos combinan dos o más de las técnicas mencionadas para mejorar la precisión y superar las limitaciones inherentes a cada uno de los métodos. Por ejemplo, un sistema híbrido puede combinar el filtrado colaborativo con el filtrado basado en contenido para ofrecer recomendaciones más completas, tanto en precisión como en diversidad. [2]
            """
        )
    with col_4:
        st.write("### Filtrado Colaborativo Basado en Modelos")
        st.write(
            """
            El filtrado colaborativo basado en modelos ha 
            demostrado ser especialmente eficaz en sistemas de 
            recomendación a gran escala. Entre los enfoques más 
            destacados se encuentran la factorización matricial y
            los modelos probabilísticos."""
        )

        st.write(
            """
            * **Factorización Matricial**: Esta técnica ha demostrado ser una de las más efectivas para el filtrado colaborativo. En la factorización matricial, la matriz de interacciones usuario-ítem se descompone en dos matrices de menor dimensión que representan factores latentes tanto para los usuarios como para los ítems. Estos factores latentes permiten realizar predicciones sobre las preferencias de los usuarios al capturar características no observadas explícitamente. Aunque la factorización matricial, especialmente con algoritmos como la descomposición en valores singulares (SVD), ha demostrado ser muy precisa, su principal limitación radica en la falta de interpretabilidad. Los factores latentes no siempre son comprensibles o intuitivos para los usuarios, lo que dificulta la explicación de las recomendaciones. [4]            
            """
        )
        st.write(
            """
            * **Modelos Probabilísticos**: En contraste con la factorización matricial, los modelos probabilísticos proporcionan una representación más clara de las incertidumbres en las preferencias de los usuarios. Uno de los enfoques más representativos es el Naive Bayes Collaborative Filtering (NBCF), que combina la simplicidad del modelo de Naive Bayes con la estructura del filtrado colaborativo. Este enfoque permite una mayor interpretabilidad, ya que ofrece una explicación probabilística de las recomendaciones. Además, el NBCF ha mostrado ser altamente adaptable a diferentes escenarios, permitiendo la incorporación de nuevas variables sin comprometer su eficiencia. [4]
            """
        )

        st.write("### Desarrollo en las Tesis y Papers")
        st.write(
            """
            El enfoque probabilístico ha sido objeto de un 
            estudio detallado en la tesis doctoral titulada 
            **"Sistema recomendador híbrido basado en modelos 
            probabilísticos"**[4]. En esta tesis, se aborda la 
            integración de modelos probabilísticos dentro de 
            sistemas de recomendación híbridos, destacando cómo 
            estos modelos no solo permiten una mayor precisión, 
            sino que también aportan una capa de interpretabilidad 
            que los métodos de factorización matricial no ofrecen. 
            El autor propone un enfoque híbrido que combina los 
            beneficios del filtrado colaborativo basado en modelos 
            probabilísticos con técnicas de filtrado basado en 
            contenido.

            En el paper titulado **"Extended Naïve 
            Bayes for Group Based Classification"**[5], los autores 
            presentan una extensión del clásico modelo 
            **Naive Bayes}, adaptándolo para su aplicación 
            en la clasificación basada en grupos. Uno de los 
            enfoques tratados, 
            denominado **Naive Pooling** (NBP), se centra en la 
            agregación de probabilidades individuales para 
            generar una probabilidad conjunta que permita la 
            clasificación efectiva de grupos de usuarios. La 
            metodología propuesta combina las probabilidades 
            individuales de cada miembro del grupo para maximizar 
            la coherencia y relevancia de la clasificación final. 
            Este método resulta particularmente útil en contextos 
            donde se deben generar recomendaciones o decisiones 
            que reflejen tanto las preferencias individuales 
            como la dinámica grupal. La capacidad de NBP para 
            mantener la simplicidad del modelo Naive Bayes, 
            al tiempo que amplía su aplicabilidad a escenarios 
            grupales, lo convierte en una herramienta poderosa 
            para la personalización colectiva en sistemas de 
            recomendación.

            Además, en el paper **"A Collaborative Filtering 
            Approach Based on Naive Bayes Classifier"**[1], 
            se profundiza en la implementación del NBCF y se 
            demuestra su viabilidad como alternativa a los 
            métodos tradicionales de filtrado colaborativo. 
            Los resultados obtenidos en este estudio muestran 
            que el NBCF puede igualar o superar el rendimiento de 
            la factorización matricial, especialmente en datasets 
            donde la interpretabilidad es tan importante como la 
            precisión.

            Finalmente, el trabajo **"Hybrid Collaborative Filtering 
            Based on Users' Rating Behavior"**[3] presenta un enfoque 
            híbrido que integra el comportamiento de valoración 
            de los usuarios con el filtrado colaborativo. 
            Este enfoque tiene una relevancia particular para 
            nuestro proyecto, ya que permite ajustar las 
            recomendaciones no solo en función de las 
            interacciones pasadas, sino también considerando la 
            manera en que los usuarios valoran los ítems, lo que 
            aporta una capa adicional de personalización.
            """
        )
    st.divider()

    col_5, col_6 = st.columns(2)

    with col_5:
        st.write("## 3. Algoritmo NBCF")
        st.write("### Descripción del Algoritmo")
        st.write(
            """
    El algoritmo **Naive Bayes Collaborative 
    Filtering** (NBCF) es una técnica innovadora dentro 
    del campo de los sistemas de recomendación 
    colaborativos. A diferencia de otros enfoques, 
    como la factorización matricial, el NBCF aprovecha la 
    simplicidad y efectividad del clasificador **Naive Bayes**
    para predecir las preferencias de los usuarios en 
    función de sus interacciones anteriores con ítems. 
    Este método considera la probabilidad de que un 
    usuario asigne una cierta calificación a un ítem, 
    basándose en las calificaciones previas tanto del 
    usuario como de otros usuarios con comportamientos 
    similares."""
        )

        st.write("### Formulación Matemática del Algoritmo NBCF")
        st.write(
            """
    El algoritmo NBCF se basa en la combinación de dos 
    enfoques principales: basado en usuarios y basado 
    en ítems. En cada uno de estos enfoques, se calcula la 
    probabilidad a priori de que un usuario califique un 
    ítem con un valor específico, y posteriormente se 
    calcula el **likelihood** para ajustar esta 
    probabilidad en función de las calificaciones 
    observadas."""
        )
        st.write(
            "* **Enfoque Basado en Usuarios:** La probabilidad a priori y el likelihood se calculan de acuerdo con los ítems que cada usuario ha votado.[4]"
        )

        st.write(
            "* **Enfoque Basado en Ítems:**  La probabilidad a priori y el likelihood se calculan de acuerdo con los votos que cada ítem ha recibido.[4]"
        )

        st.write(
            "* **Enfoque Híbrido:** Se combinan los enfoques basados en usuarios e ítems para mejorar la precisión del modelo.[4]"
        )

        st.write(
            """
Para el desarrollo de cada uno de estos enfoques se 
utiliza los siguientes conceptos de probabilidades:"""
        )

        st.write(
            """* **Probabilidad A Priori:** En el enfoque basado en ítems, se calcula la probabilidad a priori de que un usuario $u$ asigne una calificación $y$ a un ítem $i$, denotado como $P(r_u = y)$. De manera análoga, en el enfoque basado en usuarios, se calcula la probabilidad de que un ítem $i$ reciba una calificación $y$ de cualquier usuario $u$, denotado como $P (r_i = y)$.[4]"""
        )

    with col_6:
        st.write(
            """
        * **Likelihood:** El likelihood ajusta la probabilidad a priori mediante la consideración de la información adicional disponible en las calificaciones observadas. Para el enfoque basado en ítems, esto se expresa como $P(r_v = k | r_u = y)$, que representa la probabilidad de que otro usuario $v$ califique con $k$ un ítem que ha sido calificado con $y$ por el usuario $u$. Similarmente, para el enfoque basado en usuarios, se calcula el likelihood correspondiente $P(r_j = k|r_i = y)$[4]
        """
        )

        st.write(
            """
        * **Combinación de Enfoques:** En el enfoque híbrido, se integran las probabilidades obtenidas de los enfoques basados en usuarios y en ítems, proporcionando un modelo más robusto y preciso para la predicción de calificaciones.[4]
        """
        )

        st.info(
            """Consulte el informe en formato pdf para obtener más detalles sobre la formulación matemática del algoritmo NBCF."""
        )

        st.write("### Algoritmo NBCF: Implementación Paso a Paso")
        st.write(
            """
        El algoritmo NBCF se implementa de manera iterativa,
        asegurando la eficiencia computacional mediante
        técnicas de memorización que permiten evitar el
        recálculo innecesario de probabilidades. A continuación
        se describen los pasos del algoritmo:
        """
        )

        st.write(
            """
        1. **Inicialización**: Se inicializan las probabilidades a priori y los contadores utilizados en el cálculo de likelihoods.
        """
        )

        st.write(
            """
        2. **Iteración sobre Usuarios e Ítems**: Para cada usuario, se calcula la probabilidad de cada calificación posible basada en las calificaciones observadas para los ítems que ha evaluado. De manera similar, se calcula para cada ítem la probabilidad de recibir una calificación específica basada en las calificaciones anteriores recibidas.
        """
        )

        st.write(
            """
        3. **Almacenamiento de Resultados**: Los valores calculados se almacenan para ser utilizados posteriormente en la predicción de nuevas calificaciones, evitando la necesidad de recalcular durante la fase de predicción.
        """
        )

        st.write(
            """
        Este enfoque garantiza que el NBCF no solo sea 
        eficiente, sino que también se adapte bien a problemas 
        de gran escala, manteniendo una complejidad 
        computacional similar a la de otros métodos avanzados, 
        como la factorización matricial[4]. """
        )

    st.divider()
    col_7, col_8 = st.columns(2)

    with col_7:
        st.write("### Resultados Experimentales y Comparativa")
        st.write(
            """
        El algoritmo NBCF ha demostrado su eficacia en múltiples conjuntos de datos públicos (MovieLens, FilmTrust, Yahoo, BookCrossing)[4], superando en varias métricas clave a los métodos de referencia más utilizados en el campo:
        """
        )
        st.write(
            """
        * **Error Medio Absoluto (MAE)**
        * **Precisión y Recall**
        * **Ganancia acumulada descontada normalizada (nDCG)"""
        )
        st.write(
            """
        Se compararon los siguientes enfoques:
        """
        )
        st.write(
            """
        * **NBCF (usuario)**
        * **NBCF (ítem)**
        * **NBCF (híbrido)**
        * **BNMF**
        * **GGM**
        * **INBM**
        * **Bi-CF**
        * **NMF***"""
        )

        st.write("Los resultados fueron los siguientes:")
        st.write(
            "- **MovieLens**: El enfoque híbrido de NBCF ha mostrado mejoras significativas en medidas de MAE, precisión y recall, así como el enfoque basado en ítems fue mejor en la nDCG en comparación con enfoques tradicionales.\n"
        )
        st.write(
            "- **FilTrust**: el MAE de NBCF (híbrido) logra mejores resultados que los otros dos enfoques propuestos, mientras que la precisión y recall son mejores con NBCF (ítems) y NBCF (usuario). Por otro lado, cuando aumenta el número de recomendaciones, nDCG es mejor con el enfoque NBCF (híbrido).\n"
        )
        st.write(
            "- **Yahoo**: nDCG es mejor en NBCF (híbrido) en comparación con NBCF (ítem) y NBCF (usuario). Además, la precisión y el recall de los tres enfoques propuestos presentan un resultado casi similar entre ellos. Así mismo, hay una superioridad lograda en MAE de NBCF (híbrido) con respecto a los otros enfoques propuestos.\n"
        )
        st.write(
            "- **BookCrossing**: NBCF (híbrido) y NBCF (ítem) proveen mejores resultados para nDCG en comparación con los métodos de línea base de CF. A diferencia de otros conjuntos de datos en BookCrossing las métricas de precisión y recall son mejores para los métodos GGM, INBM y Bi-CF. Sin embargo muestran una mejora con respecto a los métodos BNMF y NMF. NBCF (híbrido) se muestra superior al resto de los enfoques en cuanto al MAE.\n"
        )

        st.markdown("## Conclusión")
        st.markdown(
            """
            El algoritmo NBCF representa una mejora significativa 
            en el ámbito de los sistemas de recomendación 
            colaborativos, combinando la simplicidad del 
            clasificador *Naive Bayes* con técnicas de filtrado 
            colaborativo para ofrecer recomendaciones precisas y 
            eficaces. Su capacidad para integrar múltiples 
            enfoques y adaptarse a diferentes escenarios lo 
            convierte en una herramienta valiosa para la mejora de 
            la experiencia del usuario en plataformas de 
            recomendación.
            """
        )
    with col_8:
        st.markdown("## 4. Expansión del Algoritmo NBCF")

        st.markdown("### Algoritmo NBP. Formulación Matemática")
        st.markdown(
            """
            En los sistemas de recomendación modernos, la 
            capacidad de realizar recomendaciones efectivas 
            no solo a usuarios individuales, sino también a grupos, 
            se ha vuelto un aspecto crucial. Contextos como la 
            recomendación de contenido para familias, equipos de 
            trabajo o grupos de amigos demandan un enfoque que 
            pueda considerar y equilibrar las preferencias 
            individuales dentro de un colectivo, maximizando la 
            satisfacción general del grupo. En respuesta a esta 
            necesidad, surge el algoritmo NBP, una extensión del 
            modelo clásico de *Naive Bayes* adaptado 
            específicamente para la clasificación y recomendación 
            a grupos de usuarios.

            El algoritmo se fundamenta en la misma premisa básica 
            que el *Naive Bayes*, es decir, la independencia 
            condicional de las características dadas las clases. 
            Sin embargo, lo que distingue a NBP es su capacidad 
            para combinar las probabilidades individuales de los 
            miembros de un grupo, produciendo una probabilidad 
            conjunta que guía la recomendación o clasificación 
            final para todo el grupo.
            """
        )
        st.markdown("### Ventajas y Desventajas de NBP")

        st.markdown("El algoritmo Naive Pooling presenta varias ventajas notables:")
        st.markdown(
            "- **Simplicidad y Eficiencia**: Al estar basado en el modelo *Naive Bayes*, NBP hereda la simplicidad computacional y la eficiencia del mismo, lo que permite su implementación en sistemas a gran escala sin requerir un costo computacional elevado."
        )
        st.markdown(
            "- **Interpretabilidad**: Una de las fortalezas del enfoque probabilístico es su capacidad para proporcionar una explicación clara de las recomendaciones basadas en probabilidades. Esto facilita la interpretación de por qué un grupo recibió una recomendación específica."
        )
        st.markdown(
            "- **Adaptabilidad**: NBP es altamente adaptable y puede integrarse con facilidad en sistemas de recomendación existentes que ya utilicen el enfoque *Naive Bayes* para recomendaciones individuales. Esto permite a los desarrolladores extender sus sistemas a la recomendación grupal sin necesidad de una reingeniería significativa."
        )

        st.markdown(
            "A pesar de sus ventajas, el algoritmo NBP también presenta algunas limitaciones que deben ser consideradas:"
        )
        st.markdown(
            "- **Suposición de Independencia**: Al igual que *Naive Bayes*, NBP asume que las preferencias de los usuarios dentro del grupo son condicionalmente independientes, lo cual puede no reflejar adecuadamente la realidad, donde las preferencias de los usuarios pueden estar correlacionadas."
        )
        st.markdown(
            "- **Equidad en la Recomendación**: NBP no tiene en cuenta explícitamente la equidad entre las preferencias individuales dentro del grupo. Es decir, podría favorecer las preferencias de algunos usuarios sobre otros, especialmente si las probabilidades individuales de ciertos miembros del grupo son mucho más altas que las de los demás."
        )
        st.markdown(
            "- **Escalabilidad con Grupos Grandes**: Aunque eficiente, a medida que el tamaño del grupo aumenta, la combinación de probabilidades puede llevar a situaciones donde las recomendaciones sean dominadas por usuarios con preferencias extremas, lo que podría reducir la diversidad y la satisfacción global del grupo."
        )

    st.divider()

    col_9, col_10 = st.columns(2)

    with col_9:
        st.markdown("## Aplicación del Algoritmo NBP en la Expansión de NBCF")
        st.markdown(
            """
            En el contexto de esta investigación, el algoritmo 
            *Naive Pooling* se integrará con el modelo 
            *Naive Bayes Collaborative Filtering* (NBCF). 
            Teniendo previamente calculados los valores de las 
            probabilidades para cada usuario con NBCF, ya sea con 
            su enfoque basado en usuario, basado en ítems o híbrido,
            se puede efectuar la fórmula de NBP y obtener el resultado 
            que se quiere. De esta forma, se extiende 
            la funcionalidad del NBCF para proporcionar 
            recomendaciones no solo a individuos, sino también a 
            grupos de usuarios, conservando la precisión y 
            explicabilidad del enfoque probabilístico. 
            Esta expansión permitirá que el sistema no solo 
            mantenga la calidad en las recomendaciones 
            individuales, sino que también pueda satisfacer las 
            necesidades de colectivos en situaciones donde la 
            personalización grupal es esencial.
            """
        )

        st.markdown("## Evaluación de los resultados")

        st.markdown(
            """
            En esta sección se presentará la evaluación del 
            rendimiento de la expansión del algoritmo NBCF para 
            la recomendación a grupos utilizando el algoritmo NBP. 
            Para ello, se ha seleccionado el dataset *FilmTrust*
            por sus características particulares y su amplio uso 
            en investigaciones de sistemas de recomendación.
            """
        )

        st.markdown("### Descripción del dataset *FilmTrust*")

        st.markdown(
            """
            El dataset *FilmTrust* contiene información 
            sobre la interacción de usuarios con películas, 
            lo cual lo convierte en un recurso valioso para el 
            análisis de sistemas de recomendación. 
            Sus características principales son las siguientes:
            """
        )

        st.markdown(
            """
            * **1508 usuarios**: Cada usuario ha 
            evaluado uno o varios ítems (películas) del conjunto.
            * **2071 ítems**: El conjunto de ítems representa las películas que los usuarios han visto y valorado.
            * **35494 votos**: Los usuarios han emitido un total de 35,494 calificaciones sobre las películas del conjunto.
            * **Escala de votación**: Las calificaciones originales varían entre 0.5 y 4, con incrementos de 0.5. Para simplificar la indexación y el procesamiento en la implementación del sistema, se ha multiplicado la escala por 2, de modo que las calificaciones oscilan entre 1 y 8, con incrementos de 1. Esto permite una mayor precisión y facilidad a la hora de manejar los datos.
            * **Sparsity del dataset**: El valor de sparsity (dispersión de los datos) de este conjunto es del 98.86%. Esto significa que la mayoría de las posibles combinaciones entre usuarios e ítems no tienen una calificación registrada, lo que introduce un desafío significativo para el modelo de recomendación. La sparsity es una medida que refleja el nivel de dispersión de los datos en una matriz de interacciones usuario-ítem, donde un valor elevado indica que muchas de las posiciones en la matriz están vacías, es decir, no contienen interacciones registradas. Este fenómeno es común en sistemas de recomendación, y requiere el uso de modelos capaces de manejar eficazmente la falta de datos.
            """
        )

        st.markdown("## Diseño del test")
        st.markdown(
            """
            Dado que no se encontró un dataset disponible que incluyera información sobre recomendaciones realizadas a grupos de usuarios, surgió la necesidad de diseñar un test propio que permitiera evaluar la efectividad del método propuesto para recomendaciones grupales. El enfoque adoptado utiliza el conjunto de datos de *FilmTrust*, donde se ha adaptado la estructura del dataset para generar grupos de usuarios con características comunes.
            """
        )

    with col_10:

        st.markdown("### Confección de los grupos de usuarios")
        st.markdown(
            """
            A partir de cada película en el dataset, se forma un grupo compuesto por todos los usuarios que le han asignado la misma calificación. Este proceso garantiza que los grupos compartan una opinión similar sobre el ítem en cuestión, lo que es relevante para medir la coherencia del sistema de recomendación en entornos colaborativos. Una vez conformados los grupos iniciales, de cada uno se selecciona aleatoriamente un subconjunto de usuarios, que constituirá uno de los grupos finales sobre los que se realizarán las evaluaciones.

            En particular, se pone énfasis en la evaluación de la eficacia del método para grupos que hayan otorgado altas calificaciones a las películas (valores de 6, 7 u 8 en la escala de votos) y bajas calificaciones (1, 2, 3). Esto permite analizar el rendimiento del algoritmo en escenarios donde existe un fuerte consenso positivo entre los usuarios, lo que representa un caso de uso frecuente en las recomendaciones de contenido audiovisual para grupos de amigos o familiares.

            Es válido señalar que de esta forma de armar los grupos, pueden existir intersecciones entre ellos, lo cual se ajusta a los escenarios reales y no provoca dificultades para el desempeño del método propuesto. La semilla para replicar los experimentos fue 42.
            """
        )

        st.markdown("### Procedimiento")
        st.markdown(
            """
            Una vez conformados los grupos de usuarios que han calificado de manera idéntica una película en particular, se procede a eliminar la información de los votos emitidos por esos usuarios hacia dicha película. Este paso es fundamental para simular un escenario en el que los miembros del grupo no han visto la película, lo cual nos permite evaluar la capacidad del sistema de realizar recomendaciones grupales efectivas.

            Es importante destacar que la razón de la selección de subconjuntos de usuarios de los grupos iniciales es evitar la pérdida completa de la información relacionada con los votos de la película. De esta manera, se conserva parte de la información del ítem en cuestión, lo que nos permite continuar utilizando el resto de las interacciones de los usuarios con otras películas en el proceso de recomendación.

            El objetivo de este procedimiento es evaluar si, bajo la suposición de que los usuarios del grupo no han visto la película, el sistema es capaz de predecir una calificación grupal similar a la calificación inicial. Dado que los miembros del grupo han otorgado una calificación uniforme a la película antes de eliminar esta información, se espera que el sistema de recomendación prediga un voto grupal consistente con las valoraciones originales. Además, al aplicar este procedimiento a películas que recibieron calificaciones altas (6, 7 u 8), se puede evaluar la capacidad del modelo para recomendar este tipo de películas en escenarios donde los usuarios no las hayan visto. De la misma forma al comprobar el rendimiento con las bajas calificaciones (1, 2, 3) se analiza su efectividad para no recomendar películas que no serían bien calificadas por los integrantes del grupo.
            """
        )

        st.markdown("### Resultados con votaciones altas (6, 7, 8)")
        st.markdown(
            """
            En esta sección se presentarán los resultados obtenidos al evaluar el rendimiento del método propuesto para grupos que han otorgado altas calificaciones (6, 7, 8) a las películas. Estos resultados permitirán analizar la capacidad del sistema de recomendación para predecir calificaciones grupales consistentes con las valoraciones iniciales, así como su efectividad para recomendar películas de alta calidad en escenarios donde los usuarios no las hayan visto.
            """
        )

    st.divider()

    col_11, col_12 = st.columns(2)

    with col_11:

        st.markdown("### Diferencias")
        st.markdown(
            "En la Figura 1, se muestran las diferencias entre las recomendaciones sugeridas por el sistema y las expectativas del usuario. Este gráfico permite identificar las desviaciones del sistema con respecto a las preferencias reales, lo que ofrece una visión clara de las áreas donde se puede mejorar la precisión de las recomendaciones."
        )
        st.image(
            "report/assets/Diferencias678.png",
            caption="Gráfico de diferencias entre las recomendaciones del sistema y las preferencias reales del usuario. Calificaciones 6, 7, 8.",
            use_column_width=True,
        )

        st.markdown(
            "El análisis de estas diferencias sugiere que, si bien el sistema tiende a alinearse con las preferencias de los usuarios en un número considerable de casos, persisten desviaciones en ciertos grupos."
        )

        st.markdown("### Diferencias absolutas")
        st.markdown(
            "La Figura 2 muestra las diferencias absolutas, es decir, el valor absoluto de la desviación entre las recomendaciones sugeridas y las expectativas del usuario. Esta representación visual permite observar cuán grandes son las desviaciones sin tener en cuenta la dirección (positiva o negativa) de la diferencia."
        )
        st.image(
            "report/assets/absolutas678.png",
            caption="Gráfico de diferencias absolutas entre las recomendaciones del sistema y las preferencias del usuario. Calificaciones 6, 7, 8.",
            use_column_width=True,
        )

        st.markdown(
            "Lo observado permite concluir que el sistema, en general, ofrece recomendaciones razonablemente precisas en la mayoría de los casos."
        )

    with col_12:

        st.markdown("## Tabla de Resultados")

        st.markdown(
            "En la Tabla 1 se presenta un resumen cuantitativo de los resultados. Se muestran las predicciones, valores reales, diferencias y diferencias absolutas para 25 grupos confeccionados y las películas seleccionadas. Los resultados completos pueden verse en el archivo **Grupos678.xlsx** en la ruta **report/assets/** a partir del directorio raíz del repositorio de GitHub."
        )

        st.image(
            "report/assets/tabla678.png",
            caption="Resumen cuantitativo de los resultados del sistema de recomendación. Calificaciones 6, 7, 8.",
            use_column_width=True,
        )

        st.markdown("### Otras métricas")

        st.markdown(
            "- **MSE (Error Cuadrático Medio): 0.657**: Este valor indica el promedio de los errores al cuadrado entre las predicciones del sistema y los valores reales. El MSE es sensible a errores grandes, ya que estos se amplifican al ser elevados al cuadrado. Un valor de 0.657 implica que, en promedio, las diferencias entre las recomendaciones del sistema y las preferencias reales de los usuarios son moderadamente bajas. Sin embargo, aún existen algunos errores significativos que deben ser corregidos para mejorar la precisión del sistema de recomendación."
        )
        st.markdown(
            "- **MAE (Error Medio Absoluto): 0.486**: Esta métrica refleja el error medio absoluto entre las predicciones y los valores reales, lo cual ofrece una interpretación más intuitiva de la precisión del modelo. En este caso, el valor de 0.486 indica que, en promedio, la desviación entre la predicción y las preferencias reales de los usuarios es de aproximadamente 0.49 unidades. Este valor es más fácil de interpretar que el MSE, y nos muestra que las predicciones del sistema son razonablemente precisas, aunque pueden ser perfeccionadas."
        )

    st.divider()

    col_13, col_14 = st.columns(2)

    with col_13:
        st.markdown("### Diferencias")
        st.markdown(
            "En la Figura 2, se muestran las diferencias entre las recomendaciones sugeridas por el sistema y las expectativas del usuario al analizar las películas con calificaciones bajas (1, 2, 3)."
        )
        st.image(
            "report/assets/Diferencias123.png",
            caption="Gráfico de diferencias entre las recomendaciones del sistema y las preferencias reales del usuario. Calificaciones 1, 2, 3.",
            use_column_width=True,
        )
        st.markdown(
            "El análisis indica que el método manifiesta un rendimiento no tan bueno a la hora de predecir las calificaciones de estas películas, pues en los experimentos siempre predijo valores superiores a los reales. Sin embargo, se observa que en algunos casos las diferencias son pequeñas."
        )

        st.markdown("### Diferencias absolutas")
        st.markdown(
            "La Figura 3 muestra las diferencias absolutas, que como en este caso todas las diferencias son negativas, se comporta como la gráfica anterior."
        )
        st.image(
            "report/assets/absolutas123.png",
            caption="Gráfico de diferencias absolutas entre las recomendaciones del sistema y las preferencias del usuario. Calificaciones 1, 2, 3.",
            use_column_width=True,
        )

    with col_14:

        st.markdown("## Tabla de Resultados")

        st.markdown(
            "En la Tabla 1 se presenta un resumen cuantitativo de los resultados para este conjunto de calificaciones. Los resultados completos pueden verse en el archivo **Grupos123.xlsx** en la ruta **report/assets/** a partir del directorio raíz del repositorio de GitHub."
        )

        st.image(
            "report/assets/tabla123.png",
            caption="Resumen cuantitativo de los resultados del sistema de recomendación. Calificaciones 1, 2, 3.",
            use_column_width=True,
        )

        st.markdown("### Otras métricas")

        st.markdown(
            "- **MSE (Error Cuadrático Medio): 9.222**: Este valor elevado del MSE indica que el sistema de recomendación está generando errores significativos al comparar las predicciones con los valores reales. Dado que el MSE eleva al cuadrado las diferencias entre los valores predichos y los valores reales, un valor de 9.222 refleja la presencia de desviaciones grandes en las predicciones. Este resultado es preocupante, ya que significa que, en algunos casos, el sistema está haciendo predicciones muy alejadas de lo que realmente deberían ser."
        )
        st.markdown(
            "- **MAE (Error Medio Absoluto): 2.778**: El MAE de 2.778 también es considerablemente alto, lo que sugiere que, en promedio, el sistema se está desviando casi 3 unidades en sus predicciones. Esto implica que las recomendaciones del sistema no están siendo lo suficientemente precisas para ajustarse a las expectativas de los usuarios, lo que puede resultar en una experiencia insatisfactoria para ellos."
        )

        st.markdown("## Limitaciones")

        st.markdown(
            "A pesar de los resultados obtenidos, el sistema de recomendación presenta algunas limitaciones que es importante destacar. Una de las principales dificultades observadas es la capacidad del modelo para realizar predicciones precisas en películas que se espera tengan calificaciones bajas. Este tipo de películas tienden a recibir menos atención de los usuarios, lo que genera un sesgo en los datos disponibles para su análisis y, en consecuencia, afecta la precisión de las predicciones. Esto es especialmente notable cuando se trabaja con conjuntos de datos que no cuentan con suficientes ejemplos de estas películas, dificultando la generalización de las recomendaciones."
        )

        st.markdown(
            "Otra limitación importante se refiere a la velocidad de la implementación. Durante la primera ejecución del algoritmo, el procesamiento de los datos y el cálculo de las probabilidades resulta lento debido a la necesidad de precomputar una gran cantidad de información. Sin embargo, una vez que estos cálculos iniciales se han realizado, el sistema es capaz de ejecutar las recomendaciones de manera mucho más eficiente. Esta característica hace que el sistema sea más adecuado para implementarse en computadoras con buenos recursos computacionales, donde la fase de precomputación puede realizarse sin afectar la experiencia del usuario final."
        )

    st.divider()

    st.markdown("## 6. Conclusiones")

    st.markdown(
        """
    El presente trabajo se centró en la hibridación de 
    técnicas en sistemas de recomendación, específicamente 
    comparando el enfoque probabilístico con la 
    factorización matricial en el contexto del filtrado 
    colaborativo. Los resultados permiten extraer las 
    siguientes conclusiones relevantes:
    """
    )

    st.markdown(
        """
    - El enfoque probabilístico, materializado a través del algoritmo *Naive Bayes Collaborative Filtering* (NBCF), no solo demuestra una capacidad comparable en términos de precisión frente a la factorización matricial, sino que también ofrece ventajas adicionales en cuanto a la interpretabilidad de las recomendaciones. Este aspecto resulta fundamental en aplicaciones donde la transparencia es clave para mejorar la experiencia del usuario.
        
    - La implementación del algoritmo NBCF y su expansión a la recomendación grupal mediante la técnica de *Naive Pooling* (NBP) resuelve de manera eficiente el problema de personalización colectiva. Los resultados experimentales mostraron que este enfoque proporciona recomendaciones coherentes tanto a nivel individual como grupal, lo que es particularmente útil en escenarios de recomendación para familias o grupos de amigos.

    - En la evaluación sobre el dataset *FilmTrust*, se observó que el enfoque NBCF hibridado mejoró las métricas de rendimiento clave como el error cuadrático medio (MSE) y el error medio absoluto (MAE), especialmente en grupos con votaciones positivas. No obstante, en contextos de votaciones negativas, el rendimiento fue menor, lo que apunta a posibles áreas de mejora en el tratamiento de conjuntos de datos escasos.

    - A pesar de los buenos resultados en general, el sistema de recomendación presenta limitaciones en la velocidad inicial de procesamiento y en su capacidad para predecir con precisión películas con bajas calificaciones, lo que sugiere la necesidad de optimizaciones adicionales tanto en la fase de precomputación como en la gestión de la dispersión de datos (*sparsity*) del conjunto.

    - Finalmente, se destaca la flexibilidad del enfoque probabilístico para integrar nuevas variables y su potencial para aplicaciones futuras en diferentes dominios, extendiendo así las capacidades actuales de los sistemas de recomendación colaborativos.
    """
    )

    st.markdown("La investigación confirma que la hibridación de técnicas de recomendación, en particular el enfoque probabilístico, ofrece una alternativa robusta y explicativa frente a la factorización matricial, mejorando la capacidad de personalización tanto a nivel individual como grupal, con oportunidades claras de optimización en áreas específicas. Se propuso la extensión del método NBCF para realizar recomendaciones a grupos mediando su hibridación con el método NBP.")


    st.markdown("## Bibliografía")

    st.write(
        """ 
        ###### [1] Valdiviezo-Diaz, P., Ortega, F., Cobos, E., \& Lara-Cabrera, R. (2019). A collaborative filtering approach based on Naïve Bayes classifier. IEEE Access, 7, 108581-108592."
        ###### [2] "González, O. E., \& Jacques, S. M. (2017). Estado del arte en los sistemas de recomendación. Res. Comput. Sci., 135, 25-40."
        ###### [3] Ortega, F., Rojo, D., Valdiviezo-Diaz, P., \& Raya, L. (2018). Hybrid collaborative filtering based on users rating behavior. IEEE Access, 6, 69582-69591.
        ###### [4] Valdiviezo, P. M. (2019). Sistema recomendador híbrido basado en modelos probabilísticos (Doctoral dissertation, Universidad Politécnica de Madrid).
        ###### [5] Samsudin, N. A., & Bradley, A. P. (2014). Extended naıve bayes for group based classification. In Recent Advances on Soft Computing and Data Mining: Proceedings of The First International Conference on Soft Computing and Data Mining (SCDM-2014) Universiti Tun Hussein Onn Malaysia, Johor, MalaysiaJune 16th-18th,2014 (pp. 497-505). Springer International Publishing.
        ###### [6] J. Golbeck, J. Hendler, “FilmTrust: movie recommendations using trust in web-based social networks”, CCNC 2006, 3rd IEEE Consumer Communications and Networking Conference, 2006, DOI:10.1109/CCNC.2006.159303210
        """
    )

if __name__ == "__main__":
    st.set_page_config(page_title="Informe", page_icon=":bar_chart:", layout="wide")
    main()
