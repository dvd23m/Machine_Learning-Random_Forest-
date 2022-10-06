# Erupciones volcanicas
![Volcanes](https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/jobmadrid/images/front.png) 

### Descripción del reto

Jorge es un geólogo del IGME (Instituto Geológico y Minero de España) que está desarrollando un nuevo sistema de prevención de erupciones para poder predecir qué tipo de erupción tendrá un volcán según las las vibraciones detectadas por sus sensores durante los días previos a la erupción. Esto permitirá reducir el riesgo de víctimas y destrozos materiales por este tipo de catástrofe natural. El sistema de Jorge trabaja con 5 tipos de erupciones:

> Pliniana: Se caracteriza por su alto grado de explosividad, con manifestaciones muy violentas en las cuales se expulsan grandes volúmenes de gas volcánico, fragmentos y cenizas.

> Peleana: La característica más importante de una erupción peleana es la presencia de una avalancha brillante de ceniza volcánica caliente, llamada flujo piroclástico.

> Vulcaniana: Son erupciones volcánicas de tipo explosivo. El material magmático liberado es más viscoso que en el caso de las erupciones hawaianas o estrombolianas; consecuentemente, se acumula más presión desde la cámara magmática conforme el magma asciende hacia la superficie.

> Hawaiana: Consiste en la emisión de material volcánico, mayoritariamente basáltico, de manera efusiva o no explosiva. Ocurre de este modo debido a que la difusión de los gases a través de magmas más básicos (basálticos) puede hacerse de manera lenta pero más o menos continua. Consecuentemente, las erupciones volcánicas de este tipo no suelen ser muy destructivas.

> Estromboliana: La erupción Estromboliana está caracterizada por erupciones explosivas separadas por periodos de calma de duración variable. El proceso de cada explosión corresponde a la evolución de una burbuja de gases liberados por el propio magma.

### Resultados y análisis

El reto pedía una métrica f1-macro como evaluación de los resultados. Tras aplicar distintos modelos, se ha obtenido los distintos resultados: 

|MODELO|F1-MACRO|  
|------|-------|  
|BALANCED RANDOM FOREST|0.7863|
|RANDOM FOREST|0.781335|  
|APLICANDO SMOTE|0.776991|   

### Solución adoptada  

Como se ha comentado préviamente, el reto consistía en emplear un modelo Random Forest, para clasificar tipos de erupción en función de los datos dados.
Tras un primer análisis de los datos, se ha visto que algunas de las clases tenían algunos datos más que otras. Pese a que la diferencia era mínima, se
ha decidido aplicar algunas técnicas que mitiguen esta diferencia y comprobar si realmente es reelevante o no.  

En primer lugar, se ha realizado Random Forest sobre los datos originales y sobre los datos originales eliminando los outliers. Para realizar esto se ha
creado una función que permite ejecutar este modelo enviando distinto parámetros cada vez así, es posible escoger el mejor de ellos en función de la métrica
f1-macro. La función devuelve un dataframe con el modelo que mejor puntuación ha conseguido. 

```def crear_modelos(x_train, y_train, x_test, y_test, algoritmo):
    '''
    Función que calcula distintos Random Forest.
    
    El cálculo se realiza aportando distintas configuraciones mediante ParameterGrid. 
    Para cada una de las configuraciones se ejecuta el modelo y se guarda el resultado
    en un dataframe, junto con su valor de f1-score macro.
    
    Parametros: Recibe los datos de entrenamiento y test correspondientes
    
    Retorna: La primera fila del dataframe, ordenado en función del mayor valor f1-score
    '''
    
    resultados = {'params':[], 'f1':[]}
    
    param_grid = ParameterGrid({
        "n_estimators":[100, 150, 200],
        "criterion": ["gini","entropy"], # calidad division
        "max_depth": [None, 10, 20, 50, 100, 200], # profundidad
        "max_features": [3, 4, 5, "sqrt"], # características a buscar
        "min_samples_split": [2,3,4]
        })    

    for params in param_grid:
        modelo = algoritmo(oob_score=False,
                                      n_jobs = -1,
                                      random_state = 42,
                                      ** params)
        
        modelo.fit(x_train, y_train)
        y_pred = modelo.predict(x_test)
        score = (f1_score(y_test, y_pred, average='macro'))
        resultados['params'].append(params)
        resultados['f1'].append(score)
        
    # Se crea el dataframe mediante el diccionario generado
    res = pd.DataFrame.from_dict(resultados['params'])
    res['f1'] = resultados['f1']
    res = res.sort_values('f1', ascending=False).reset_index()
    return res.head(1)
 ```

Debido a que los datos no estaban bien balanceados, se ha decidido utilizar el modelo [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)
por un lado y, por el otro, aplicar a los datos la técnica de [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html),
la cual, permite realizar un over-sample en los datos.  

Una vez obtenidas las métricas de estos 3 modelos, se ha escogido el que aportaba un f1 más alto y se ha realizado la predicción con los 
datos de test aportados para este proyecto.  

Finalmente, se ha generado el fichero csv "erupciones_volcanicas", que contiene las predicciones realizadas por el modelo sobre los datos de test.

### Sobre el autor

David Molina, estudiante del Grado de Data Science en UOC.  
Comparto mi linkedin: https://www.linkedin.com/in/david-molina-pulgarin-298253101
