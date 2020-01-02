Para esta tarea hicimos lo siguiente:
=======================

## Carga de las features
Extrajimos las features de cada foto utilizando la CNN, como sale en el enunciado, las features las dejamos en dos
directorios y se ven así:

```
test
├── features.des
├── labels.npy
└── names.txt
training
├── features.des
├── labels.npy
└── names.txt
```

Para procesar las imágenes las leemos con el siguiente código:
```
features = np.fromfile('{dir}/features.des'.format(dir=dir), dtype=np.int32)[3:]
```

Ahí el directorio varía entre (test|training).
Los tres primeros parámetros son las cabeceras.

## Square-Root Normalization

Después de cargar los datos preguntamos si debemos normalizar o no:

```
if squarerootnormalization:
    sign = np.sign(features)
    features = sign * np.sqrt(np.abs(features))
    features = features/np.linalg.norm(features)
```

## Calculando la distancia de cada imagen de test a cada imagen de training.

Por cada imagen de test calculamos la distancia a todas las imagenes de training y seleccionamos las 10 más cercanas.
Esto nos sirve para calcular el AP y luego el mAP.

## Dibujar ejemplos

A veces (dependiendo de un random) dibujamos un ejemplo donde se muestran la imagen de la query concatenada con las
imágenes de prueba.

## El cálculo del mAP

El mAP nos resultó en:
- Con normalización el mAP es:  0.40358884316998356
- Sin normalización el mAP es:  0.45194464248971516

Esto contrasta con la tarea1 donde calculamos el mAP con HOG y allí tuvimos un mAP de ~0.35.

## Dibujar los ejemplo con T-SNE

Para esto utilizamos `seaborn`.
Seleccionamos un subset de 200 ejemplos de test (distribuidos uniformemente) y se ven así:
![](https://github.com/lfalvarez/tarea2_vision/raw/master/200_ejemplos_random_from_test.png)

## Conclusiones

Descubrimos que las CNN son mejores para obtener features que HOG. Logramos aumentar de 0.35 a 0.45(sin normalizar) y a 0.4(con normalización).

Además descubrimos que la CNN lograba detectar que objetos se parecían pero que tenían otra orientación, como por ejemplo:

![](https://github.com/lfalvarez/tarea2_vision/raw/master/ejemplos_normalizados/comparado_10_629.jpg)
