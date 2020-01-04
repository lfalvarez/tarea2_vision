import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import pandas as pd


def leer_features_y_labels_desde_dir(dir, squarerootnormalization):
    ## Devuelve las features, labels y nombres de archivo leidos desde un directorio dado.
    ## Recibe un parametro booleano que indica si haremos squarerootnormalization o no.
    features = np.fromfile('{dir}/features.des'.format(dir=dir), dtype=np.int32)[3:]
    if squarerootnormalization:
        sign = np.sign(features)
        features = sign * np.sqrt(np.abs(features))
        features = features/np.linalg.norm(features)
    first_dim = int(len(features) / 1024)
    features = np.reshape(features, (first_dim, 1024))
    labels = np.fromfile('{dir}/labels.npy'.format(dir=dir), dtype=np.int32)
    with open('{dir}/names.txt'.format(dir=dir)) as f:
        names = f.readlines()
    return {'features': features,
            'labels': labels,
            'names': names}

def get_classes(dir):
    ## Lee las clases de los ejemplos
    data = pd.read_csv(os.path.join(dir, 'classes.txt'), sep='\t', header=None, names=['clase', 'id'], usecols=['clase'])
    return data.to_dict(orient="dict")['clase']

def imprimir_imagenes(distances, filename, i, relevant_counter, squarerootnormalization, imagesworkingdir):
    str_datadir = '{imagesworkingdir}/png_w256'.format(imagesworkingdir=imagesworkingdir)
    ## esto imprime las fotitos concatenadas
    ## la primera es la de prueba y las que le siguen son la respuesta a la query
    r = np.random.random()
    ## Sólo para que no dibuje tanto le puse ese límite.
    ## los que tienen relevant counter == 2 son chistosos y no le achuntan a nada
    ## mientras que los que tienen mayor a 10 son aburridos porque le achuntaron a toda la cosa.
    imprimir = False
    if (relevant_counter == 0 and r > 0.98) or (relevant_counter >= 5 and r > 0.85) or relevant_counter > 9:
        imprimir = True
    if imprimir:
        para_mostrar = [os.path.join(str_datadir, filename)]
        for d in distances[:10]:
            para_mostrar.append(os.path.join(str_datadir, d['other_filename']))
        final_dim = (256 * len(para_mostrar), 256)
        new_im = Image.new('L', final_dim)
        for j, im_file in enumerate(para_mostrar):
            im = Image.open(im_file)
            new_im.paste(im, (j * 256, 0))
        if squarerootnormalization:
            dir_name = 'ejemplos_normalizados'
        else:
            dir_name = 'ejemplos_no_normalizados'
        new_im.save(dir_name + '/comparado_{}_{}.jpg'.format(relevant_counter, i))

def distancias_ordenadas_a_imagen_de_test(test_feature, current_label, filename, training_features, original_filenames,
                                          original_labels):
    ## Dada una imagen (sólo sus features) calcula las imagenes más cercanas en términos de
    ## sus features.
    ## Y devuelve una lista de diccionarios, cada diccionario describe una imagen, donde se indica si es relevante o no.
    ## la distancia, el nombre del otro archivo y el nombre de este archivo.
    distances = []
    for index, (other_feature, other_label, other_filename) in enumerate(
            zip(training_features, original_labels, original_filenames)):
        dist = np.linalg.norm(
            test_feature - other_feature)  # # calculando la distancia euclidiana entre el vector de la imagen de test y
        # la previamente calculada
        is_relevant = current_label == other_label
        datos_de_comparacion = {'is_relevant': is_relevant,
                                'dist': dist,
                                'other_filename': other_filename.strip(),
                                'this_filename': filename.strip()
                                }
        distances.append(datos_de_comparacion)
    sorted_distances = sorted(distances, key=lambda d: d['dist'])
    # y seleccionamos las 10 más cercanas.
    distances = sorted_distances[:10]
    return distances, filename

def calcula_distancia_a_todas_las_imagenes(test, training_data, squarerootnormalization, imagesworkingdir):
    ## Esta función calcula el mAP a todas las imagenes de test
    ## e imprime ejemplos de las querys.
    training_features = training_data['features']
    original_filenames = training_data['names']
    original_labels = training_data['labels']
    sum_of_aps = 0
    for i, test_feature in enumerate(test['features']):
        current_label = test['labels'][i]
        filename = test['names'][i].strip()
        distances, filename = distancias_ordenadas_a_imagen_de_test(test_feature,
                                                                    current_label,
                                                                    filename,
                                                                    training_features,
                                                                    original_filenames,
                                                                    original_labels)

        ## Si ya tengo las distancias de este ejemplo de test a todos los otros ejemplos
        ## Entonces puedo calcular su AP
        relevant_counter = 0
        AP = 0
        for index, d in enumerate(distances):
            dividendo = 0
            if d['is_relevant']:
                ## Si es relevante entonces el número de arriba de la división es la cantidad de aciertos
                relevant_counter += 1
                dividendo = relevant_counter
            ## el númnero de abajo siempre es el índice de donde estamos.
            AP += dividendo / (index + 1)
        if relevant_counter:
            ## La suma de todos los AP se divide por la cantidad total de
            ## elementos relevantes.
            AP = AP / relevant_counter
            sum_of_aps += AP
        imprimir_imagenes(distances, filename, i, relevant_counter, squarerootnormalization, imagesworkingdir)
    mAP = (sum_of_aps / len(test['features']))
    if squarerootnormalization:
        print("Con normalización")
    else:
        print("Sin normalización")
    print("El mAP es: ", mAP)

def plot(numbers, labels, classes):
    ## Dibuja un scatterplot con los ejemplos dados
    colors = tuple([int(label) for label in labels])
    groups = tuple(classes[label] for label in labels)
    data = []
    for pos, color, group in zip(numbers, colors, groups):
        x = pos[0]
        y = pos[1]
        data.append((x, y, color, group))
    df2 = pd.DataFrame(data, columns=['x', 'y', 'color','group'])
    sns.catplot(x="x", y="y", hue='group', legend="full", data=df2)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-imagesworkingdir', type=str, help="el directorio donde estan las imagenes", required = True)
    parser.add_argument('--squarerootnormalization', action="store_true")
    parser.add_argument('--tsne', action="store_true")
    args = parser.parse_args()
    training_data = leer_features_y_labels_desde_dir('training', args.squarerootnormalization)
    test = leer_features_y_labels_desde_dir('test', args.squarerootnormalization)
    if args.tsne:
        examples = np.random.randint(0, 6000, 200)
        x = []
        labels = []
        for i in examples:
            x.append(test['features'][i])
            labels.append(test['labels'][i])
        tsne = TSNE().fit_transform(x)
        classes = get_classes(parser.imagesworkingdir)
        plot(tsne, labels, classes)
    else:
        calcula_distancia_a_todas_las_imagenes(test, training_data, args.squarerootnormalization, args.imagesworkingdir)

