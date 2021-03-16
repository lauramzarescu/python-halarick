import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

"""functia pentru calcularea celor 5 features principale"""
def run(path):

    """citirea imaginii de pe disc"""
    img = cv2.imread(path)
    
    props = {}

    """pentru a procesa cele 4 features, va trebui sa tranformam imaginea din RGB in gri"""
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    distances = [5]
    angles = [0]

    """calcularea glcm"""
    glcm = greycomatrix(gray_image, 
                        distances=distances, 
                        angles=angles,
                        symmetric=True,
                        normed=True)

    """definirea celor 5 proprietati (features)"""
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']

    """realizam un dictionar pentru imaginea noastra care sa contina cele 5 proprietati, dar si calea catre imaginea de pe disc (path)"""
    props['path'] = path
    for prop in properties:
        res = greycoprops(glcm, prop)[0][0]
        props[prop] = res
        
    return props

"""functia care calculeaza distanta euclidiana intre 2 puncte (in cazul nostru intre 2 matrici ce contin cele 5 proprietati)"""
def euclidean_distance(points):

    """excludem proprietatea "path" din calculul distantei deoarece nu avem nevoie de ea"""
    exclude_keys = {'path'}
    point1 = {x: points[0][x] for x in points[0] if x not in exclude_keys}
    point2 = {x: points[1][x] for x in points[1] if x not in exclude_keys}

    """pentru ca vom apela aceasta functie cu o lista de dictionare ca parametru, va trebui sa calculam punctele
    transformand valorile proprietatilor din dictionar in array-uri
    """
    p1 = np.array(list(point1.values())) 
    p2 = np.array(list(point2.values())) 

    """calculul propriu-zis al distantei euclidiene"""
    dist = np.linalg.norm(p1 - p2) 

    return dist
