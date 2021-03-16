from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.button import Button 
from kivy.uix.label import Label 
 
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar 
from kivy.properties import ObjectProperty
from kivy.clock import Clock

import cv2
import os
from os import path

from kivy.core.window import Window

"""marimea ferestrei"""
Window.size = (1600, 800)

from processing import *

import threading 
import json
import time


class MyWidget(FloatLayout):

    progress_bar = ObjectProperty()

    """functie ce returneaza numar total al imaginilor din folderul de lucru"""
    def getTotalImages(self):
        total = 0
        folder = "./image.orig"
        for filename in os.listdir(folder):
            total += 1
        return total

    ############## FUNCTII PENTRU LUCRUL CU POPUP ###########################
    """functie de ajutor pentru popup care reseteaza counterul la 1"""
    def pop(self, popup):
        self.progress_bar.value = 1
        popup.open()

    """functie care atat timp cat counterul este mai mic decat numar total al imaginilor, va executa o alta functie de procesare.
    De observat faptul ca functia de procesare va fi executata intr-un thread separat deoarece nu putem modifica mai multe elemente
    de GUI in acelasi timp, fapt ce va duce la blocarea efectiva a aplicatiei pana procesarea va lua sfarsit"""
    def next(self):
        if self.progress_bar.value >= self.getTotalImages():
            return False
        t1 = threading.Thread(target = self.processing)
        t1.start() 

    """functie care deschide popup-ul"""
    def puopen(self, instance):
        self.next()

    #########################################################################

    """functia de procesare va scrie intr-un fisier toate informatiile necesare pentru fiecare imagine in parte.
    Astfel vom avea salvat intr-un fisier de tip text toate proprietatile + calea imaginii pentru a nu fi nevoiti sa calculam de fiecare
    data functia run().
    Daca avem un folder de 1000 de imagini, vom procesa proprietatile de 1000 de ori, in loc sa o facem ori de cate ori vom vrea sa gasim cele 
    mai disimilare 3 imagini"""
    def processing(self):
        folder = "./image.orig"
        images = []
        
        """verificam daca fisierul exista"""
        if path.exists('features.txt') and os.stat("features.txt").st_size > 0:
            print("features are already extracted!")    
        else:
            """daca nu exista, il vom crea si deschide pentru scriere"""
            with open('features.txt', 'a') as file:
                for filename in os.listdir(folder):
                    img = os.path.join(folder,filename)

                    """rulam functia run() pentru fiecare imagine in parte"""
                    properties = run(img)
                    file.write(json.dumps(properties) + "\n")
                    
                    """incrementam valorea counter-ului din progress bar"""
                    self.progress_bar.value += 1
        
        """deschidem fisierul text pentru citire"""
        features = open('./features.txt', 'r') 

        """citim linie cu linie"""
        lines = features.readlines() 
        
        count = 0
        props = self.getImageProperties()
        distances = []

        """pentru fiecare linie din fisier care contine informatiile prelucrate, vom calcula distanta euclidiana intre imaginea principala
        selectata si imaginea 'i' din fisier"""
        for line in lines: 
            extracted_properties = json.loads(line.strip())
            euc_distance = euclidean_distance([props, extracted_properties])
            """scriem intr-o lista toate distantele prelucrate impreuna cu calea catre imagine"""
            distances.append({"path": extracted_properties['path'], "distance": euc_distance})

        """prelucram cele mai disimilare 3 imagini pe baza tuturor distantelor calculate"""
        most_dissimilar = self.mostDissimilar(distances)

        """actualizam sursa imaginii si textul aferent distantei pentru a afisa in frontend"""
        self.ids.first_diss.source = most_dissimilar[0]['path'].replace(os.sep, '/')
        self.ids.first_diss_label.text = 'Euc. distance: ' + str(most_dissimilar[0]['distance'])

        self.ids.second_diss.source = most_dissimilar[1]['path'].replace(os.sep, '/')
        self.ids.second_diss_label.text = 'Euc. distance: ' + str(most_dissimilar[1]['distance'])

        self.ids.third_diss.source = most_dissimilar[2]['path'].replace(os.sep, '/')
        self.ids.third_diss_label.text = 'Euc. distance: ' + str(most_dissimilar[2]['distance'])
    

    """functie care returneaza cele mai mari 3 distante"""
    def mostDissimilar(self, distances):
        distances_arr = sorted(distances, key=lambda k: k['distance'], reverse = True)
        return [distances_arr[0], distances_arr[1], distances_arr[2]]

    """functie care ne ajuta la selectarea imaginii din file chooser"""
    def selected(self, filename):
        try:
            self.ids.image.source = filename[0]
            print(filename[0])
        except:
            pass

    """calculam proprietatile imaginii principale selectate"""
    def getImageProperties(self):
        properties = run(self.ids.image.source)
        return properties
    
    """functia principala care ne afiseaza proprietatile imaginii principale selectate"""
    def calculateFeatures(self):

        """daca nu avem nicio imagine selectata, vom afisa o eroare"""
        if not self.ids.image.source:
            popup = Popup(title='ERROR', content=Label(text='Select an image first'),
                        auto_dismiss=True, size=(300, 200),
                        size_hint=(None, None))
            popup.open()
        else: 
            """altfel, actualizam textul"""
            properties = self.getImageProperties()

            self.ids.contrast.text = "Contrast: " + str(properties.get('contrast'))
            self.ids.energy.text = "Energy: " + str(properties.get('energy'))
            self.ids.homogeneity.text = "Homogeneity: " + str(properties.get('homogeneity'))
            self.ids.correlation.text = "Correlation: " + str(properties.get('correlation'))
            self.ids.dissimilarity.text = "Dissimilarity: " + str(properties.get('dissimilarity'))

    """functia care controleaza butonul de preluare a imaginilor disimilare si care porneste popup-ul pentru prelucrarea tuturor imaginilor"""
    def getDissimilarImages(self):
        if not self.ids.image.source:
            popup = Popup(title='ERROR', content=Label(text='Select an image first'),
                        auto_dismiss=True, size=(300, 200),
                        size_hint=(None, None))
            popup.open()
        else:
            self.progress_bar = ProgressBar(max = self.getTotalImages())

            box = BoxLayout(orientation='vertical')
            box.add_widget(Label(text='Calculating features for all images...'))
            box.add_widget(self.progress_bar)
 
            popup = Popup(
                title = 'INFO', content = box,
                auto_dismiss = True, size = (500, 300),
                size_hint = (None, None)
            )
            popup.bind(on_open = self.puopen)
            self.pop(popup)

            if path.exists('features.txt') and os.stat("features.txt").st_size > 0:
                popup.dismiss()

            
        

class FileChooserWindow(App):
    def build(self):
        return MyWidget()
    
    
 
if __name__ == "__main__":
    window = FileChooserWindow()
    window.run()
