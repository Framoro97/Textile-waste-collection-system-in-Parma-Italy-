# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:49:07 2025

@author: Davide
"""
from shapely.geometry import Polygon,Point
import xml.etree.ElementTree as ET
import re

def estrai_giro_da_descrizione(testo):
    try:
        testo_pulito = re.sub('<[^<]+?>', '', testo) #elimino tag
        match = re.search(r'\b(\d+)\b', testo_pulito) 
        return int(match.group(1)) if match else None
    except: 
        return None
    
def estrai_giro_vocale(testo):
    testo_pulito = re.sub('<[^<]+?>', '', testo) #elimino tag
    return testo_pulito

def estrai_nome_cassonetto(testo): 
    #non forza, non è molto utile ma meglio così
    match = re.search(r'cassonetto\s*\d+', testo, re.IGNORECASE)
    return match.group(0).strip() if match else testo.strip()

def estrai_dati_kml(percorso_file): #ManoMano
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(percorso_file)
    root = tree.getroot()

    dati = []

    for placemark in root.findall('.//kml:Placemark', ns):
        try:
            nome = placemark.find('kml:name', ns)
            descrizione = placemark.find('kml:description', ns)
            coordinate = placemark.find('.//kml:coordinates', ns)

            if nome is not None and descrizione is not None and coordinate is not None:
                lon, lat, *_ = coordinate.text.strip().split(',')
                cassonetto = estrai_nome_cassonetto(nome.text)
                giro = estrai_giro_da_descrizione(descrizione.text)

                dati.append({
                    'Cassonetto': cassonetto,
                    'Giro': giro,
                    'Latitudine': float(lat),
                    'Longitudine': float(lon)
                })
        except Exception as e:
            print(f"Errore di un placemark: {e}")

    return dati

def estrai_dati_cento_kml(percorso):
    lettere_to_numeri = {
    "A": 8,"B": 9,"C": 10,
    "D": 11,"E": 12,"F": 13}


    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(percorso)
    root = tree.getroot()
    dati=[]
    for placemark in root.findall('.//kml:Placemark', ns):
        try:
            nome = placemark.find('kml:name', ns)
            descrizione = placemark.find('kml:description', ns)
            coordinate = placemark.find('.//kml:coordinates', ns)

            if nome is not None and descrizione is not None and coordinate is not None:
                lon, lat, *_ = coordinate.text.strip().split(',')
                cassonetto = estrai_nome_cassonetto(nome.text)
                giro = estrai_giro_vocale(descrizione.text)
                giro = lettere_to_numeri.get(giro, None)
                
                dati.append({
                    'Cassonetto': cassonetto,
                    'Giro': giro,
                    'Latitudine': float(lat),
                    'Longitudine': float(lon)
                })
        except Exception as e:
            print(f"Errore di un placemark: {e}")

    return dati
    
def estrai_quartieri_kml(percorso_file):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(percorso_file)
    root = tree.getroot()

    quartieri = []

    for placemark in root.findall('.//kml:Placemark', ns):
        nome = placemark.find('kml:name', ns)
        #estraiamo poligoni per garantire che sia un quartiere
        coords_tag = placemark.find('.//kml:Polygon//kml:coordinates', ns)
        
        if nome is not None and coords_tag is not None:
            try:
                raw_coords = coords_tag.text.strip().split()
                punti = []
                for coord in raw_coords:
                    lon, lat, *_ = coord.split(',')
                    punti.append((float(lon), float(lat)))
                poligono = Polygon(punti)
                quartieri.append({'Nome': nome.text.strip(), 'Poligono': poligono})
            except Exception as e:
                print(f"Errore poligono per {nome.text}: {e}")

    return quartieri


'''
quartieri = estrai_quartieri_kml("ManoMano.kml")
for q in quartieri:
    print(f"{q['Nome']} - {q['Poligono'].centroid}")
    

percorso = "ManoMano.kml"
cassonetti = estrai_dati_kml(percorso)
#list c riporta 71 cas
cassonetti = [c for c in cassonetti if re.fullmatch(r'cassonetto\s*\d+', c["Cassonetto"], re.IGNORECASE)]

# Stampa risultati
for c in cassonetti:
    print(f"Cassonetto: {c['Cassonetto']}, Giro: {c['Giro']}, Lat: {c['Latitudine']}, Lon: {c['Longitudine']}")
'''