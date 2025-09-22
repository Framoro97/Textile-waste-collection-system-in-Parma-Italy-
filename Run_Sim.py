# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:28:00 2024

@author: zammo
"""

from typing import Optional, Iterable, NamedTuple, Callable, Any, Generator
from Time_Calendar import Calendar
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import simpy as sp
import re 
import pickle
import os
import time
from People import C_Trip, Citizen_type,f_influence, f_incentive, f_move, f_distance, green_variation, waste_gen, Citizen, Family, waste_log_gen
from Map import normal_pop, Neighborhood, All_Neighborhood, DMatrix, City
from Bins import Bin
from Vehicles import Truck
from Sim_Manager import Manager
from NewLista import  estrai_dati_kml, estrai_quartieri_kml, estrai_dati_cento_kml

""" UNITA' DI GRANDEZZA
tempo in ore
distanze in km
velocità km/h
quantità kg generati al mese """
s=time.time()
percorsoM = "ManoMano.kml" #contiene i quartieri per la sim,
                           #e i bidoni di mano a mano
percorsoC = "Centomiglia.kml" #bidoni centomiglia

estraiQ = estrai_quartieri_kml(percorsoM)
estraiC=estrai_dati_cento_kml(percorsoC)
estraiM = estrai_dati_kml(percorsoM)

random.seed(10) # seme fissato per avere sempre gli stessi valori !!!
env = sp.Environment() 

estraiC=[i for i in estraiC if i["Giro"]!=None] #elimino le posizioni che non possiedono un giro
#nel file centomiglia ci sono quattro punti, di cui tre che possiedono una descrizione che vengono estratti

n= 40
oriG=(44.754278, 10.26)
lat0, lon0 = oriG
d=0.25
c = 111.32
delta_lat = round(d / c,6)
delta_lon = round(d / (c * math.cos(math.radians(lat0))),6)

one_day = 24
one_week = one_day*7
one_month = one_week*4
n_months = 24

#Imposto il calendario della simulazione
sim_calendar = Calendar(env)
sim_calendar.date['yy'] = 2026
sim_calendar.date['mm'] = 1
sim_calendar.date['dd'] = 1
env.process(sim_calendar.make_calendar())

"""************ PARAMETRI POPOLAZIONE - QUARTIERI ***************** """
pop_vigatto = normal_pop(mu = 50, sigma = 25, min_pop = 15) # crea popolazione normale, ma si potrebbe passare anche un valore fisso!
pop_molinetto= normal_pop(mu = 67, sigma = 25, min_pop = 15)
pop_lubiana= normal_pop(mu = 39, sigma = 25, min_pop = 5)
pop_sanpancrazio= normal_pop(mu = 9, sigma = 4, min_pop = 1)
pop_sanlazzaro= normal_pop(mu = 11, sigma = 4, min_pop = 2)
pop_cittadella= normal_pop(mu = 33, sigma = 10, min_pop = 5)
pop_montanara= normal_pop(mu = 167, sigma = 25, min_pop = 15)
pop_oltretorrente= normal_pop(mu = 225, sigma = 40, min_pop = 15)
pop_sleonardo= normal_pop(mu = 142, sigma = 25, min_pop = 15)
pop_csmartino= normal_pop(mu = 5, sigma =3, min_pop = 1)
pop_golese= normal_pop(mu = 6, sigma = 3, min_pop = 1)
pop_centro= normal_pop(mu = 238, sigma = 40, min_pop = 15)
pop_pablo= normal_pop(mu = 92, sigma = 15, min_pop = 15)
'''
gen_waste = waste_gen(0.5, 3, 4) # kg generati ogni mese, distrib triangolare
'''
#distribuzione basata su una logaritmica per simulare stagionalità
gen_waste=waste_log_gen() #sostiuisco la funzione di generazione
tipo_conf = "non_positiva" # positiva oppure non_positiva


if tipo_conf == "positiva":
    """ CONFIGURAZIONE POSIIVA
    che tendenzialmente spinge verso un incremento della green awareness """
    green_inc = green_variation(delta = 0.01, increase = True) # incremento se trova bidone vuoto, distribuzione esponenziale negativa
    green_dec = green_variation(delta = -0.02, increase = False) # decremento se trova bidone pieno, distribuzione esponensiale negativa
    f_inf = f_influence(alpha = 0.5, beta = 2.5, delta_inc = 0.01, delta_dec = -0.02)
    f_dist = f_distance(tau = (0.55, 0.6), traslazione = (2.5, 2.0), troncamento = 0.2) # sino a 200 metri effetto nullo, a 5000 metri praticamente la gaw si annulla.
elif tipo_conf == "non_positiva": 
    """ CONFIGURAZIONE NEGATIVA 
    che tendenzialmente spinge ad un peggiorament di green awareness """
    green_inc = green_variation(delta = 0.001, increase = True) # incremento se trova bidone vuoto, distribuzione esponenziale negativa
    green_dec = green_variation(delta = -0.04, increase = False) # decremento se trova bidone pieno, distribuzione esponensiale negativa#
    f_inf = f_influence(alpha = 0.5, beta = 2.5, delta_inc = 0.001, delta_dec = -0.05)
    f_dist = f_distance(tau = (0.55, 0.6), traslazione = (2.5, 2.0), troncamento = 0.2) # sino a 200 metri effetto nullo, a 5000 metri praticamente la gaw si annulla.


f_inc = f_incentive(low = 0.1, high = 0.2) # incremento percentuale a seguito di incentivo, distribuzione uniforme
f_mv = f_move(d_max = 0.3, v_feet = 2.5, v_car = 30)

""" Supponiamo incentivo basso, con effetto solo sui cittadini 'neutrali' 
     e, se presente, effetto della distanza solo su quelli medi e negativi """
eco = Citizen_type(kind ='eco_friendly', green_awareness = (0.7, 0.8), radius_of_influence = (0.2, 0.3), n_depth = 1,
                  gen_waste = gen_waste, green_inc = green_inc, green_dec = green_dec, 
                  f_influence = f_inf, f_incentive = None, f_dist_penalty = None, f_move = f_mv) # loro non risentono dell'effetto

neutral = Citizen_type(kind ='neutral', green_awareness = (0.45, 0.55), radius_of_influence = (0.2, 0.5), n_depth = 1,
                  gen_waste = gen_waste, green_inc = green_inc, green_dec = green_dec, 
                  f_influence = f_inf, f_incentive = f_inc, f_dist_penalty = f_dist, f_move = f_mv) # loro risentono dell'effetto

non_eco = Citizen_type(kind ='non_eco', green_awareness = (0.15, 0.4), radius_of_influence = (0.4, 0.6), n_depth = 2,
                  gen_waste = gen_waste, green_inc = green_inc, green_dec = green_dec, 
                  f_influence = f_inf, f_incentive = None, f_dist_penalty = f_dist, f_move = f_mv) # loro risentono dell'effetto


"""I Quartieri"""

mappa_nomi = {"MOLINETTO":(Neighborhood(population = pop_molinetto, type_prob = {eco:0.6, neutral:0.4}),"purple"),
              "VIGATTO": (Neighborhood(population = pop_vigatto, type_prob = {neutral:0.4, non_eco:0.6}),"red"),
              "MONTANARA":(Neighborhood(population = pop_montanara, type_prob = {eco:0.6, neutral:0.4}),"beige"),
              "CITTADELLA":(Neighborhood(population = pop_cittadella, type_prob = {neutral:0.4, non_eco:0.6}),"orange"),
              "LUBIANA":(Neighborhood(population = pop_lubiana, type_prob = {neutral:0.4, non_eco:0.6}),"brown"),
              "SAN LAZZARO":(Neighborhood(population = pop_sanlazzaro, type_prob = {eco:0.6, neutral:0.4}),"pink"),
              "SAN PANCRAZIO":(Neighborhood(population = pop_sanpancrazio, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"cyan"),
              "OLTRETORRENTE":(Neighborhood(population = pop_oltretorrente, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"magenta"),
              "PABLO":(Neighborhood(population = pop_pablo, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"teal"),
              "PARMA CENTRO":(Neighborhood(population = pop_centro, type_prob = {eco:0.6, neutral:0.4}),"navy"),
              "S.LEONARDO":(Neighborhood(population = pop_sleonardo, type_prob = {neutral:0.4, non_eco:0.6}),"coral"),
              "GOLESE":(Neighborhood(population = pop_golese, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"lightgray"),
              "C.S. MARTINO":(Neighborhood(population = pop_csmartino, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"salmon")}

""" 
#per test simulazone più veloci
mappa_nomi = {"MOLINETTO":(Neighborhood(population = 20, type_prob = {eco:0.6, neutral:0.4}),"purple"),
              "VIGATTO": (Neighborhood(population = 15, type_prob = {neutral:0.4, non_eco:0.6}),"red"),
              "MONTANARA":(Neighborhood(population = 30, type_prob = {eco:0.6, neutral:0.4}),"beige"),
              "CITTADELLA":(Neighborhood(population = 25, type_prob = {neutral:0.4, non_eco:0.6}),"orange"),
              "LUBIANA":(Neighborhood(population = 20, type_prob = {neutral:0.4, non_eco:0.6}),"brown"),
              "SAN LAZZARO":(Neighborhood(population = 11, type_prob = {eco:0.6, neutral:0.4}),"pink"),
              "SAN PANCRAZIO":(Neighborhood(population = 12, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"cyan"),
              "OLTRETORRENTE":(Neighborhood(population = 16, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"magenta"),
              "PABLO":(Neighborhood(population = 13, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"teal"),
              "PARMA CENTRO":(Neighborhood(population = 14, type_prob = {eco:0.6, neutral:0.4}),"navy"),
              "S.LEONARDO":(Neighborhood(population = 18, type_prob = {neutral:0.4, non_eco:0.6}),"coral"),
              "GOLESE":(Neighborhood(population = pop_golese, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"lightgray"),
              "C.S. MARTINO":(Neighborhood(population = 9, type_prob = {eco:0.3, neutral:0.3, non_eco:0.4}),"salmon")}
"""

The_Map = All_Neighborhood(tuple(quart[0] for quart in mappa_nomi.values()))

s1=time.time()
ss=(s1-s)/60
print(ss)
#Quartieri: dict[str, list[tuple[int, int]]] = {}
node_colors:dict[tuple[float,float],str] = {}  
for nd in The_Map.gen_node(start_node=(0, 0), nr=n, nc=n):
    r, c = nd
    lat = lat0 + r * delta_lat
    lon = lon0 + c * delta_lon
    q = The_Map.assign_neigh(lat, lon, estraiQ)  #Restituisce dict con 
    
    if q:
        nome = q['Nome'].upper().strip()
        #Quartieri.setdefault(nome, []).append(nd)
        valore = mappa_nomi.get(nome)
        if valore is not None:
            tipo, colore = valore 
            The_Map[nd] = tipo
            
            # Prepara i colori da applicare più tardi
            #Passare il dict in City
            node_colors[nd] = colore
    else: node_colors[nd]="white" #Non assegnati

# Conferimento settimanale spannometrico
'''
# Cf = persone per nodo*numero di nodi*conferimento medio(3)*prob_conferimento(0.5)/giro camion(4)
# Cf/bidoni= capacità
'''

#Dovrei controllare come mai estrae più cassonetti, puliamo

cassonetti = [c for c in estraiM if re.fullmatch(r'cassonetto\s*\d+', c["Cassonetto"], re.IGNORECASE)]
cassonetti+=estraiC #manomano+centomiglia

s9=time.time()
ss=(s9-s)/60
print(ss)



# Creazione oggetti Bin, ho aggiunto la variabile giro per semplificare la creazione dei truck
if tipo_conf == "positiva":
    """ CONFIGURAZIONE DI BASE
    Usando questa ha livello di servizio molto alto, tendente ad 1 :
    bidoni ben dimensionati e con opzione a chiamata attiva """
    bins_gps = [ Bin(env=env, capacity=180*b['Quantita'], threshold=None, 
                low_treshold=20, gps=(b['Latitudine'], b['Longitudine']),
                giro=b['Giro']) for b in cassonetti] #Alcuni bidoni hanno capacità aumentate
    
else:
    """ CONFIGURAZIONE ALTERNATIVA
    Usando questa si avrebbe una situazione con livello di servizio basso:
    bidoni piccoli, mal dimensionati e mai a chiamata """
    bins_gps = [ Bin(env=env, capacity=180*b['Quantita'], threshold=None, 
                low_treshold=20, gps=(b['Latitudine'], b['Longitudine']),
                giro=b['Giro']) for b in cassonetti]



The_City = City(env = env, n_nodes = n*n, distance = d, the_map = The_Map,start_gps=oriG,
                bins = bins_gps, in_out = ((0, 0),((n-1),(n-1))), color=node_colors, calendar = sim_calendar)


The_City.show(True, True)
s2=time.time()
ss=(s2-s)/60
print(f"Mappa stampata tempo simulazione trascorso: {ss} minuti")
""" automezzi """
Std_Truck = Truck(env = env, the_map = The_City.gen_dmat, capacity = 2000, threshold = 1750, velocity = 35, n_in = (0, 0), n_out = (n-1, n-1),
                  t_load = lambda kg: kg*0.001, t_unload = lambda kg: 0.2)

#Crea ed assegna un camion per ogni tipologia di giro
trucks = [Std_Truck.copy([b for b in bins_gps if b.giro == g]) for g in {b.giro for b in bins_gps}]
s3=time.time()
ss=(s3-s)/60
print(ss)
Mg = Manager(env= env, the_city = The_City, trucks = trucks, calendar = sim_calendar)
s4=time.time()


"""
#Test su calendario
for cz in Mg.city.iter_citizens():
    print("Calendar per cittadino:", cz.calendar)
    break  # ne stampi solo uno
"""
Mg.gen_waste_single_citizen(mean_time = one_month, rnd_bin = True)
s5=time.time()
env.process(Mg.update_green_awareness(dt = one_month, min_dist = 0.05, wgh = True))

# FRA_ Aggiunta media Gaw dei cittadini che risiedono nei nodi serviti dai bidoni
env.process(Mg.monitor_avg_gaw())

for tr in Mg.tr_bin.keys():
    env.process(Mg.schedule_trucks(tr, dt = one_week))


env.process(Mg.progress(24*one_month, 0.01))
s6=time.time()
env.run(until = n_months*one_month)
print(f'{float(Mg.env.now):.1} - {1.00:.2%}') 
print()

print(f'Simulation has ended, a total of {n_months} of simulated months have passed.')

# per debug calcolo del livello di servizio di ogni cittadino, ora ok!
# sl = tuple(cz.s_level for cz in Mg.city.iter_citizens())

# statistics and plots
Mg.show_green_awareness()
for idx in [i for i in range(0,len(cassonetti))]:
    Mg.show_bin_trend(idx)
print("All plots are visible in the Plots panel.")
print()
print("*** THE MAIN STATISTICS ARE SHOWN NEXT ****")
x = Mg._bin_kg()
for label, stat in  [('Bins stat', Mg.show_bins_stat()),
                     ('Trucks stat', Mg.show_trucks_stat()),
                     ('Citizens stat', Mg.show_ctzs_stat()),
                     ('Citizens green awaraness variation', Mg.show_ctzs_gaw_variation()),
                     ('Total Cost in euro', f'€{Mg.tot_cost(truck_amm = 5000, bin_amm = 500, euro_km = 0.05, euro_kg = 0.1):.2f}')
                     ]:
    print()
    print(label)
    print(stat)

#estrazione dati set n simulazioni 
#dataset pandas? 
#pickle
s7=time.time()
ss=(s7-s)/60
print(f"Fine, tempo simulazione trascorso: {ss} minuti o {ss/60} ore")

'''
#per debug 
#statistiche di un cittadino
cz=Mg.city.get_cz((10,10),idx=7)
for i, trip in cz.sts.items():
    print(f"Viaggio {i}:")
    print(f"  Kg generati: {trip.kg}")
    print(f"  Nodo cassonetto: {trip.tr}")
    print(f"  Kg riciclati: {trip.kg_recycled}")
    print(f"  Kg persi: {trip.kg_rlost}")
    print(f"  Bidone pieno?: {trip.full_bin}")
    print(f"  A piedi?: {trip.bf}")
    print(f"  Δ green awareness: {trip.var_gaw}")
    print()
The_City.gf.nodes[(0,0)]['gps'] #per nodi 

#per vedere truck
sts-->statistiche 
sts[].full_route riferito al singolo giro
'''

os.makedirs("output", exist_ok=True)
"""
# salvo Mg
with open("output/Mg_simulato.pkl", "wb") as f:
    pickle.dump(Mg, f)

# 1. Bins stat
bins_stat = Mg.show_bins_stat()
df_bins = pd.DataFrame.from_dict(bins_stat, orient='index')
df_bins.to_pickle("output5/bins_stat.pkl")

# 2. Trucks stat
trucks_stat = Mg.show_trucks_stat()
df_trucks = pd.DataFrame.from_dict(trucks_stat, orient='index')
df_trucks.to_pickle("output5/trucks_stat.pkl")

# 3. Citizens stat (cum e mean)
czs_stat = Mg.show_ctzs_stat()
with open("output3/citizens_stat.pkl", "wb") as f:
    pickle.dump(czs_stat, f)
"""


"""
# Al termine della simulazione
for idx, b in enumerate(Mg.city.bins):
    print(f"Bin {idx} @ nodo {b.nd}: avg_gaw = {b.avg_gaw:.3f}")
"""