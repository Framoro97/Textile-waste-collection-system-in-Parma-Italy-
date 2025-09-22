# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Dec 17 15:04:26 2024
@author: zammo

Classi che creano e popolano i quartieri
della città

"""
from Time_Calendar import Calendar 
import networkx as nx
import simpy as sp
import random
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from functools import partial
from typing import Callable, Optional, Any, Generator, Iterable, Any

from People import Citizen, Citizen_type, Family
from Bins import Bin

""" Definizione tipi variabili """           
f_pop = Callable[[],int] # tipo funzione randomica per creare la popolazione
Nd = tuple[int, int] # tipo nodo
Gr = nx.classes.graph.Graph # tipo grafo
Df = pd.core.frame.DataFrame

"""**** ALCUNE FUNZIONI D'APPOGGIO **** """

def normal_pop(mu:int, sigma:int, min_pop:int = 1):
    """ genera una popolazione gaussiana 
            eventualmente troncando an valore minimo """
    def inner():
        return max(min_pop, int(random.normalvariate(mu, sigma)))
    inner.__name__ = f'normal_pop({mu}, {sigma})'
    return inner

def gen_coord(nr:int, nc:int, dst:float) -> Generator[tuple[float, float], None, None]:
    """ Genera le coordinate cartesiane dei nodi del network, 
            i nodi devono essere a griglia
            e equispaziati con distanza pari a dst  """
    x, y = 0, 0
    for r in range(nr):
        for c in range(nc):
            yield x, y
            x += dst
        x = 0
        y += dst

def gen_neighbor(nd:Nd, n_layer:int = 1) ->  Generator[Nd, None, None]: 
    """ Genera la lista dei nodi vicini ad un nodo di partenza, 
            compreso il nodo stesso: es. (1, 1) genera (0, 0), (1, 0), ecc. 
                con n_layer = 0 viene restituito il nodo stesso, 
                        con n_layer = 1 vengono restituiti gli 8 nodi vicini pià il nodo stesso """
    if n_layer == 0: yield nd      
    else:
        x, y = nd
        xmin, xmax = x - n_layer, x + n_layer + 1   
        ymin, ymax = y - n_layer, y + n_layer + 1 
        for ng in itertools.product(range(xmin, xmax), range(ymin, ymax)):
            yield ng
       
def random_circle(center:tuple[float, float], radius:float)-> tuple[float, float]:
    """ Genera una coordinata cartesiana casuale all'interno di un cerchio """
    teta = random.uniform(0, 2*math.pi)
    r = random.uniform(0, radius)
    x, y = center
    return round(x + r*math.sin(teta),2), round(y + r*math.cos(teta),2)

""" **** LE CLASSI **** """

class Neighborhood:
    """ Classe che definisce le specifiche di un quartiere 
            i.e., la percentuale di ciascun tipo di cittadino presente nel quartiere"""
    
    def __init__(self, population: int|Callable[[], float], type_prob:dict[Citizen_type, float]):
        # population è un valore intero o una distribuzione che verrà usata
        # per assegnare la popolazione al quartiere 
        # type_prov è un dizionario che associa alla tipologia del cittadino la probabilità
        # che quel cittadino venga creato 
        if sum(pr for pr in type_prob.values()) != 1: raise Exception('Probabilità errata!')
        if not callable(population): 
            self.pop = partial(lambda x: x,  int(population))
            self._fname = f'{int(population)} fixed'
        else:
            # quando copiamo un vicinato col metodo copy(), population è sempre callable,
            # ma potrebbe essere una lambda parzializzata che restituisce un valore fisso
            # e che non ha nome, in questo caso prima mettiamo self._fname a None e dopo
            # sarà copy a settarlo correttamente.
            try: self._fname = population.__name__
            except: self._fname = None 
            self.pop = population
        self.tpr: dict[Citizen_type, float] = type_prob
    
    def copy(self) -> Neighborhood:
        """ copia del vicinato """
        ng = Neighborhood(population = self.pop, type_prob = self.tpr)
        ng._fname = self._fname 
        return ng
    
    def __repr__(self) -> str:
        return f'population: {self._fname}, kinds: {tuple((cz.kind, prob) for cz, prob in self.tpr.items())}'
    
    def __call__(self, env, center:tuple[float, float] =(0, 0), radius:float = 0,
                 nd:Nd=(0, 0), bins:Optional[Iterable[Bin]]=None, family_ratio=0.5,
                 calendar:Optional[Calendar] = None) -> tuple[Citizen]:
        n_total = self.pop()
        n_fam = round(n_total * family_ratio)
        n_sing = n_total - n_fam
        c_types = tuple(self.tpr.keys())
        probabilities = tuple(self.tpr.values())

        res = []

        for _ in range(n_fam):
            pos = random_circle(center, radius)
            att = random.choices(c_types, probabilities, k=1)[0]
            res.append(Family(env=env,xy_coor=pos, reference_node=nd,
                          distance_from_node=round(math.dist(center, pos), 4),
                          bins=bins, attributes=att, calendar = calendar))  # o random.randint(3,6)
#NB : n_members è il numero di membri della famiglia, il settaggio finale avviene all'interno del metodo init della classe Family
            pos = random_circle(center, radius)
            att = random.choices(c_types, probabilities, k=1)[0]
            res.append(Citizen(env=env, xy_coor=pos, reference_node=nd,
                        distance_from_node=round(math.dist(center, pos), 4),
                           bins=bins, attributes=att, calendar=calendar))

        return tuple(res)
    '''
    def __call__(self, env, center:tuple[float, float] = (0, 0), radius:float = 0, 
                 nd:Nd = (0,0), bins:Optional[Iterable[Bin]] = None) -> tuple[Citizen]:
        """ crea i cittadini del quartiere 
                piazzandoli all'interno di un'area circolare """
        probabilities = tuple(self.tpr.values())
        c_types = tuple(self.tpr.keys())
        n_ctz = self.pop()
        positions = (random_circle(center, radius) for _ in range(n_ctz))
        return tuple(Citizen(env = env, xy_coor = pos, reference_node = nd, distance_from_node = round(math.dist(center, pos), 4),
                             bins = bins, attributes = random.choices(c_types, probabilities, k = 1)[0]) 
                             for pos in positions)
    '''
class All_Neighborhood:
    """crea tutti i quartieri della città, indicando ad ogni nodo popolato
            quali tipologie di cittadini e in che percentuali ci stanno 
                es. nel nodo x,y il 60% è eco friendly e il 40% è non-eco. 
                    per farlo è necessario usare un metodo di creazione (all_equal, all_random), ecc.
                        oppure farlo manualmnete, nodo per nodo usando il metodo di assegnazione [] """
        
    def __init__(self, neighborhoods:Iterable[Neighborhood]):
        """ registra le tipologie di vicinato e predispone
                il dizionario all_ngh che associa ad ogni nodo popolato il tipo di vicinato """
        self.ngh = tuple(neighborhoods)
        self.all_ngh:dict[Nd, Neighborhood] = {}
        self.czk:tuple[str] = self._find_cz_kind() # la lista dei tipi di cittadini presenti in città
       
    def _find_cz_kind(self) -> tuple[str]:
        """ trova i tipi di cittadini presenti in città """
        kd = set() 
        for ng in self.ngh:
            kd.update(tuple(cz.kind for cz in ng.tpr.keys()))
        return tuple(kd)
    
    def __repr__(self) -> str: 
        return f'pop_mix: {len(self.ngh)}, cz_kind = {self.czk}, sn_nodes: {len(self.all_ngh)}'
    
    """ METODI DI GENERAZIONE DEI VICINATI """
         
    def all_equal(self, nodes:Iterable[Nd], idx_ng:int): 
        """ nodes -> i nodi a cui assegnare una tipologia di vicinato 
            idx -> l'indice della tipologia di vicinato 
            tutti i nodi passati in input avranno la stessa composizione """   
        self.all_ngh.update({nd:self.ngh[idx_ng].copy() for nd in nodes})
    
    def all_random(self, nodes:Iterable[Nd]):
        """ nodes -> i nodi a cui assegnare una tipologia di vicinato 
            ad ogni nodo verrà assegnata una tipologia scelta in maniera
            random tra quelle in self.ngg """
        self.all_ngh.update({nd:random.choice(self.ngh).copy() for nd in nodes})
        
    def __getitem__(self, nd:Nd) -> Neighborhood|None:
        return self.all_ngh.get(nd, None)
    
    def __setitem__(self, nd:Nd, ng:Neighborhood):
        self.all_ngh[nd] = ng
    
    def __iter__(self) -> Generator[tuple[Nd, Neighborhood], None, None]:
        """ restituise, una ad una, le coppie nodo-vicinato """
        for nd, ng in self.all_ngh.items():
            yield nd, ng
         
    @staticmethod
    def assign_neigh(lat: float, lon: float, lista_quartieri: list[dict]) -> dict | None:
        """
        Restituisce la tipologia di quartiere a cui appartiene il nodo.
        """
        punto = Point(lon, lat)
        # punto contenuto nel poligono
        for q in lista_quartieri:
            if q['Poligono'].contains(punto):
                return q
            
    @staticmethod
    def gen_node(start_node = (0, 0), nr = 1, nc = 1, *, step_r = 1, step_c = 1) -> Generator[Nd, None, None]:
        """ genera insiemi di nodi contigui, 
               nr ->  quante righe
                  nc -> quante colonne """
        rs, cs = start_node
        for r in range(rs, rs + nr, step_r):
            for c in range(cs, cs + nc, step_c):
                yield r, c

class DMatrix:
    """ La classe che gestisce una matrice delle distanze 
           matrice definita come dizionario nodo-nodo: distanza """
    
    def __init__(self, Dm:dict[tuple[Nd, Nd], float]):
        """ Dm è la matrice delle distanze, non è necessario che sia 
            simmetrica. Se è simmetrica basta la parte superiore """
        self.dm = Dm
    
    def __getitem__(self, ns:tuple[Nd, Nd]) -> float:
        """ restituisce la distanza tra due nodi, se collegati """
        n1, n2 = ns
        if n1 == n2: return 0
        try: return self.dm[ns]
        except: return self.dm.get((n2, n1), math.inf) # se non c'è un collegamento allora distanza infinita!!!
    
    def all_nodes(self) -> list[Nd]:
        """ la listsa di tutti i nodi del grafo """
        nds = []
        for n1, n2 in self.dm.keys():
            if n1 not in nds: nds.append(n1)
            if n2 not in nds: nds.append(n2)
        return nds
                
    def _as_df(self) -> Df:
        """ Converte la matrice in un data frame
            per poterla visualizzare graficamente """ 
        nds = self.all_nodes()
        # crea il dizionario che verrà convertito in DF
        # il formato è il seguente n1: {n1:0, n2:5, ..., nn = 7} ... nn:{n1:7, ..., nn:0}
        dct = {str(nd):{str(ot_nd):self[nd, ot_nd] for ot_nd in nds} for nd in nds}
        return pd.DataFrame.from_dict(dct)
    
    def __repr__(self) -> str:
        return repr(self._as_df())
    
class City:
    """ L'intera città popolata """
    
    def __init__(self, env, 
                       n_nodes:int, # numero totale di nodi
                       distance:float, # distanza fissa tra nodi adiacenti
                       the_map: All_Neighborhood, *, 
                       bins:Optional[tuple[Bin]] = None, # bidoni
                       in_out: Optional[tuple[Nd]] = None, 
                       start_gps:Optional[tuple[float,float]]= None,
                       color:Optional[dict[tuple[float,float], str]]=None,
                       calendar:Optional[Calendar] = None):  #nodo di input e nodo di output

        """ 
        Attributi di base
        non andrebbero mai modificati, ma non ho messo _ per pigrizia @_@ 
        """
        self.env = env
        self.color=color
        self.origin= start_gps
        self.d = distance 
        self.nn = int(n_nodes**0.5) # numero di nodi per riga e per colonna  
        self.map: All_Neighborhood = the_map 
        self.calendar = calendar
        # input e outpt 
        # se non passati in input vengono messi nei due vertici opposti: basso-sx, alto-dx"""
        try: self.n_in, self.n_out = in_out
        except: self.n_in, self.n_out = (0, 0), (self.nn - 1, self.nn - 1)      

        self.gf:Gr = self._gen_graph() # il grafo 2d
        self._assign_gps_coordinates(self.origin)
        
        #colore basato sul quartiere
        if self.color:
            for i,j in self.color.items():
                self.gf.nodes[i]['color'] = j
        #potrei eliminarlo dal metodo gen_graph()
        self.gf.nodes[self.n_in]['color']='green'
        self.gf.nodes[self.n_out]['color']='green'
        if bins:
            # Se i cassonetti non hanno un nodo associato, li posizioniamo
            if any(B.nd is None for B in bins):
                self.assign_bins_to_nearest_nodes(bins)
                self.bins = bins
        # se non vengono definiti ne mettiamo 4 disposti a croce
        else: self.bins = Bin(env = env, capacity = random.randint(500, 1000), 
                    threshold = None).gen_a_croce(lw = 0, md = self.nn//2, hg = self.nn - 1)
        
        
        # aggiungiamo il colore ai nodi con bin e assegnamo le coordinate cartesiane ai bin
        for B in self.bins:
            self.gf.nodes[B.nd]['color'] = "yellow"
            B.crd = self.gf.nodes[B.nd]['coord']
        
        # creiamo la popolazione di ogni nodo
        # oss. ngh è un oggetto callable che crea la popolazione, 
        # gli passiamo i bin e la distanza dal nodo di riferimento per completare la generazione
        self.ctzs: dict[Nd:tuple[Citizen]] = {nd:ngh( #in questo momento viene eseguito __call__() nella classe Neighborhood, generando e restituendo la tupla di cittadini.
                                                        env = self.env, 
                                                        center = self.gf.nodes[nd]['coord'], 
                                                        radius = round(self.d/2, 4),
                                                        nd = nd, 
                                                        bins = {b:nx.shortest_path_length(self.gf, source = nd, target = b.nd, weight='distance') for b in self.bins},
                                                        calendar = self.calendar, # argomento = valore già posseduto
                                                        )
                                                for nd, ngh in self.map} # i cittadini di ogni quartiere
        
        # FRA_Assegna ogni nodo al bin più vicino --> il bin più vicino è il primo per ogni cittadino
        
        for ctz in self.iter_citizens():
            new_bin = tuple(ctz.bins.keys())[0]
            new_bin.nd_ctzs[0].add(ctz.nd)
            new_bin.nd_ctzs[1].add(ctz)
        

        # aggiorniamo la popolazione associata ad ogni nodo 
        for nd, cs in self.ctzs.items(): self.gf.nodes[nd]['population'] = len(cs)
        # aggiungiamo gli influencer ad ogni cittadino
        for cz in self.iter_citizens(): self._get_influencers(cz)
        """ Attributi Calcolati """
        # numero di cittadini per tipologia 
        # numero di citadini totali
        self.population = {kd:self.n_citizens(flt = partial(lambda k, cz: cz.att.kind == k, kd)) for kd in self.map.czk}
        self.population['global'] = self.n_citizens()
    
            
    def _assign_gps_coordinates(self, origin_gps):
        """
        Assegna coordinate GPS a ogni nodo partendo dal nodo (0,0)
        e da una latitudine/longitudine iniziale.
        """
        c=111.32 
        lat0, lon0 = origin_gps
        delta_lat = self.d /c  # gradi di latitudine per ogni step verticale
        delta_lon = self.d/(c * math.cos(math.radians(lat0))) 
        
        for (r, c), attr in self.gf.nodes(data=True): #aggiunge ad ogni nodo la relativa posizione gps
            lat = lat0 + r * delta_lat
            lon = lon0 + c * delta_lon
            attr['gps'] = (round(lat, 6), round(lon, 6))
            
    def assign_bins_to_nearest_nodes(self, bins: Iterable[Bin]):
        """
        Posiziona ogni cassonetto nel nodo più vicino rispetto alle sue coordinate GPS.
        Aggiorna il riferimento del nodo (nd) e la coordinata (gps) del cassonetto.
        """
        # Coordinate GPS dei nodi già calcolate nella funzione _assign_gps_coordinates
        node_gps = nx.get_node_attributes(self.gf, 'gps')

        for B in bins:
            if B.gps is None:
                raise ValueError("Il cassonetto non ha coordinate GPS definite.")
            # Estrae le coordinate GPS del cassonetto
            lat_b, lon_b = B.gps # Supponiamo che siano le coordinate GPS reali
		
	        # Trova il nodo più vicino che non ha già un cassonetto assegnato
            candidate_n = [(nd, coord) for nd, coord in node_gps.items() 
                           if not self.gf.nodes[nd].get('has_bin', False)]

            if not candidate_n: # Per debug
            	raise Exception("Non ci sono più nodi disponibili senza cassonetto!")

            closest_nd = min(candidate_n, key=lambda item: math.dist((lat_b, lon_b), item[1]))[0]

            self.gf.nodes[closest_nd]['has_bin'] = True
            # Assegna il nodo più vicino e aggiorna la posizione del cassonetto
            B.nd = closest_nd
            B.crd = self.node_coord(closest_nd)  # Coordinate del nodo

    def _gen_graph(self) -> Gr:
        """ genera un grafo 2d a griglai 
            i nodi vengono generati ed etichettati in questo modo:
               - prima riga bassa (0, 0), (0, 1),...,(0, n)
                   ...
               - ultima riga alta (n, 0), (n, 1),..., (n, n) """ 
        G = nx.grid_2d_graph(self.nn, self.nn)
        for nd, (x, y) in zip(G.nodes(), gen_coord(self.nn, self.nn, self.d)):
            if nd not in (self.n_in, self.n_out): G.nodes[nd]['color'] = "skyblue" # colore di base
            else: G.nodes[nd]['color'] = "green" # input output
            G.nodes[nd]['population'] = 0
            G.nodes[nd]['coord'] = (x, y) # assegnamo coordinate cartesiane ai nodi
        for i, j in G.edges():
            G[i][j]['distance'] = self.d
        return G
    
    def _get_influencers(self, cz:Citizen):
        """ cerca gli influencer entro un raggio predefinito """
        for nd in gen_neighbor(nd = cz.nd, n_layer = cz.att.n_depth): # circondari da considerare
            for other in self.ctzs.get(nd, ()):
                if (other is not cz) and ((d:= math.dist(cz.xy, other.xy)) <= cz.rof):
                    cz.infs[other] = d
    
    """ Proprietà e metodi di visualizzazione """
    
    def __repr__(self) -> str:
        return f'nodes: {self.n_nodes}, bins: {len(self.bins)}, population: {self.population}'
    
    def show(self, distance:bool = False, population:bool = False, fg_size:tuple[int, int] = (35, 35), nd_size:int = 450):
        """ Mostra il grado della città con indicazioni sul numero di residenti """
        G = self.gf
        palette = tuple(nx.get_node_attributes(G,'color').values())
        pos = {(x, y): (y, x) for x, y in G.nodes()}  # Disposizione a griglia con (x, y) come coordinate
        plt.figure(figsize = fg_size)
        wl = False if population else True
        nx.draw(G, pos, with_labels = wl, node_color = palette, node_size = nd_size, font_weight = "bold")
        
        if distance: 
            distance_dict  = nx.get_edge_attributes(G,'distance')
            nx.draw_networkx_edge_labels(G, pos, edge_labels = distance_dict)
            
        if population:
            population_dict = nx.get_node_attributes(G, 'population')
            pop_labels = {nd: f'{nd}\n{pop}' for nd, pop in population_dict.items()}
            nx.draw_networkx_labels(G, pos, labels = pop_labels, font_size = 8, font_weight = "bold", font_color = 'black')
       
        plt.title("The city")
        plt.show()
    
    @property
    def n_nodes(self) -> int:
        return self.nn**2
    
    @property
    def l_edge(self) -> float:
        return self.d
    
    @property
    def gen_dmat(self)-> DMatrix:
        """ Genera la matrice delle distanze """
        return DMatrix({(n1, n2): 
                            nx.shortest_path_length(self.gf, source = n1, target = n2, weight='distance') 
                            for n1, n2 in itertools.product(self.gf.nodes(), self.gf.nodes())})
    
    
    """ Altri Metodi """        
    
    def iter_citizens(self) -> Generator[Citizen, None, None]:
        """ restituisce tutti i cittadini """
        for cs in self.ctzs.values():
            yield from iter(cs)
    
    def n_citizens(self, nodes:Optional[Iterable[Nd]] = None, flt:Callable[[Citizen], bool] = lambda ctz: True) -> int:
        """ numero di cittadini, eventualmente filtrato per quartiere/nodo e altri criteri """
        if nodes is None: nodes = tuple(self.ctzs.keys())
        return sum(sum(1 for ctz in self.ctzs[nd] if flt(ctz)) for nd in nodes)
    
    def bin_coord(self, B:Bin|int = 0) -> tuple[float, float]:
        """ restituisce la coordinata di un bin. Un po' obsoleto dato
            che ho aggiunto la coordinata al bin stesso """
        if not isinstance(B, Bin): B = self.bins[B] 
        nd = B.nd # il nodo
        return self.gf.nodes[nd]['coord']
    
    def node_coord(self, nd:Nd) -> tuple[float, float]:
        """ le coordinate di un nodo """
        return self.gf.nodes[nd]['coord']
    
    def node_pop(self, nd:Nd) -> tuple[float, float]:
        """ la popolazione di un nodo """
        return self.gf.nodes[nd]['population']
    
    def get_cz(self, nd:Nd, idx:int = 0) -> Citizen|None:
        """ uno specifico cittadino (idx) di un certo nodo (nd) """
        try: return self.ctzs[nd][idx]
        except: return None
        
'''
pop_nr:f_pop = normal_pop(500, 50, 50)
env=sp.Environment()

#Tre tipologie di cittadino 
C1 = Citizen_type(kind ='eco_friendly', green_awareness = (0.65, 0.75), radius_of_influence = (200, 300), n_depth = 1)
C2 = Citizen_type(kind ='neutral', green_awareness = (0.4, 0.5), radius_of_influence = (200, 300), n_depth = 1)
C3 = Citizen_type(kind ='non_eco', green_awareness = (0.1, 0.25), radius_of_influence = (250, 500), n_depth = 1)

#quattro quartieri tipo 
Ng1 = Neighborhood(population = 100, type_prob = {C2:0.4, C3:0.6})
Ng2 = Neighborhood(population = pop_nr, type_prob = {C1:0.2, C2:0.3, C3:0.5})
Ng3 = Neighborhood(population = 150, type_prob = {C1:1})
Ng4 = Neighborhood(population = pop_nr, type_prob = {C1:0.1, C3:0.9})


# I bidoni, messi negli stessi posti in cui verrebero generati in maniera randomica 
lista= [{'Giro': 1, 'Via': 'STRADA BIXIO NINO', 'GPS': '44.797914,10.319654'},
{'Giro': 1, 'Via': 'STRADA CASA BIANCA', 'GPS': '44.789807,10.352405'},
{'Giro': 1, 'Via': 'VIA CASSIO PARMENSE', 'GPS': '44.794498, 10.351678'},
{'Giro': 1, 'Via': 'VIA FRANK ANNA', 'GPS': '44.788984,10.336313'},
{'Giro': 1, 'Via': 'VIA PICASSO PABLO', 'GPS': '44.782622,10.363807'},
{'Giro': 1, 'Via': 'VIA TRAVERSETOLO', 'GPS': '44.782471, 10.34294'},
{'Giro': 1, 'Via': 'VIALE PELACANI BIAGIO', 'GPS': '44.795178, 10.335309'},
{'Giro': 1, 'Via': 'STRADA CASA BIANCA', 'GPS': '44.790902, 10.347451'},
{'Giro': 1, 'Via': 'VIA DE GASPERI ALCIDE', 'GPS': '44.783814, 10.338083'},
{'Giro': 1, 'Via': 'VIA MONTEBELLO', 'GPS': '44.787009,10.337966'},
{'Giro': 1, 'Via': 'VIA ROMA', 'GPS': '44.716245, 10.229817'},
{'Giro': 2, 'Via': 'VIA PONTICELLE ', 'GPS': '44.727148, 10.398692'},
{'Giro': 2, 'Via': 'PIAZZALE M.L. KING - FOGNANO', 'GPS': '44.819897, 10.282101'},
{'Giro': 2, 'Via': 'VIA CREMONESE', 'GPS': '44.812154, 10.292906'},
{'Giro': 2, 'Via': 'VIA EMILIO LEPIDO', 'GPS': '44.791583,10.360084'},
{'Giro': 2, 'Via': 'VIA MURATORI LUDOVICO ANTONIO', 'GPS': '44.785590, 10.365075'},
{'Giro': 2, 'Via': 'VIA S.EUROSIA', 'GPS': '44.784611,10.334970'},
{'Giro': 2, 'Via': 'VIA TERRACINI UMBERTO', 'GPS': '44.786099, 10.351589'},
{'Giro': 2, 'Via': 'VIA VARSAVIA', 'GPS': '44.789562, 10.348233'},
{'Giro': 2, 'Via': 'VIA VENTIQUATTRO MAGGIO', 'GPS': '44.787452, 10.359750'},
{'Giro': 2, 'Via': 'VIA VIETTA - SAN PANCRAZIO', 'GPS': '44.812825, 10.273440'},
{'Giro': 2, 'Via': 'LOC. S. PROSPERO - VIA EMILIO LEPIDO', 'GPS': '44.778938,10.398975'}]


# Creazione oggetti Bin da lista
bins_gps = [
    Bin(env=env, capacity=300, threshold=290, low_treshold=20,
        gps=tuple(map(float, b['GPS'].replace(" ", "").split(','))))
    for b in lista ]

# Crea la mappa dei quartieri
c_map = All_Neighborhood([Ng1, Ng2, Ng3, Ng4])
nn = int(400 ** 0.5)
c_map.all_equal(All_Neighborhood.gen_node(nr=nn, nc=nn), idx_ng=0)  # tutti Ng1

# la città 
C = City(env = env, n_nodes = 400, distance = 0.25, 
         the_map = c_map, bins = bins_gps, in_out = None, start_gps=(44.797914, 10.319654))

def mappa_quartieri_su_mappa(all_neigh, mapping, tipi_ng):
    for nodo, nome_q in mapping.items():
        if nome_q in tipi_ng:
            all_neigh[nodo] = tipi_ng[nome_q]
        else:
            print(f"⚠️ '{nome_q}' non è stato associato a nessun tipo di quartiere.")
cz = C.get_cz((1, 1))


C.show(True, True)
'''
"""
x = {((0,0), (0, 1)): 10, ((0,1), (1, 0)): 20}
nw = DMatrix(x)
y = nw._as_df()
"""

"""
# ⚠️ questo va eseguito *dopo* la creazione della City, es. nel main o in un notebook


"""
"""
# Check su bin specifico
bin_012 = next((b for b in  C.bins if b.nd == (0, 12)), None)
if bin_012 is None:
    print("Nessun bin trovato in (0, 12)")
print("Nodi associati:", bin_012.nd_ctzs[0])
print("Cittadini associati:", bin_012.nd_ctzs[1])

"""
