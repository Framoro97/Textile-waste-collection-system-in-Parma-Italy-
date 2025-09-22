# -*- coding: utf-8 -*-
from __future__ import annotations
"""
from Provolone import Ciao

c = Ciao()
c.saluta()
"""

"""
Created on Tue Dec 17 15:21:34 2024
@author: zammo

La classe che modellizza il processo/agente "cittadino".
Ogni cittadino genera rifiuti tessili e sceglie in base alla sua green awareness se 
conferirli alle isole ecologiche o se smaltirli nell'indifferenziato.
La sua propensione si modifica nel tempo sia in base al livello di servizio percepito
sia in base al comportamento dei suoi vicini.

"""
import numpy as np
import math
import random
from typing import Optional, Iterable, NamedTuple, Callable, Generator
from Time_Calendar import Calendar

from Bins import Bin
from Stats import Trip

Nd = tuple[int, int]

""" **** ALCUNE FUNZIONI PROBABILISTICHE D'APPOGGIO **** """ 
def f_move(d_max:float, v_feet:float, v_car:float) -> Callable:
    """ tipo e tempo di spostamento, 
            se output booleano è vero, allora va a piedi """
    def inner(km:float)-> tuple[bool, float]:
        thr = random.uniform(min(0, d_max*0.9), d_max*1.1) # modifichiamo di volta in volta il treshold
        if km <= thr: return True, round(2*km/v_feet, 4)
        return  False, round(2*km/v_car, 4)
    inner.__name__ = f'f_move(d_max = {d_max})'
    return inner      

def f_influence(alpha:float, beta:float, delta_inc:float, delta_dec:float) -> Callable:
    """ effetto dell'influenza reciproca """
    delta_inc = abs(delta_inc)
    delta_dec = -abs(delta_dec)
    def inner(delta_gaw:float) -> float:
        gamma = 1 - abs(delta_gaw)
        pr = alpha * math.exp(-beta*gamma)
        rnd = random.random()
        if rnd <= pr:
            if delta_gaw < 0: return delta_inc
            else: return delta_dec
        return 0
    inner.__name__ = f'f_infl(a = {alpha}, b = {beta}, d+ = {delta_inc}, d- = {delta_dec})'
    return inner  
     
def f_incentive(low:float, high:float) -> Callable:
    """ effetto dell'incentivo 
            la percetauel d'incremento che verrà usata come fattore moltiplicativo 
            (1 + x) del gr_aw"""
    def inner() -> float:
        return random.uniform(low, high)
    inner.__name__ = f'f_incn({low}, {high})'
    return inner

def logistica(t:float, alpha:float = 1, emme:float = 0, enne:float = 1, tau:float = 0.5) -> float:
    """ alpha asintoto a x infinito
            alpha*(1 + emme) / (1 + enne) valore in zero
                alpha*emme/enne asintoto a x -infinito
                    tau fattore di forma (ripidità) """
    num = 1 + emme*math.exp(-t/tau)
    den = 1 + enne*math.exp(-t/tau)
    return alpha*num/den
    
            
def f_distance(tau:tuple[float, float], traslazione:tuple[float, float], troncamento:float, in_km:bool = True) -> Callable:
    """ 
        in_km = True se i valori sono passati in km e non in metri 
        effetto disincentivante legato alla distanza dal bin più vicino
            se la distanza è troppa la green awareness cala...  
                 il risulato, similmente a quello fi f_incentive 
                        è la percetauel d'incremento che verrà usata come 
                            fattore moltiplicativo  gr_aw  """
    if in_km:
        tau = (1000*tau[0], 1000*tau[1])
        traslazione = (1000*traslazione[0], 1000*traslazione[1])
        troncamento *= 1000

    def inner(t:float) -> float:
        """ una doppia funzione logistica che 
                determina il range di variazione
                    vedi foglio excel allegato """
        if in_km: t *= 1000
        if t <= troncamento: return 0
        estremo_inferiore = logistica(t = t - traslazione[0], alpha = 1, emme = 0, enne = 1, tau = tau[0])
        estremo_superiore = logistica(t = t - traslazione[1], alpha = 1, emme = 0, enne = 1, tau = tau[1])
        if estremo_inferiore > estremo_superiore:
            estremo_superiore, estremo_inferiore = estremo_inferiore, estremo_superiore 
        # return round(estremo_inferiore,4), round(estremo_superiore,4)
        return random.uniform(estremo_inferiore, estremo_superiore)
       
    inner.__name__ = f'f_logistic(tau{tau})'
    return inner
    
def green_variation(delta:float, increase:bool = True) -> float: 
    """ effetto bidone pieno o vuoto """
    if increase: delta = abs(delta)
    else: delta = -abs(delta)
    def inner():
        return random.expovariate(lambd = 1/delta)
    ch = '+' if increase else '-'
    inner.__name__ = f'Grw_expovar{ch}(mu = {delta})'
    return inner
    
def waste_gen(low:int, mode:int, high:int) -> float:
    """ materiale generato """
    def inner():
        return random.triangular(low, mode, high)
    inner.__name__ = f'Wg_triangular({low}, {mode}, {high})'
    return inner

#usare una logaritmica a intervalli
def waste_log_gen(A=1.5, B=0.3, C=0.5):
    """
    Restituisce una funzione che accetta un intervallo (t_min, t_max)
    e genera un valore casuale logaritmico entro quell'intervallo.
    """
    def inner(t_min: float, t_max: float) -> float:
        def funzione_logaritmica(t):
            return A * np.log(B * t + 1) + C

        v_min = funzione_logaritmica(t_min)
        v_max = funzione_logaritmica(t_max)
        return np.random.uniform(v_min, v_max)

    return inner

""" **** LE CLASSI *** """

class C_Trip(Trip):
    """ Classettina per registrare 
        le statistiche dei cittadini """
    def __init__(self, t:float, calendar:Optional[str] = None, n_members:int = 1, avg_gaw_residents:Optional[float] = None):
        super().__init__(t)
        # il nodo in cui è andato 
        # e che ha trovato pieno o vuoto
        self.tr:Nd = tuple()
        self.tr1:Nd = tuple() # il secondo bin, se scelto
        # None se non aveva intenzione di riciclare, 
        # negativo se voleva riciclare, ma ha trovato il bidone pieno
        # il valore negativo è quanto avrebbe conferito se avesse potuto ...
        self.r_kg:float|None = None
        # vero se va a piedi 
        self.bf: bool = False 
        # per debug registriamo anche la variazione di green awareness
        self.var_gaw:float = 0  
        self.n_members = n_members # numero di membri della famiglia, 1 se single
        self.calendar = calendar if isinstance(calendar, str) else repr(calendar) 
        self.avg_gaw_residents = avg_gaw_residents  # la green awareness media dei cittadini assegnati ai nodi agganciati ad un determinato bin
        
        
    def __repr__(self):
        """ kg indifferenziati, kg riciclati, kg che avrebbe riciclato """
        return f'landf.:{self.kg_waste}, rec.:{self.kg_recycled}, r_lost: {self.kg_rlost}'
    
    @property
    def recycle(self) -> bool:
        """ true se ha effettivamente riciclato"""
        # deve essere andato al bin, e averlo trovato vuoto, quindi r_kg positivo 
        return self.tr != () and self.r_kg!=None and self.r_kg > 0
    
    @property
    def tried_recycle(self) -> bool:
        """ true se ha provato a riciclare"""
        # deve essere andato al bin, 
        return self.tr != ()
        
    @property
    def kg_waste(self) -> float:
        """ kg nell'indifferenziato """
        return self.kg - self.kg_recycled 
        
    @property
    def kg_recycled(self) -> float:
        """ kg reciclati """
        if self.recycle: return self.r_kg
        return 0
    
    @property
    def full_bin(self) -> float|None:
        """ True se il bin era pieno """
        if not self.tried_recycle: return None # è andato al bidone, ma r_kg è negativo 
        return 1 - self.recycle
    
    @property 
    def kg_rlost(self) -> float: 
        """ kg che avrebbe differenziato, 
                ma che non ha potuto differenziare """
        if self.full_bin: return abs(self.r_kg)
        return 0
    
    @property
    def kg_lost(self):
        if self.r_kg is not None and self.r_kg<0: return abs(self.r_kg)
        #si può mantenere così perche il metodo Mg.bin_kg non subisce modifiche: 
        #Nel secondo if non c'è la tupla di nodi, quindi non viene considerato tr.tr=()
        
class Citizen_type(NamedTuple):
    """ tupla ... che alla fine poteva essere una classe ....
            che contiene tutti gli attributi che definiscono un cittadino.
                Molti sono "probabilistici" e si concretizzeranno per il cittadino specifico """
    
    """ attributi descrittivi """
    kind:str|None = None
    sex:str|None = None
    instruction_level:str|None = None
    income:float|None = None
    
    """ parametri comportamentali """
    green_awareness:tuple[float, float]|None = None # il valore puntuale verrà creato come uniforme tra i due valori
    radius_of_influence:tuple[float, float]|None = None
    n_depth:int|None = None
    
    """ distribuzioni di probabilità """
    gen_waste:Callable[[], float]|None = None # kg creati
    green_inc:Callable[[], float]|None = None # funzione incremento di green awareness
    green_dec:Callable[[], float]|None = None # funzione decremento di green awareness
    f_influence:Callable[[float], float] = None # funzione passa_parola, influenza tra vicini
    f_incentive:Callable[[], float] = None # funzione incremento di green awareness in base a incentivo, basso, medio, ecc.
    f_dist_penalty:Callable[[float], float] = None # funzione di decremento della green awareness in base alla distanza dal bidone più vicino, generata una volta sola come per l'effetto dell'incentivo
    f_move:Callable[[float], tuple[True, float]] = None # funzione che determina il tipo (True -> a piedi) e il tempo di spostamento dell'utente (usato solo come statistica)
    m_season:tuple[int] = (5, 11) # maggio e novembre come mesi stagionali
    s_factor:float = 2.5 # quando c'è stagionalità la quantità è più che doppia
    
    """ Metodi di supporto """
    
    def _function_list(self) -> tuple[str]:
        """ lista delle funzioni """
        return tuple(key for key, value in self._asdict().items() if callable(value))
    
    def non_null_rep(self, *taboo:str) -> str:
        """ rappresentazione con solo campi valorizzati """
        taboo += self._function_list() # esclude le funzioni!
        non_null_att = [f'{key} = {value}' for key, value in self._asdict().items() if (key not in taboo) and (value is not None)]
        return ", ".join(non_null_att)
    
    def function_desc(self) -> tuple[str]:
        """ lista dei nomi delle funzioni """
        return tuple(getattr(self, foo).__name__ for foo in self._function_list())

class Citizen:
    def __init__(self, env, 
                         xy_coor:tuple[float, float] = (0.0, 0.0), # coordinate x, y
                         reference_node:tuple[int,int] = (0, 0), # nodo di riferimento / quartiere
                         distance_from_node: float = 0.0, # distanza dal nodo di riferimento
                         bins: Optional[dict[Bin, float]] = None, # i bidoni ai quali può andare e la relativa distanza
                         attributes: Optional[Citizen_type] = None, # gli attributi descrittivi
                         influencers: Optional[dict[Citizen, float]] = None, # cittadino e distanza
                         calendar:Optional[Calendar] = None, # il calendario  
                         n_members:int = 1 # numero di membri della famiglia, 1 se single --> la classe Family lo sovrascriverà con il suo
                         ):
    
        """ Serie di attributi che non vanno mai cambiati!!! 
                ... per pigrizia non avevo voglia di mettere _  @_@ """
        self.env = env # l'ambiente simpy
        self.xy = xy_coor # la posizione cartesiana
        self.n_members= n_members 
        self.nd = reference_node # il nodo di riferimento
        self.ds =  distance_from_node # la distanza dal nodo di riferimento 
        self.kg_stored=0
        self.calendar = calendar  # il calendario per la gestione delle date
        self.c=0    #contatore per decidere la quantità di rifiuti da generare usato in confere
        # aggiungiamo la distanza tra la posizione della persona 
        # e il nodo di riferimento, e ordiniamo i bidoni per distanza 
        self.bins:dict[Bin, float] = {} # lista dei bin e relativa distanza
        if bins is not None: self._add_bins(bins) 
        # aggiungiamo gli influencers, i vicini che possono influenzare il comportamento 
        self.infs:dict[Citizen, float] = influencers if influencers is not None else {} # i vicini e la loro distanza 
        # gli attributi descrittivi del cittadino
        self.att:Citizen_type  =  attributes 
        """ attributi che vengono calcolati """
        # la green awareness di base viene calcolata usando un'uniforme tra un minimo e un massimo
        # tale valore può cambiare se c'è un incentivo ...
        self.gr_aw = round(random.uniform(*self.att.green_awareness), 4) # green awareness calcolata come uniforme
        if self.att.f_incentive is not None: self.gr_aw *= 1 + round(self.att.f_incentive(), 3) # eventualmente aumentata
        if self.att.f_dist_penalty is not None and self.bins != {}: self._distance_penalty() # riduce la gaw per effetto della distanza dal bidone più vicino
        self.rof = round(random.uniform(*self.att.radius_of_influence), 4) # il raggio d'influenza calcolato come uniforme
        
        """ attributi aggiunti """
        self.n_trip:int = 0 # numero totale di conferimenti
        self.sts:dict[int:C_Trip] = {} # le statistiche 1:T1, 2:T2, ...
        self._avg_gaw:float = 0.1 # la green awareness media, fissata ad un valore casuale tanto verrà aggiornata
        self._inf_delta_gaw:list[float] = [] # per debug registriamo le variazioni di green awareness per effetto dell'influenza
        # self._wgh è un altro attribut che viene aggiungo dal metodo _add_bins
        self.seasonal_conferred: bool = False     # Flag: già fatto il conferimento stagionale?
        self.last_month: int = -1                 # Per tenere traccia del mese corrente
    
    """ funzioni usate in __init__ """    
    
    def _add_bins(self, bins: dict[Bin, float]):
        """ aggiunge i bin e la loro distanza (passata in input), 
                alla distanza originaria (calcolata sul grafo)
                    viene aggiunto lo spostamento necessario a 
                        raggiungere il nodo di riferimento """
        self.bins.update({B: d + self.ds for B, d in bins.items()})
        self.bins = dict(sorted((Bd for Bd in self.bins.items()), key = lambda x: x[1])) # ordinati per distanza
        self._wgh = self._make_wgh() # genera i pesi per la selezione
    
    def _distance_penalty(self):
        # se c'è l'effetto di penalizzazione in base alla distanza, 
        # aggiorna la green awareness
       
        penalty = self.att.f_dist_penalty
        closest_bin = tuple(self.bins.keys())[0] # il primo
        min_dist = self.bins[closest_bin]
        self.gr_aw *= (1 - round(penalty(min_dist), 4))
    
    def _make_wgh(self) -> tuple[float]:
        """ pesi inversamente proporzionali alla distanza """
        inv_dst = tuple(1/max(d, 0.1) for d in self.bins.values()) # l'inverso della distanza
        return tuple(round(max(d, 0.1)/sum(inv_dst), 4) for d in inv_dst)
    
    """ Proprietà o funzioni descrittive """    
    def __repr__(self) -> str:
        s = self.att.non_null_rep("green_awareness", "radius_of_influence")
        return f'{s}, rof: {self.rof}, gr_aw: {self.gr_aw}, pos:{self.xy}, node: {self.nd}, infs: {self.n_influencers}'
    
    def __getitem__(self, idx:int) -> C_Trip:
        """ restituisce una statistica di un viaggio specifico """
        return self.sts.get(idx, None)
    
    def __iter__(self) -> Generator[C_Trip, None, None]:
        for tr in self.sts.values():
            yield tr    
    
    @property
    def n_influencers(self) -> int:
        """ numero d'influencers """
        try: return len(self.infs)
        except: return 0
    
    @property
    def show_distrib(self) -> tuple[str]:
        """ il nome delle distribuzioni utilizzate """
        return self.att.function_desc()
    
    @property
    def tot_trip(self) -> int:
        """ numero totale di conferimenti """
        return self.n_trip
    
    @property 
    def tot_kg(self) -> float:
        """ totale generato """
        return sum(trip.kg for trip in self)
    
    @property 
    def tot_rkg(self) -> float:
        """ totale riciclato """
        return sum(trip.kg_recycled for trip in self)
    
    @property 
    def tot_lost(self) -> float:
        """ totale indifferenziato dopo essere andato al bidone"""
        return sum(trip.kg_rlost for trip in self)
        
    @property 
    def tot_km(self):
        """ km in auto percorsi """
        return sum(trip.km for trip in self if trip.bf == False)
    
    @property
    def n_recycle(self) -> int:
        """ numero di volte che ha riciclato """
        return sum(trip.recycle for trip in self) 
    #trip viene estratto dal metodo iter, come scrivere self.sts.values()
    
    @property
    def n_tried_recycle(self) -> int:
        """ numero di volte in cui è andato al bidone
                per ricilcare, indipedentemetne dal fatto
                    che l'abbia trovato pieno o vuote  """
        return sum(trip.tried_recycle for trip in self)
    
    @property
    def n_full(self) -> int:
        """ numero di volte in cui il bidone era pieno """
        return sum(trip.full_bin for trip in self if trip.full_bin is not None)
   
    @property
    def s_level(self) -> float:
        """ livello di servizio """
        nr = self.n_tried_recycle
        nf = self.n_full
        # giusto per debug, 
        # controllo su nf e nr
        if nf > nr: 
            breakpoint()
            raise Exception("Valori incongruenti in People, per il calcolo del livello di servizio")
        if nr == 0: return math.nan
        return round(1 - nf/nr, 4)
    
    @property
    def used_bins(self) -> dict[Nd, dict[str, float]]:
        """ numero di volte 
                in cui è andato a ciascun bin 
                    con indicazione del quantitativo conferito """
        stat = {}
        for tr in self:
            if (nd:= tr.tr) != ():
                sub_dict = stat.setdefault(nd, {'n_go' : 0, 'kg_tot' : 0, 'kg_rlost' : 0})
                stat[nd] = {key:val + dv for (key, val), dv in zip(sub_dict.items(), (1, round(tr.kg_recycled, 2), round(tr.kg_rlost, 2)))}
        return stat
    
    def gr_aw_mod(self, labels:tuple[str] = ('eff_servizio', 'eff_influenza')) -> dict[str, list[float]]:
        """ mostra le variazioni di green awareness che sono occorse durante la simulazione """
        out = {}
        out[labels[0]] = [round(tr.var_gaw, 4) for tr in self] 
        out[labels[1]] = [round(dg, 4) for dg in self._inf_delta_gaw]
        return out
     
    def choose_bin(self, rnd=True) -> tuple[Bin, list[Bin]]:
        """
    Restituisce:
    - Il bidone scelto (più vicino o casuale)
    - Una lista di bidoni vicini (distanza < 0.5 dal bin scelto), escluso lui stesso
        """
        keys = list(self.bins.keys())
        dists = list(self.bins.values())

        idx = 0 if not rnd else random.choices(range(len(keys)), self._wgh, k=1)[0]
        chosen_bin = keys[idx]

        near_bins = []
        for offset in [-2, -1, 1, 2]:
            i = idx + offset
            if 0 <= i < len(keys):
                dist_diff = abs(dists[idx] - dists[i])
                if dist_diff < 0.5:
                   near_bins.append([keys[i],dist_diff])
    
        closest_nearby = min(near_bins, key=lambda x: x[1])[0] if near_bins else None
        return chosen_bin, closest_nearby

    """ Metodi legati al processo di conferimento dei rifiuti """
    '''
    def choose_bin(self, rnd=True) -> tuple[Bin, Bin | None]:
        """
    Restituisce una tupla con:
    - Il bidone scelto (casuale o il più vicino)
    - Il secondo più vicino (se esiste), per fallback o controllo aggiuntivo.
        """
        keys = tuple(self.bins.keys())

        if not rnd:
            return (keys[0], keys[1] if len(keys) > 1 else None)

        idx = random.choices(range(len(keys)), self._wgh, k=1)[0]
        sorted_by_dist = sorted(self.bins.items(), key=lambda x: x[1])
        chosen_bin = keys[idx]

        # Trova il secondo più vicino diverso dal primo
        second_bin = next((b for b, _ in sorted_by_dist if b != chosen_bin), None)

        return (chosen_bin, second_bin)
    '''
    '''
    def choose_bin(self, rnd = True) -> Bin:
        """ restituisce il più vicino, 
            oppure un bin casuale scelto in base alla distanza """
        idx = 0 if not rnd else random.choices(tuple(range(len(self.bins))), self._wgh, k = 1)[0]
        return tuple(self.bins.keys())[idx]
    #devo farne restituire 2
    '''
    def _recycle(self) -> bool:
        """ riciclo oppure no? """
        rnd = random.random()
        return rnd <= self.gr_aw
    
    def _seasonality(self) -> float:
            """ se il mese è stagionale restituisce il fattore stagionale
                                altrimenti restituisce 1 """
            if self.calendar is not None and self.calendar.month in self.att.m_season: return self.att.s_factor
            return 1   

    def gen_and_confere(self, rnd_bin: bool = True): #FRA_Integrato con stagionalità, gaw_settimanale delle zone assegnate a bin, e calendario x estrazione
        """ Genera una quantità di rifiuti e, se le condizioni sono soddisfatte,
    tenta il conferimento presso i bidoni disponibili.

    - Crea una nuova istanza di C_Trip a ogni generazione.
    - La decisione di riciclare dipende dal superamento della green awareness.
    - La generazione è soggetta a stagionalità e dimensione del nucleo familiare."""
        self.n_trip += 1

        target_bin, _ = self.choose_bin(rnd=rnd_bin) #prendo solo il primo dei return che da choose_bin, 
                                                     #ovvero il bin più vicino al cittadino

        avg_gaw_residents = target_bin.avg_gaw  # valore gaw_settimanale dei residenti ai nodi associati al bin aggiornato

        Tr = C_Trip(t=round(self.env.now, 4), n_members=self.n_members, calendar=self.calendar 
                    if isinstance(self.calendar, str) else repr(self.calendar),
                    avg_gaw_residents=avg_gaw_residents)  # Crea una nuova istanza di C_Trip
        # Recupero il mese corrente

        curr_month = self.calendar.month if self.calendar else -1
        # Se è cambiato il mese → resetto il flag
        if curr_month != self.last_month:
            self.seasonal_conferred = False  # boolean settato per monitorare che avvenga una sola generazione stagionale alla volta
            self.last_month = curr_month

        # Calcolo del fattore stagionale effettivo
        s_factor = self._seasonality()

        if s_factor > 1.0 and not self.seasonal_conferred:

            self.seasonal_conferred = True
            #print(f"[{self.calendar}] Generazione maggiorata attivata - agente: {self}, mese: {self.calendar.month}")
            # Se siamo in un mese stagionale e non abbiamo ancora effettuato una generazione maggiorata,
            # usiamo il moltiplicatore. Altrimenti, generazione standard.
        else:
            s_factor = 1.0  # annulla effetto se già generato nel mese

        new_kg = round(self.att.gen_waste(0, 3.3), 2) * s_factor * self.n_members
        self.kg_stored += new_kg
        Tr.kg = new_kg

        soglia_conf = 1.5 * self.n_members   # soglia aumentata proporzionalmente

        if self._recycle(): #True o false in base a gaw, moneta lanciata ogni qualvolta il cittadino genera
            if self.kg_stored >= soglia_conf:
                B1, B2 = self.choose_bin(rnd_bin)
                Tr.tr = B1.nd
                Tr.km = 2 * self.bins[B1]

                if B1.put(self.kg_stored):
                    Tr.r_kg = self.kg_stored
                    Tr.var_gaw = self.att.green_inc()
                    self.gr_aw += Tr.var_gaw
                    self.kg_stored = 0.0
                else:
                    if B2:
                        if self.gr_aw > random.uniform(0.45, 0.65):
                            if B2.put(self.kg_stored):  # verifica conferimento
                                Tr.tr1 = B2.nd
                                Tr.km += 2 * self.bins[B2]
                                Tr.r_kg = self.kg_stored
                                Tr.var_gaw = self.att.green_inc()
                                self.gr_aw += Tr.var_gaw
                                self.kg_stored = 0.0
                                B1 = B2 # Aggiorno B1 perché il conferimento è avvenuto su B2, e il resto del codice usa B1
                            else:  # secondo bidone pieno
                                Tr.tr1 = B2.nd
                                Tr.km += 2 * self.bins[B2]
                                Tr.r_kg = -self.kg_stored
                                Tr.var_gaw = self.att.green_dec() * 2  # doppia penalità due volte ha provato a conferire
                                self.gr_aw += Tr.var_gaw
                                self.kg_stored = 0
                                B1 = B2 # stessa cosa di sopra, in questo caso perché l'ultimo mancato conferimento avviene sul secondo bin
                        else:  # Primo bidone pieno, e utente non prova il secondo
                            Tr.r_kg = -self.kg_stored
                            Tr.var_gaw = self.att.green_dec()
                            self.gr_aw += Tr.var_gaw
                            self.kg_stored = 0
                    else:  # Nessun secondo bidone disponibile
                        Tr.r_kg = -self.kg_stored
                        Tr.var_gaw = self.att.green_dec()
                        self.gr_aw += Tr.var_gaw
                        self.kg_stored = 0

                if self.att.f_move is not None:
                    bin_used = B2 if Tr.tr1 else B1 #Già presente associazione B1 = B2 nel codice sopra quando il tentato conferimento va sul secondo bin
                    by_foot, tr_time = self.att.f_move(self.bins[bin_used])
                    Tr.bf = by_foot
                    Tr.t_end = Tr.t_str + tr_time
            else:
                Tr.r_kg = 0  # evitare errori
                Tr.bf = True
        else: # anche se Recycle = False, la gaw del cittadino può aumentare o diminuire!
            if self.gr_aw < random.uniform(0.05, 0.45):
                self.kg_stored = 0
                Tr.bf = True
                Tr.var_gaw = self.att.green_dec()
                self.gr_aw += Tr.var_gaw
                Tr.r_kg = 0
            else: 
                Tr.bf = True
                Tr.var_gaw = self.att.green_inc()
                self.gr_aw += Tr.var_gaw
                Tr.r_kg = 0
       
        self._check_gaw()
        self.sts[self.n_trip] = Tr
            
    def pr_confere_waste(self, mean_time:int, rnd_bin = False):
        """ il processo simpy se attivato per ogni persona """
        while True:
            time = round(random.expovariate(lambd = 1/mean_time), 4)
            yield self.env.timeout(time)
            self.gen_and_confere(rnd_bin)
    
    def compute_av_gaw(self, wgh: bool = True, min_dist:float = 1) -> float:
        """ calcola la media pesata della green awareness nel vicinato"""
        if self.infs == (): return self.gr_aw # se non ci sono influencer la media coincide con sè stesso, e quindi il gap sarà nullo
        if wgh: 
            t_gaw, t_inv_dist = 0, 0
            for cz in self.infs:
                inv_dist = 1/max(min_dist, self.infs[cz])
                t_gaw += cz.gr_aw * inv_dist
                t_inv_dist += inv_dist  
            return t_gaw/t_inv_dist
        else: 
           return sum(cz.gr_aw for cz in self.infs)/self.n_influencers
    
    def update_gaw(self, avg_gaw:Optional[float] = None):
        """ ricalcola la green awareness che si modifica
                 per effetto dell'influenza esercitata dai vicini """
        if avg_gaw is None: avg_gaw = self._avg_gaw
        delta_gaw = self.gr_aw - avg_gaw # il delta rispetto alla media pesata della green awareness
        inc_dec_gaw = self.att.f_influence(delta_gaw) # calcola la probabilità di cambiamento e l'eventuale cambiamento
        self.gr_aw += inc_dec_gaw
        self._inf_delta_gaw.append(inc_dec_gaw)
        self._check_gaw()
    
    def _check_gaw(self):
        if self.gr_aw > 1: self.gr_aw = 1
        if self.gr_aw <= 0: self.gr_aw = 0.0001

class Family(Citizen): #eredita citizen
    def __init__(self, *args, n_members: int = 3, **kwargs): 
        super().__init__(*args, n_members=n_members, **kwargs) #sovrascrive n_members STD di Citizen con quelli impostati in init
        #modifica di conferimento. Aumento kg generati famiglia
        self.n_members = n_members
        # Puoi aumentare i kg generati o altri fattori usando n_members come moltiplicatore

"""
foo = f_influence(alpha = 0.9, beta = 2.5, delta_inc=0.01, delta_dec = 0.02)
for gap in (-0.5 + i * 0.1 for i in range(11)):
    print(gap, ":", foo(gap))
         
C1 = Citizen_type(kind ='eco_friendly', green_awareness = (0.65, 0.75), radius_of_influence = (200, 300), n_depth = 1,
                  gen_waste = waste_gen(10,30, 40), green_inc = green_variation(delta = 0.01, increase = True),
                  green_dec = green_variation(delta = 0.02, increase = False),
                  f_influence = f_influence(alpha = 0.5, beta = 2.5, delta_inc = 0.01, delta_dec = -0.02))
                    
                      
x = C1.function_desc()

Cz = Citizen(env = None, xy_coor = (10, 12), reference_node = (0, 0), bins = None, attributes = C1, influencers = None)
"""


"""
if __name__ == "__main__":
    print("Test del modulo Citizen")

    # Definisci un profilo di test
    C1 = Citizen_type(
        kind='eco_friendly',
        green_awareness=(0.65, 0.75),
        radius_of_influence=(200, 300),
        n_depth=1,
        gen_waste=waste_log_gen(),
        green_inc=green_variation(delta=0.01, increase=True),
        green_dec=green_variation(delta=0.02, increase=False),
        f_influence=f_influence(alpha=0.5, beta=2.5, delta_inc=0.01, delta_dec=0.02),
    )

    # Crea un cittadino
    Cz = Citizen(env=None, xy_coor=(10, 12), reference_node=(0, 0), attributes=C1)

    # Fai un conferimento
    Cz.gen_and_confere()

    # Stampa la statistica
    print("Risultato conferimento:")
    print(f"Rifiuti generati: {Cz[1].kg}")
    print(f"Ha provato a riciclare? {Cz[1].tried_recycle}")
    print(f"Ha riciclato? {Cz[1].recycle}")
    print(f"Quantità persa (non riciclata): {Cz[1].kg_rlost}")
    """


