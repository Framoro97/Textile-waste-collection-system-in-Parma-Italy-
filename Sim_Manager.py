# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:27:02 2024
@author: zammo

Il gestore che organizza la simulazione 
lanciando i vari processi.
Si occupa anche di raccogliere e visualizzare le statistiche. 

"""
from Time_Calendar import Calendar
from typing import Callable, Optional, Any, Iterable, NamedTuple
from functools import partial
import simpy as sp
import random
import matplotlib.pyplot as plt
import sys
import math

from Map import City
from Vehicles import Truck
from Bins import Bin
from People import Citizen

Nd = tuple[int, int] # tipo nodo

class Manager:
    def __init__(self, env, the_city: City, trucks:Iterable[Truck], calendar:Optional[Calendar] = None):
        """ I truck devono già essere stati assegnati ai nodi! """
        self.env = env
        self.city:City = the_city
        self.tr_bin: dict[Truck, tuple[Bin]] = {tr:tuple(tr.bins.values()) for tr in trucks}
        self.ss_gaw:dict[str, list[float]] = {kd:[] for kd in self.city.map.czk} # la green awareness per tipologia di cittadino
        self.ss_gaw['global'] = []
        self.gaw_settimanale = {} #FRA_la gaw settimanale dei cittadini che risiedono nei nodi serviti dai bidoni
        self.calendar = calendar
        
    """ Metodi e Proprietà di visualizzazione e statistiche """
    
    def __repr__(self):
        return f'{self.city}, n_truck = {len(self.tr_bin)}, n_bins = {sum(len(b) for b in self.tr_bin.values())}'
    
    def show_green_awareness(self, *ct_kind:str):
        """ grafica la green awareness di ogni gruppo """
        if not ct_kind: ct_kind = self.ss_gaw.keys()
        n_points = len(self.ss_gaw['global']) # la numerosità dei dati
        x = list(range(0, n_points))
        for lbl in ct_kind: plt.plot(x, self.ss_gaw[lbl], label = lbl)
        plt.xlabel('i-th upgrade') #Aggiornamento i-esimo
        plt.ylabel('Green Awareness')
        plt.legend()
        plt.show()
    
    def _get_bin(self, bin_idx:Optional[int] = None, bin_nd:Optional[Nd] = None) -> Bin|tuple[Bin]:
        """ ricerca o per nodo o per indice, 
                si suppone che ogni truck visiti bidoni distinti"""
        bins = tuple()
        for bs in self.tr_bin.values(): 
            if bin_nd is not None:
                for b in bs:
                    if b.nd == bin_nd: return b
            bins += bs
        if bin_idx is not None: return bins[bin_idx]
        return bins
    
    def show_bin_trend(self, idx_bin:int, coor_bin:Optional[Nd] = None, 
                                   t_start:int = 0, t_end:int  = math.inf):
        """ mostra il grafico dell'andamento del riempimento di un bin """
        b:Bin = self._get_bin(idx_bin, coor_bin)
        b.show_pattern(t_start, t_end)
        
    def show_bins_stat(self, *bin_idx:int) -> dict[int, dict[str, float]]:
        """ statistiche riassuntive di tutti o di alcuni bin
                    totale conferito, extra non conferito   
                           numero di volte pieno, ecc. """
        
        def _is_in(val:int) -> bool:
            """ true se il bin interessa """
            if bin_idx == (): return True # se tupla vuota vanno bene tutti
            return val in bin_idx
        
        # i bidoni che ci interessano
        bins = tuple(b for idx, b in enumerate(self._get_bin()) if _is_in(idx)) 
        # le statistiche di ogni bin d'interesse
        bin_stat = {b.nd: b.stat for b in bins}
        # aggiungiamo le statistiche di conferimento:
        conf_stat = self._bin_kg() 
        for bin_node, node_stat in conf_stat.items():
            if bin_node in bin_stat.keys(): bin_stat[bin_node].update(node_stat)
        return bin_stat
    
    def _bin_kg(self):
        """ kg conferiti e kg che avrebbero potuto essere conferiti 
                ad ogni bin """
        conf_kg = {} # conterrà un dizionario bin_stat per ogni bin
        for cz in self.city.iter_citizens():
            for bin_node, node_stat in cz.used_bins.items(): # used_bins:dict[Nd, dict[str, float]]  
                bin_stat = conf_kg.setdefault(bin_node, {key:0 for key in node_stat.keys()}) 
                conf_kg[bin_node]= {key:round(val + node_stat[key], 2) for key, val in bin_stat.items()}
        return conf_kg
            
    def show_trucks_stat(self) -> dict[int, dict[str, float]]:
        """ statistiche riassuntive di tutti i truck"""
        tr = tuple(self.tr_bin.keys())
        attr = ('n_trips', 'n_calls', 'tot_km', 'avg_n_sub_tr', 'avg_time', 'avg_km', 'avg_usage')
        stat = {idx:{att:getattr(tr, att) for att in attr} for idx, tr in enumerate(tr)}
        return stat
    
    def show_ctzs_stat(self, ct_kind:Optional[str] = None) -> dict[str, dict[str, float]]:
        """ le principali statistiche dei cittadini, medie  e cumulate """
        if ct_kind is None: ct_kind = self.city.map.czk # tutti i tipi di cittadini
        else: ct_kind = (ct_kind, )
        cum_vals = {'tot_kg':0, 'tot_rkg': 0, 'tot_lost':0, 'tot_km':0, 'n_trip': 0, 'n_full':0, 'n_tried_recycle':0}
        n_record = cum_vals.copy()
        for cz in self.city.iter_citizens():
            if cz.att.kind in ct_kind:
                for att in cum_vals.keys():
                    val = getattr(cz, att)
                    if not math.isnan(val) and val is not None: 
                        cum_vals[att] += val
                        n_record[att] += 1    
        cum_vals = {k:round(cum_vals[k], 2) for k in cum_vals.keys()}
        mean_vals = {k:round(cum_vals[k]/n_record[k], 2) for k in cum_vals.keys() if k != 'n_full'}    
        nr, nf = cum_vals['n_tried_recycle'], cum_vals['n_full']
        if nf > nr :
            breakpoint()
            raise Exception('Valori incongruenti in Sim_Manager per il calcolo del livello di servizio')
        mean_vals['s_level'] = round(1 - nf/nr, 3)
        return {'cum_vals': cum_vals, 'mean_vals': mean_vals}
    
    def show_ctzs_gaw_variation(self) -> dict[str, dict[str, float]]:
        """ la variazione media di green awareness per tipo di cittadino, 
            dovuta ai due effetti """ 
        variations = {key:{'eff_servizio': 0, 'n_ser': 0, 'eff_influenza' : 0, 'n_inf' : 0}  for key in self.city.map.czk}
        for cz in self.city.iter_citizens():
            d_es, d_ei = tuple(cz.gr_aw_mod().values()) # le liste contenenti le variazioni legate ai due effetti
            variations[cz.att.kind]['eff_servizio'] += sum(d_es)
            variations[cz.att.kind]['eff_influenza'] += sum(d_ei)
            variations[cz.att.kind]['n_ser'] += len(d_es)
            variations[cz.att.kind]['n_inf'] += len(d_ei)
        return {key:{'eff_servizio': round(variations[key]['eff_servizio']/variations[key]['n_ser'],3), 
                     'eff_influenza': round(variations[key]['eff_influenza']/variations[key]['n_inf'],3)
                     } for key in variations.keys()
                }
            
            
    def tot_cost(self, truck_amm:float, bin_amm:float, euro_km:float, euro_kg:float) -> float:
        """ banale funzione di costo del sistema """
        n_truck = len(tuple(self.tr_bin.keys()))
        n_bin = len(self._get_bin())
        r_kg = sum(cz.tot_rkg for cz in self.city.iter_citizens())
        tot_km = sum(tr.tot_km for tr in self.tr_bin.keys())
        return truck_amm*n_truck + bin_amm*n_bin + r_kg*euro_kg + tot_km*euro_km 
        
    """ I Processi Simpy """
    
    def gen_waste_single_citizen(self, mean_time:int, rnd_bin:bool = True):
        """ lancia il processo di generazione e smaltimento rifiuti 
                    per ogni singolo cittadino """
        for cz in self.city.iter_citizens():
            self.env.process(cz.pr_confere_waste(mean_time, rnd_bin))
    
    def gen_waste_all_citizen(self, mean_time:int, rnd_bin:bool = True):
        """ processo eterno che fa spostare un cittadino alla volta,
            in modo da non avere enne_mila processi paralleli da gestire """
        all_citizens = tuple(self.city.iter_citizens())
        mean_time = mean_time/len(all_citizens)
        while True:
            cz = random.choice(all_citizens)
            dt = round(random.expovariate(lambd = 1/mean_time), 4)
            yield self.env.timeout(dt)
            cz.confere_waste(rnd_bin)
    
    def update_green_awareness(self, dt:int, min_dist:float, wgh: bool = True):
        """ Processo periodico di aggiornamento della green awareness
                se wgh è true si usa l'inverso della distanza come elemento di peso,
                    min dist è la distanza minima che viene utilizzata nel calcolo """
        while True:
            
            stat = {kd: 0 for kd in self.ss_gaw.keys()}
            
            for cz in self.city.iter_citizens():
                # calcola la average green awareness dei vicini di ogni cittadino
                cz._avg_gaw = cz.compute_av_gaw(wgh = wgh, min_dist = min_dist) 
                # registra in senso cumulato la average green awareness di ogni cittadino
                stat['global'] += cz.gr_aw
                stat[cz.att.kind] += cz.gr_aw
           
            # usa i valori cumulati per calcolare e registrare la media per tipologia """
            for kd in self.ss_gaw.keys(): 
                self.ss_gaw[kd].append(round(stat[kd]/self.city.population[kd], 5))
            yield self.env.timeout(dt)
            
            # esegue l'aggiornamento della green awareness di ogni cittadino
            for cz in self.city.iter_citizens():
                cz.update_gaw()
                
  

    def schedule_trucks(self, Tr:Truck, dt:int):
        """ Processo di gestione dei truck 
                va istanziato per ogni truck utilizzato nella simuazione! """
        corrected_dt = dt
        while True:
            intertempo_fisso = self.env.timeout(corrected_dt)
            segnali = [B.tr for B in Tr.bins.values()] # gli eventi segnale
            # parte o a intervallo fisso, o su segnalazione degli smart bin
            evento_triggher = yield sp.events.AnyOf(env = self.env, events = [intertempo_fisso] + segnali)
            start_time = self.env.now
            # si richiede la risorsa (truck) per evitare che possa partire prima di aver ultimato il giro
            with Tr.request() as req:
                on_call = False if intertempo_fisso in evento_triggher else True
                yield self.env.process(Tr.collect_from_bins(on_call))
            # si registra il tempo che manca alla prossima chiamata
            if self.env.now - start_time >= dt: corrected_dt = 0
            else: corrected_dt = round(dt - (self.env.now - start_time), 4)
    
    def progress(self, tot_time:int, perc:float = 0.01):
        n_print = 1/perc
        dt = tot_time / n_print
        pc = perc
        while True:
            yield self.env.timeout(dt)
            print(f'{round(self.env.now,0)} - {perc:.2%}') 
            perc += pc

  # FRA_ Aggiornamento della green awareness media per i cittadini che risiedono nei nodi serviti dai bidoni
    
        """
        Monitora settimanalmente la green awareness media per ciascun nodo (dove vivono i cittadini)
        e la salva nel dizionario `self.gaw_settimanale` sotto chiave temporale `self.env.now`.

        Parametri:
        - step: passo temporale (default 7*24 ore = una settimana, se tempo espresso in ore)
        """
    def monitor_avg_gaw(self, step=7*24):
        #print(f"[DEBUG] Avviato monitor_avg_gaw a t={self.env.now}")
        while True:
            #print(f"[init] Ora t = {self.env.now}")
            yield self.env.timeout(step)  # aspetta il prossimo step (es. una settimana)
            #print(f"[post-yield] Ora t = {self.env.now}")
            #print(f"[DEBUG] snapshot a t={self.env.now}")
            snapshot = {}  # dizionario temporaneo: nodi → media green awareness
            for B in self.city.bins:  # scorre tutti i bidoni
                residents = B.nd_ctzs[1]  # cittadini che vivono nel nodo servito dal bidone
                if residents:
                    avg = sum(cz.gr_aw for cz in residents) / len(residents)
                else:
                    avg = 0.0
                snapshot[B.nd] = avg # associa la media al nodo
                B.avg_gaw = avg  # valore aggiornato, settimanale, usato dal cittadino
            self.gaw_settimanale[self.env.now] = snapshot  # registra lo snapshot settimanale



            
                
