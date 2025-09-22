import simpy as sp
from itertools import cycle

class Calendar():
    def __init__(self, env):
        self.env = env
        self.date = {'yy':1, 'mm':1, 'dd':1}
    
    def make_calendar(self, show:bool = False):
        days = cycle((31,28,31,30,31,30,31,31,30,31,30,31))
        while True:
            n_days = next(days)
            for day in range(1, n_days + 1):
                print(self)
                yield self.env.timeout(24)
                self.date['dd'] = day if day <= n_days else 1
            self.date['mm'] +=1
            if self.date['mm']  == 13:
                self.date['mm'] = 1
                self.date['yy'] += 1
    
    def __repr__(self) -> str:
        return f'{self.day}.{self.month}.{self.year}'       
    
    @property
    def day(self):
        return self.date['dd']
    
    @property
    def month(self):
        return self.date['mm']
    
    @property
    def year(self):
        return self.date['yy']

    def __str__(self):
        return f'DATA - Giorno {self.day} - Mese {self.month} - Anno {self.year}'
    
    def __repr__(self) -> str:
        return f'{self.day}.{self.month}.{self.year}' 
    
    """ FRA_Test calendar
    env = simpy.Environment()
cal = Calendar(env)  # qui crei l'oggetto
calendar_gen = cal.make_calendar()  # qui crei il generatore

for _ in range(10):  # simula 10 giorni
    next(calendar_gen)
    env.run(until=env.now + 24)
    print(cal.date)
    """