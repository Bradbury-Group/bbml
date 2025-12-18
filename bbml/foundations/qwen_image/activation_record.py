




import statistics
from collections import defaultdict


class ActivationReport:
    def __init__(self):
        self.clear()
    
    def record(self,name,mean,max):
        if not self.active:
            return
        self.recorded_mean[name].append(mean)
        self.recorded_max[name].append(max)
    
    def __getitem__(self, name):
        return self.recorded_mean[name], self.recorded_max[name]
    
    def summary(self):
        return {
            name: statistics.mean(mean)
            for name, mean in self.recorded_mean.items()
        }, {
            name: statistics.mean(max)
            for name, max in self.recorded_max.items()
        }
    
    def __str__(self):
        return "ActivationReport: "+str(self.summary())
    
    def clear(self):
        self.recorded_mean = defaultdict(list)
        self.recorded_max = defaultdict(list)
        self.active = True
