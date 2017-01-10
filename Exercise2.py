# -*- coding: utf-8 -*-
"""
Created on Tue May 10 05:59:11 2016

@author: kramerPro
"""

import time
import simpy
import random
from functools import partial, wraps
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

'''Need to make methods to populate the logs, but just making a global 
dataframe for now'''
##part frame of reference
#part_log = pd.DataFrame()
#pcolumns
## do stuff to initialize the part_log
#machine_log = pd.DataFrame()
#parts = []


class Part(object):
    """ 
    part class is a process containing all the processes needed to make a
    part and logs of the simulation from a part perspective
    """
    def __init__(self, env, name):
        self.env = env
        self.name = name
        # data is just a list. I'm going to append data and process later
        self.data = []
        self.part_process(env)
        self.pass_inspection = -1
        self.time_in_queue = []
        self.enter_system = self.env.now
        self.leave_system = -1
        self.time_in_system = -1

    def use_machines(self, env, machines):
        for machine in machines:
            with machine.request() as req:
                self.data.append((self.name, machine.name, "enter queue", env.now))
                yield req
                self.data.append((self.name, machine.name, "start", env.now))
                self.time_in_queue.append((
                self.name, machine.name, self.data[-1][3]-self.data[-2][3]))
                yield env.timeout(machine.get_time())
                # don't like having to specify each type, but I 
                # don't know how to get them in order in a better way now
                if type(machine) == Inspection:
                    self.pass_inspection = machine.inspect()
                    self.leave_system = self.env.now
                    self.time_in_system = self.leave_system - self.enter_system
                self.data.append((self.name, machine.name, "leave", env.now))
    
    def part_process(self, env):
        """ 
        The machines are hard coded. Should generalize at some 
        point, but this needs to be updated for each individual 
        simulation. Machines need to be initialized globally. env
        also global
        """
        # one series process
        self.env.process(self.use_machines(env, machines))



class Machine(simpy.Resource):
    """
    gives machine a name, uses a global log if needed, overloads request
    and release methods for monitoring. includes an abstract method to
    enforce the definition of a process time (get_time()
    in the specific machine
    """
    def __init__(self, env, name, capacity=1):
        simpy.Resource.__init__(self, env, capacity=1)
        self.env = env
        self.name = name
        self.data = []
        self.parts_made = 0
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_time(self):
        """Time spent at station"""
        pass

    def request(self, *args, **kwargs):
        self.data.append((self._env.now, len(self.queue), self.count,
        "request", self.name))
        return simpy.Resource.request(self, *args, **kwargs)

    def release(self, *args, **kwargs):
        self.data.append((self._env.now, len(self.queue), self.count,
        "release", self.name))
        if len(self.queue) > 1:
            self.data.append((self._env.now, len(self.queue), self.count,
                             "auto-req", self.name))
        self.parts_made +=1
        return simpy.Resource.release(self, *args, **kwargs)

class Drill(Machine):
    
    def get_time(self):
#        return random.triangular(1, 6, 3)
        return 5

class Wash(Machine):
    def get_time(self):
#        return random.triangular(1, 6, 3)
        return 5

class Inspection(Machine):
    
    def get_time(self):
        return 5
    
    def inspect(self):
        if random.randint(1,100) > 20:
            return 1
    

def test():
    while True:
        yield env.timeout(1)
        print "running"

def generate_parts():
    '''
    parts -> global list
    '''
    i=0
    while True:
        part = Part(env, "part {0}".format(i))
        parts.append(part)
        i+=1
        yield env.timeout(2)
#        yield env.timeout(random.expovariate(.2))


def parts_log(parts):
    columns = ["Part","Machine","Action","Time"]
    parts_log = pd.DataFrame(columns = columns)
    for part in parts:
        for data in part.data:
            parts_log.loc[len(parts_log)] = data
    return parts_log
    
def queue_log(parts):
    columns = ["Part", "Machine", "Time in Queue"]
    queue_log = pd.DataFrame(columns=columns)
    for part in parts:
        for time_in_queue in part.time_in_queue:
            queue_log.loc[len(queue_log)] = time_in_queue
    return queue_log


def get_avg_time_in_queue(queue_log, machines):
    avg_time_in_queue = []
    for machine in machines:
        times = queue_log[queue_log["Machine"]==machine.name]
        avg_time_in_queue.append((times["Time in Queue"].mean(0), machine.name))
    return avg_time_in_queue

def get_time_in_system(parts):
    time_in_system = pd.DataFrame(columns=["Name", "Time"])
    for part in parts:
#        print part.name
        time_in_system.loc[len(time_in_system)] = [part.name, part.time_in_system]
    return time_in_system


def get_max_time_in_system(time_in_system):
    return time_in_system["Time"].max()


def get_avg_time_in_system(time_in_system):
    return time_in_system[time_in_system["Time"] > 0]["Time"].mean()


def get_time_in_system_stats(time_in_system):
    return time_in_system.Time[time_in_system.Time > 0].describe()


def machine_log(machines):
    columns = ["Time", "Queue Length", "Slots Used", "Action", "Name"]
    machine_log = pd.DataFrame(columns=columns)
    for machine in machines:
        for data in machine.data:
            machine_log.loc[len(machine_log)] = data
    return machine_log



def get_slots_used(machine_log, machines):
    """
    helper function for utilization
    list should lbe appended to the machine_log
    """
    slots_used = []
    for machine in machines:
        print machine.name
        data = machine_log[machine_log["Name"]== machine.name]
        
        for i in range(len(data)):
            if (data.iloc[i]["Action"] == "request"
                or data.iloc[i]["Action"] == "auto-req"):   
                if data.iloc[i]["Slots Used"] < machine.capacity:
                    slots_used.append(data.iloc[i]["Slots Used"] + 1)
                if data.iloc[i]["Slots Used"] == machine.capacity:
                    slots_used.append(data.iloc[i]["Slots Used"])
            else:
                if (data.iloc[i]["Queue Length"] > 0):
                    slots_used.append(data.iloc[i]["Slots Used"])
                else:
                    slots_used.append(data.iloc[i]["Slots Used"] - 1)
    
    return slots_used

#slots = get_slots_used(machine_log, machines)
#machine_log["Slots in use"] = slots
#%%
def get_time_diff(machine_log, machines):
    time_diff = []
    for machine in machines:
        data = machine_log[machine_log["Name"]== machine.name]
        for i in range(len(data)-1):
            diff = machine_log.iloc[i+1]["Time"] - machine_log.iloc[i]["Time"]
            time_diff.append(diff)
        diff = machine._env.now - data.iloc[-1]["Time"]
        time_diff.append(diff)
    return time_diff

#diff = get_time_diff(machine_log, machines)
#machine_log['Time Difference'] = diff
#%%    
def get_utilization(machine_log, machines):
    capacity = []
    util_per_machine = []
    for machine in machines:
        data = machine_log[machine_log["Name"]== machine.name]
        for i in range(len(data)):
            capacity.append(machine.capacity)
    time_diff = get_time_diff(machine_log, machines)
    slots_used = get_slots_used(machine_log, machines)
    utilization = np.array(time_diff)*(np.array(slots_used)/np.array(capacity))
    mac_log = machine_log
    mac_log['util'] = utilization
    for machine in machines:
        data = mac_log[mac_log["Name"]== machine.name]
        
        util_per_machine.append(np.sum(data['util'])/env.now)
    return util_per_machine

#util  = get_utilization(machine_log, machines)

#%%

#class Sim_Stats(object):
#    """
#    Input: simpy env, parts [], machines []
#    Process: Provides tools for the statistical analysis and ploting
#    of the simulation from the varibles tracked in the part
#    and machine objects. Dataframe objects can take a while.
#    """
#    def __init__(self, env, parts, machines):
#        


if __name__ == "__main__":
    sim_start_time = time.time()
    env = simpy.Environment()
    drill = Drill(env, "drill", 1)
    wash = Wash(env, "washing station", 1)
    inspection = Inspection(env, "inspection", 1)
    ## need a list for all the machines in series for use_machines
    machines = [drill, wash, inspection]
    parts = []
        
    env.process(generate_parts())
    env.run(until=20)
#    print('test')
    parts_log = parts_log(parts)
    machine_log = machine_log(machines)
    queue_log = queue_log(parts)
    avg_time_in_queue = get_avg_time_in_queue(queue_log, machines)
    time_in_system = get_time_in_system(parts)
    max_time_in_system = get_max_time_in_system(time_in_system)
    avg_time_in_system = get_avg_time_in_system(time_in_system)
    sim_time_stop = time.time()
    print "elapsed time {0}".format(sim_time_stop - sim_start_time)
#    utilization = get_utilization(machine_log)

#test git Test git
