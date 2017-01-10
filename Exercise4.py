import simpy
import numpy as np
import pandas as pd
import scipy.stats as st
import random
import matplotlib.pyplot as plt

random.seed(42)
RUNTIME = 240
NUMTRIALS = 9
PARTS = 0
TRIAL = 0

def Customer(env, resources, travel, log):
    global PARTS
    name = str(PARTS)
    PARTS += 1

    print('{0}: customer{1} arrives'.format(env.now, name))
    arrival = env.now

    # time entering establishment
    env.timeout(travel[0]())

    # enter ordering sequence
    for i in range(len(resources)):
        with resources[i].request() as request:
            # requests entrance to station
            checkpoint = env.now
            yield request

            # orders food & pays
            log.QueueWait(resources[i].station, env.now, env.now-checkpoint)
            print('{0}: {1} enters {2}'.format(env.now, name, resources[i].station))
            yield env.process(resources[i].use(name, log))
            print('{0}: {1} leaves {2}'.format(env.now, name, resources[i].station))

            # walks to pickup
            yield env.timeout(travel[i]())

    span = env.now - arrival
    log.Span(name, span)


class Arrival(object):
    """
    The Arrival object is used to describe structured customer generation.
    Pass in the interarrival time and number per arrival distributions
    """
    def __init__(self, env, numperarrival, interarrivaltimes, type):
        self.env = env
        self.per = numperarrival
        self.times = interarrivaltimes
        self.type = type

    def spawn(self, resources, travel, log):
        # a static arrival distribution over the whole trial duration
        while True:
            # arrival frequency
            yield self.env.timeout(self.times())

            num = self.per()

            # customers arrive
            for i in range(num):
                self.env.process(Customer(self.env, resources, travel, log))

            print 'Arrival of {0} customer(s) by {1}'.format(num, self.type)

            yield env.timeout(1)


class Bus(Arrival):
    """
    Inherits Arrival, just uses a different spawn function
    """
    def __init__(self, env, numperarrival, interarrivaltimes):
        self.type = 'bus'
        Arrival.__init__(self, env, numperarrival, interarrivaltimes, self.type)

    def spawn(self, resources, travel, log):
        # a variable arrival distribution depending on the time of day
        while True:
            # if between 10AM-11AM or 1PM-2PM
            if self.env.now < 60 or self.env.now > 180:
                pass
                # yield self.env.timeout(self.times())
                # num = self.per()
                # for i in range(num):
                #     self.env.process(Customer(self.env, resources, travel, log))
                # print 'Bus has arrived with {0} customers'.format(num)

            # if between 11AM-1PM
            elif self.env.now > 60 and self.env.now < 180:
                delay = self.times()
                yield self.env.timeout(delay)
                num = self.per()
                for i in range(num):
                    self.env.process(Customer(self.env, resources, travel, log))
                print 'Bus has arrived with {0} customers'.format(num)
                yield self.env.timeout(120-delay)

            yield self.env.timeout(1)


class Counter(simpy.Resource):
    def __init__(self, env, cap, station, times):
        simpy.Resource.__init__(self, env, cap)
        self.env = env
        self.station = station
        self.process_time = times
        self.lastused = 0

    def use(self, partname, log):
        # update the cumulative free time
        log.Free(self.station, env.now - self.lastused)

        # enter queue length into logbook
        log.QueueSize(self.station, self.env.now, len(self.queue))

        # processing time
        for i in range(len(self.process_time)):
            try: yield self.env.timeout(self.process_time[i]())
            except ValueError:
                print('>> set negative delay to zero')
                yield self.env.timeout(0)
            print('{0}: {1} finishes with {2}'.format(env.now, self.station, partname))

        self.lastused = self.env.now

    def breaktime(self):
        with self.request() as request:
            yield request
            print 'One employee going on break from {0}'.format(self.station)
            yield self.env.timeout(10)
            print 'One employee returning from break to {0}'.format(self.station)


class schedule(object):
    # Variable Shift Schedule
    def __init__(self, env, OrderPaySched, PickUpSched):
        self.env = env
        self.OrderPaySched = OrderPaySched
        self.PickUpSched = PickUpSched

    def shifts(self, counters):
        # the generator for assigning off-time
        # 'off-time' refers to time in the model in which an employee isn't working
        # so, at the beginning some workers may be set to off-time until rush hour
        while True:
            # light traffic
            for i in range(self.OrderPaySched[1]-self.OrderPaySched[0]):
                self.env.process(self.off(counters[0]))
            for i in range(self.PickUpSched[1]-self.PickUpSched[0]):
                self.env.process(self.off(counters[1]))
            yield self.env.timeout(180)

            # light traffic again
            for i in range(self.OrderPaySched[1]-self.OrderPaySched[0]):
                self.env.process(self.off(counters[0]))
            for i in range(self.PickUpSched[1]-self.PickUpSched[0]):
                self.env.process(self.off(counters[1]))
            yield self.env.timeout(60)

    def off(self, counter):
        # the manager for scheduling off-time for employees
        with counter.request() as request:
            yield request
            print 'Time off for One employee at {0}'.format(counter.station)
            yield self.env.timeout(60)
        print 'One employee coming to work at {0}'.format(counter.station)

    def breaks(self, counters):
        # the generator for employees taking breaks
        f = 0
        while True:
            if f == 0:
                # first time, wait 50 min. for break
                yield self.env.timeout(50)
                self.env.process(counters[0].breaktime())
                self.env.process(counters[1].breaktime())
                yield self.env.timeout(10)
                f = 1

            # send one person to break from each counter each hour
            # dining room doesn't count as a counter (counters[2})
            yield self.env.timeout(60)
            self.env.process(counters[0].breaktime())
            self.env.process(counters[1].breaktime())


class logbook(object):
    def __init__(self):
        # trial marker
        self.Trial = 0

        # simulation time
        self.SimTime = 0

        # number of centers to track
        self.Centers = 0

        # queue stats
        self.QSize = ['Trial', 'Station', 'Time', 'Size']
        self.QWait = ['Trial', 'Station', 'Time', 'Wait']

        # duration stats
        self.PartSpans = ['Trial', 'Part Name', 'Duration']

        # free time, for each
        self.FreeTime = {}

        # utilization stats
        self.Utilization = ['Trial', 'Center', 'Utilization']

    def LoadCenters(self, centers):
        self.Centers = len(centers)

        # dict of accumulated free time for each center
        for i in range(self.Centers):
            self.FreeTime = dict([ (p.station, 0) for p in centers])

    def QueueSize(self, center, time, entry):
        # empty catch
        if entry == []:
            entry = 0

        self.QSize = np.vstack((self.QSize, [self.Trial, center, time, entry]))

    def QueueWait(self, center, time, entry):
        self.QWait = np.vstack((self.QWait, [self.Trial, center, time, entry]))

    def Span(self, partname, duration):
        self.PartSpans = np.vstack((self.PartSpans, [self.Trial, partname, duration]))

    def Free(self, center, time):
        self.FreeTime[str(center)] += time

    def Record(self):
        pd.DataFrame(self.QWait).to_csv('QueueWait.csv', sep=',', index=False, header=False)
        pd.DataFrame(self.QSize).to_csv('QueueSize.csv', sep=',', index=False, header=False)
        pd.DataFrame(self.PartSpans).to_csv('PartSpan.csv', sep=',', index=False, header=False)
        pd.DataFrame(self.Utilization).to_csv('Utilization.csv', sep=',', index=False, header=False)

    def NewTrial(self):
        # store the utilization data
        for c in self.FreeTime.keys():
            util = (self.SimTime - self.FreeTime[str(c)])/self.SimTime
            self.Utilization = np.vstack((self.Utilization, [self.Trial, c, util]))

        # increment trial marker
        self.Trial += 1

        # reset free time tracker
        self.FreeTime = {}


def graph(*args, **kwargs):

    def duration(data):
        plt.figure(3)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            plt.plot(data[data.Trial == str(i)].Duration.values.astype(float))
            plt.xlim(0, 200)
            plt.ylabel('Duration (min)')
        plt.savefig('Durations.png')

        plt.show()

    def hist(data):
        plt.figure(5)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            plt.hist(data[data.Trial == str(i)].Duration.values.astype(float), 25)
        plt.savefig('histogram.png')

        plt.show()

    def qSize(data):
        plt.figure(6)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            samp = data[data.Trial == str(i)]
            plt.plot(samp[samp.Station == 'OrderPay'].Time.values.astype(float), samp[samp.Station == 'OrderPay'].Size.values.astype(float))
            plt.plot(samp[samp.Station == 'PickUp'].Time.values.astype(float), samp[samp.Station == 'PickUp'].Size.values.astype(float))
            plt.plot(samp[samp.Station == 'DineIn'].Time.values.astype(float), samp[samp.Station == 'DineIn'].Size.values.astype(float))
            plt.ylabel('Size (# customers)')
        plt.savefig('qSize.png')

        plt.show()

    def qWait(data):
        plt.figure(7)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            samp = data[data.Trial == str(i)]
            plt.plot(samp[samp.Station == 'OrderPay'].Time.values.astype(float), samp[samp.Station == 'OrderPay'].Wait.values.astype(float))
            plt.plot(samp[samp.Station == 'PickUp'].Time.values.astype(float), samp[samp.Station == 'PickUp'].Wait.values.astype(float))
            plt.plot(samp[samp.Station == 'DineIn'].Time.values.astype(float), samp[samp.Station == 'DineIn'].Wait.values.astype(float))
            plt.ylabel('Wait (min.)')
        plt.savefig('qWait.png')

        plt.show()

    plt.style.use('seaborn-dark')

    if 'duration' in args:
        duration(data)
    if 'hist' in args:
        hist(data)
    if 'size' in args:
        qSize(data)
    if 'wait' in args:
        qWait(data)


def setup(env, log):

    # the number of employees working each station in a day's shift
    # if less than the max is used at a time, the remaining employees rest (not rush hour)
    OrderPaySched = [6, 6] # [3, 9]
    PickUpSched = [2, 2] # [1, 4]

    DiningCapacity = 30

    OrderPayTimes = [lambda: np.random.triangular(1, 2, 4), lambda: np.random.triangular(1, 2, 3)]
    PickUpTimes = [lambda: random.uniform(0.5, 2)]
    DineInTimes = [lambda: np.random.triangular(10, 20, 30)]

    # set up all resources
    OrderPay = Counter(env, max(OrderPaySched), 'OrderPay', OrderPayTimes)
    PickUp = Counter(env, max(PickUpSched), 'PickUp', PickUpTimes)
    DineIn = Counter(env, DiningCapacity, 'DineIn', DineInTimes)

    # the set of all resources
    counters = [OrderPay, PickUp, DineIn]

    # set the employee scheduling
    sched = schedule(env, OrderPaySched, PickUpSched)
    env.process(sched.shifts(counters))

    # set break scheduling
    env.process(sched.breaks(counters))

    # update logbook with trial info
    log.LoadCenters(counters)
    log.SimTime = RUNTIME

    # travel times between counters
    enter = lambda: random.expovariate(2)     # 0.5 min. mean
    pickup = lambda: random.expovariate(2)    # 0.5 min. mean
    dine = lambda: random.expovariate(2)     # 0.5 min. mean
    exit = lambda: random.expovariate(1)        # 1 min. mean

    # the set of all travel times
    travel = [enter, pickup, dine, exit]

    carload = st.rv_discrete(name='carload', values=([1, 2, 3, 4], [0.2, 0.3, 0.3, 0.2]))

    # initialize arrival generators ( environment, num_per_arrival, inter-arrival_time, generator name)
    walk = Arrival(env, lambda: 1, lambda: random.expovariate(0.3333), 'walking')
    # customers arrive by foot 1 at a time with an exponential distribution with mean 3 min.
    car = Arrival(env, lambda: carload.rvs(), lambda: random.expovariate(0.2), 'car')
    # cars arrive with an exponential distribution with mean 5 min. with custom discrete distribution of customers
    bus = Bus(env, lambda: st.poisson.rvs(30), lambda: random.uniform(0, 120))
    # 1 bus arrives at some point during busy hours with 30 customers

    # customer spawn!
    env.process(walk.spawn(counters, travel, log))
    env.process(car.spawn(counters, travel, log))
    env.process(bus.spawn(counters, travel, log))


if __name__ == "__main__":

    # initialize logbook
    log = logbook()

    # run trials, aka repeats
    for i in range(NUMTRIALS):
        # global TRIAL
        TRIAL += 1

        print('\n * * * * * * Trial {0} * * * * * * \n'.format(i))
        log.NewTrial()

        # defining the environment
        env = simpy.Environment()

        # trigger startup process
        setup(env, log)

        # execute simulation!
        env.run(until=RUNTIME)
        del env

        global PARTS
        PARTS = 0

    # export to .csv
    log.Record()

    # prep data of interest
    data = log.PartSpans
    data = pd.DataFrame(data)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:, :]

    # inspect customer trip duration distribution
    graph('hist', data=data)
    # inspect customer trip durations over time
    graph('duration', data=data)
    del data

    data = log.QSize
    data = pd.DataFrame(data)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:, :]
    graph('size', data=data)
    del data

    data = log.QWait
    data = pd.DataFrame(data)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:, :]
    graph('wait', data=data)
    del data

    # peace