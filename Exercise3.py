import simpy
import numpy as np
import pandas as pd
import scipy.stats as st
import random
import matplotlib.pyplot as plt

random.seed(42)
RUNTIME = 200000
NUMTRIALS = 5

def Part(env, name, resources, trial, log):
    print('{0}: {1} is born'.format(env.now, name))
    arrival = env.now

    for i in range(len(resources)):
        with resources[i].machine.request() as request:
            # requests entrance to station
            start = env.now
            yield request

            # gets processed
            log.QueueWait(resources[i].name, env.now, env.now-start)
            print('{0}: {1} enters {2}'.format(env.now, name, resources[i].name))
            yield env.process(resources[i].use(name, trial, log))

            # leaves station
            resources[i].machine.release(request)

            # record the time
            print('{0}: {1} leaves {2}'.format(env.now, name, resources[i].name))

    span = env.now - arrival
    log.Span(name, span)

class logbook(object):
    def __init__(self):
        # trial marker
        self.Trial = 0

        # simulation time
        self.SimTime = 0

        # number of centers to track
        self.Centers = 0

        # queue stats
        self.QSize = ['Trial', 'Time', 'Size']
        self.QWait = ['Trial', 'Time', 'Wait']

        # duration stats
        self.PartSpans = ['Trial', 'Part Name', 'Duration']

        # free time, for each
        self.FreeTime = {}

        # utilization stats
        self.Utilization = ['Trial', 'Center', 'Utilization']

    def LoadCenters(self, centers):
        self.Centers = len(centers)

        # dict of accumulated free time for each center
        for i in range(len(centers)):
            self.FreeTime = dict([ (p.name, 0) for p in centers])

    def QueueSize(self, center, time, entry):
        # empty catch
        if entry == []:
            entry = 0

        self.QSize = np.vstack((self.QSize, [self.Trial, time, entry]))

    def QueueWait(self, center, time, entry):
        self.QWait = np.vstack((self.QWait, [self.Trial, time, entry]))

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

    def batch(self, data, sets):
        # number of trials
        t = len(data.Trial.unique())
        # number of data in each trial
        n = []
        # number of data points in each batch for each trial
        per = []
        # a panel of all batches from all trials
        batches = [object for i in range(t)]

        # over all trials
        for i in range(t):
            n.append(len(data[data.Trial == str(i+1)]))
            per.append(int(n[i]/sets))

        for i in range(t):
            k = 0
            batch = np.zeros((per[i], sets))
            print(batch.shape)
            # over number of batch sets
            for j in range(sets):
                batch[:, j] = data[data.Trial == str(i+1)].Duration.iloc[k:k+per[i]]
                k += per[i]

            batches[i] = batch
            del batch

        return batches

    def stats(self, batches):
        t = len(batches)
        means = [object for i in range(t)]
        stds = [object for i in range(t)]
        ns = [object for i in range(t)]
        sems = [object for i in range(t)]

        Gmeans = [object for i in range(t)]
        Gstds = [object for i in range(t)]
        Gns = [object for i in range(t)]
        Gsems = [object for i in range(t)]

        for i in range(t):
            num = len(batches[i][0, :])
            mean, std, n, sem = [], [], [], []
            for j in range(num):
                mean = np.append(mean, batches[i][:, j].mean())
                std = np.append(std, batches[i][:, j].std())
                n = np.append(n, len(batches[i][:, j]))
                sem = np.append(sem, st.sem(batches[i][:, j]))

            # store grand mean, etc from all batches
            Gmeans[i] = np.mean(mean)
            Gstds[i] = np.mean(std)
            Gns[i] = np.mean(n)
            Gsems[i] = np.mean(sem)
            # store stats for all batches
            means[i] = mean
            stds[i] = std
            ns[i] = n
            sems[i] = sem
            del mean, std, n, sem

        CI = []
        for i in range(NUMTRIALS):
            CI.append(st.t.interval(0.95, int(Gns[i])-1, loc=Gmeans[i], scale=Gsems[i]))

        ts = np.zeros((NUMTRIALS, NUMTRIALS))
        ps = np.zeros((NUMTRIALS, NUMTRIALS))
        for i in range(NUMTRIALS):
            for j in range(i+1, NUMTRIALS):
                a = means[i]
                b = means[j]
                if len(a)>len(b):
                    a = a[:len(b)]
                elif len(b)>len(a):
                    b = b[:len(a)]
                ts[i, j], ps[i, j] = st.ttest_rel(a, b)
                del a, b

        stats = [Gmeans, Gstds, Gns, Gsems]

        return stats, CI, ts, ps


class Center(object):
    def __init__(self, name, env, cap):
        self.name = name
        self.env = env
        self.machine = simpy.Resource(env, capacity=cap)
        self.lastused = 0

    def use(self, partname, trial, log):
        # update the cumulative free time
        log.Free(self.name, env.now - self.lastused)

        # enter queue length into logbook
        log.QueueSize(self.name, self.env.now, len(self.machine.queue))

        # processing time
        try: yield self.env.timeout(trial.process_time())
        except ValueError:
            print('>> set negative delay to zero')
            self.env.timeout(0)
        print('{0}: {1} finishes with {2}'.format(env.now, self.name, partname))

        self.lastused = env.now


class Trial(object):
    def __init__(self):
        self.arrival_time = object
        self.process_time = object
    def set_arrival_time(self, timefunc):
        self.arrival_time = timefunc
    def set_process_time(self, timefunc):
        self.process_time = timefunc


def graph(*args, **kwargs):

    def basic(stats):

        stats = pd.DataFrame(stats)

        plt.figure(1)

        plt.subplot(311)
        plt.title('Processing Time Statistics')
        plt.xticks(x)
        plt.plot(x, stats.ix[0, :])
        plt.ylabel('Mean')
        plt.xlim(0.9, 5.1)
        plt.grid(True)

        plt.subplot(312)
        plt.xticks(x)
        plt.plot(x, stats.ix[1, :])
        plt.ylabel('STD')
        plt.xlim(0.9, 5.1)
        plt.grid(True)

        plt.subplot(313)
        plt.xticks(x)
        plt.plot(x, stats.ix[2, :])
        plt.ylabel('N')
        plt.xlim(0.9, 5.1)
        plt.grid(True)

        plt.savefig('Stats.png')
        plt.show()

    def meanci(stats, CI):

        stats = pd.DataFrame(stats)

        err = []
        # this finds the width of the 95% confidence interval
        # for use with plt.errorbar
        for i in range(NUMTRIALS):
            # check to see if NaN
            if CI[i][0] != CI[i][0]:
                err.append(0)
            else: err.append((CI[i][1]-CI[i][0])/2)

        plt.figure(2)
        plt.title('Processing Time Statistics')

        plt.xticks(x)
        plt.plot(x, stats.ix[0, :])
        plt.errorbar(x, stats.ix[0, :], yerr=err)
        plt.ylabel('Mean')
        plt.xlim(0.9, 5.1)
        plt.grid(True)
        plt.savefig('Mean.png')

        plt.show()

    def duration(data):
        plt.figure(3)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            plt.plot(data[data.Trial == str(i)].Duration.values.astype(float))
            plt.xlim(0, 20000)
            plt.ylabel('Duration (min)')
        plt.savefig('Durations.png')

        plt.show()

    def autoc(data):
        autocorr = [object for i in range(NUMTRIALS+1)]
        norms = [object for i in range(NUMTRIALS+1)]
        for i in range(1, NUMTRIALS+1):
            si = data[data.Trial == str(i)].Duration.values.astype(float)
            autocorr[i-1] = np.correlate(si, si, mode='full')
            autocorr[i-1] = autocorr[i-1][autocorr[i-1].size/2:]
            norms[i-1] = autocorr[i-1]/autocorr[i-1].max()

        plt.figure(4)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            plt.plot(autocorr[i-1])
            plt.xlim(0, 20000)
            plt.ylabel('Autocorrelation')
        plt.savefig('Autocorrelations.png')

        plt.show()

    def hist(data):
        plt.figure(5)
        for i in range(1, NUMTRIALS+1):
            plt.subplot(str(str(NUMTRIALS)+'1'+str(i)))
            plt.hist(data[data.Trial == str(i)].Duration.values.astype(float), 25)
        plt.savefig('histogram.png')

        plt.show()

    x = np.arange(1, NUMTRIALS+1)

    plt.style.use('seaborn-dark')

    if 'basic' in args:
        basic(stats)
    if 'meanci' in args:
        meanci(stats, CI)
    if 'duration' in args:
        duration(data)
    if 'autoc' in args:
        autoc(data)
    if 'hist' in args:
        hist(data)


def setup(env, trial, runtime, log):
    # create all the resources!
    Server = Center('Server', env, 1)

    # the set of all resources
    centers = [Server]

    # update logbook with rial info
    log.LoadCenters(centers)
    log.SimTime = runtime

    # the first part!
    env.process(Part(env, 'Part0', centers, trial, log))

    # part spawn!
    num = 0
    while True:
        num += 1
        name = 'Part'+str(num)

        yield env.timeout(trial.arrival_time())
        env.process(Part(env, name, centers, trial, log))

if __name__ == "__main__":

    # initialize logbook
    log = logbook()

    # define trials
    A, B, C, D, E = Trial(), Trial(), Trial(), Trial(), Trial()
    trials = [A, B, C, D, E]

    # set part arrival time functions for each trial
    A.set_arrival_time(lambda: 10)
    B.set_arrival_time(lambda: random.expovariate(0.1))
    C.set_arrival_time(lambda: random.expovariate(0.1))
    D.set_arrival_time(lambda: random.expovariate(0.1))
    E.set_arrival_time(lambda: random.expovariate(0.1))

    # set processing time functions for each trial
    A.set_process_time(lambda: 9)
    B.set_process_time(lambda: 9)
    C.set_process_time(lambda: np.random.uniform(7, 11))
    D.set_process_time(lambda: random.expovariate(0.1111))
    E.set_process_time(lambda: np.random.normal(9, 9))

    # run Trials A-E
    for i in range(NUMTRIALS):

        print('\n * * * * * * Trial {0} * * * * * * \n'.format(i))
        log.NewTrial()

        # defining the environment
        env = simpy.Environment()

        # trigger startup process
        env.process(setup(env, trials[i], RUNTIME, log))

        # execute simulation!
        env.run(until=RUNTIME)
        del env

    # export to .csv
    log.Record()

    # prep data of interest
    data = log.PartSpans
    data = pd.DataFrame(data)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:, :]

    # inspect part duration distribution
    graph('hist', data=data)
    # inspect part durations over time
    graph('duration', data=data)
    # inspect autocorrelation to inform batching size
    graph('autoc', data=data)

    # batch the data
    batches = log.batch(data, 12)

    stats, CI, ts, ps = log.stats(batches)
    pd.DataFrame(ts).to_csv('ts.csv')
    pd.DataFrame(ps).to_csv('ps.csv')

    # show off your analysis results
    print('\n')
    print(stats)
    print('\n')
    print(CI)
    print('\n')
    print(ts)
    print('\n')
    print(ps)

    # visualize basic statistics
    graph('basic', stats=stats)
    # visualize the confidence intervals
    graph('meanci', stats=stats, CI=CI)

