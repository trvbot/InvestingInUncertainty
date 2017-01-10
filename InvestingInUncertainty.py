import simpy
import numpy as np
import random
import quandl
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# iterations
NUMTRIALS = 1000

# 251 days in trading year
YEAR = 251
RUNTIME = 25*YEAR

# to grab data from quandl or just import from csv
GRAB = False

# determines if you print to console, record all statistics
# if false, all printing suppressed and only initial and final net worth recorded for each trial.
LONGFORM = False

BUYRATE = 0.03
SELLRATE = 0.03

initial_investment = 500


class user(object):
    def __init__(self, env, initial):
        self.env = env
        # environment, max capacity, initial amount
        self.wallet = simpy.Container(env, 1000000, initial)
        self.wallethistory = []

        # accumulates any overdrafts if wallet empty
        self.debt = simpy.Container(env, 1000000, 0)
        self.debthistory = []

        #   minimum wage ($15k/year)
        self.wage = 1250
        #   ~ min. wage for pos. ROI
        # self.wage = 1300
        #   ~ median per cap income in 1980 ($20k/year)
        # self.wage = 1666.66

        # counts the number of bad events experienced in simulation period
        self.bad_things = 0

        # interested or not in investing
        # logic in market.run()
        self.interest = True

        # interest in selling investments
        # logic in market.run()
        self.sell = False

        # cost-of-living.careertrends.com
        food = 271
        healthcare = 273
        housing = 560
        necessities = 401
        taxes = 370
        transportation = 493
        # totals $2368 a month
        self.fixed_costs = food+healthcare+housing+necessities+taxes+transportation

        # paycheck comes every x days
        # 20 is one month
        PayPeriod = 20

        # just as  a precaution, print every time
        YearIncome = self.wage*YEAR/PayPeriod
        print('Yearly Income: {0}\n'.format(YearIncome))

        # schedule processes!
        env.process(self.life())
        env.process(self.bills(self.fixed_costs))

        env.process(self.income(PayPeriod))


    def life(self):
        while True:
            # probability of bad event
            # 8% chance of 1 bad thing in a 5 year period
            badLuck = 25

            # alea iacta est
            chance = random.randint(0, 100)

            if chance <= badLuck:
                # pick a day for it to occur within next five years
                trouble = np.random.uniform(0, 5*YEAR)
                yield self.env.timeout(trouble)

                # increase counter of bad events
                self.bad_things += 1

                # determine financial impact
                cost = random.triangular(5000, 10000, 7500)
                if LONGFORM: print('\n>> something terrible has happened. ${0} lost.').format(cost)

                if cost > self.wallet.level:
                    # pay the full amount in debt
                    self.debt.put(cost)

                    if LONGFORM: print('>> only had {0}, {1} in debt\n').format(self.wallet.level, cost)
                else:
                    # withdraw entire cost
                    self.wallet.get(cost)
                    if LONGFORM: print('>> Paid off in full. phew\n')

                # wait the remainder of time in 5-year period
                yield self.env.timeout(5*YEAR-trouble)

            else: yield self.env.timeout(5*YEAR)

    def bills(self, fixed_costs):
        while True:
            # pay bills once a month
            # (every 20 days for adjusted fiscal year)
            yield self.env.timeout(20)

            # pay off some debt before bills
            # pay off debt before bills
            if self.debt.level != 0 and self.debt.level*0.1 < self.wallet.level:
                # pay off 10% of debt each month
                payback_rate = 0.1
                payback = payback_rate*self.debt.level
                # remove money from wallet
                self.wallet.get(payback)
                # pay off some debt
                self.debt.get(payback)

            # determine misc expenses
            extra = np.random.normal(200, 100)
            # catch for negative costs
            if extra < 0:
                extra = 0

            # calculate total expenses
            expenses = float(fixed_costs+extra)

            # determine if debt needs to be taken out to pay bills
            if expenses > self.wallet.level:
                # pay off what you can, leaving some money
                balance = expenses - 0.5*self.wallet.level
                expenses = expenses - balance
                # only pay bills if you can in full, else all in debt
                self.debt.put(expenses)
                self.wallet.get(balance)
                # expenses = self.wallet.level
                if LONGFORM: print('>> Over-drawn on bills. ${0} taken in debt, ${1} paid from wallet\n').format(expenses, balance)
            else:
                # pay bills in full
                self.wallet.get(expenses)
                if LONGFORM: print('>> bills paid: ${0}\n').format(expenses)

    def income(self, period):
        while True:
            # get paid once per pay period
            yield self.env.timeout(period)

            # if wage was a dynamic parameter
            if self.wage > 0.0:
                # deposit paycheck into wallet
                self.wallet.put(self.wage)
                if LONGFORM: print('>> payday: ${0}\n').format(self.wage)


class Market(object):
    def __init__(self):

        if GRAB == True:
            print 'acquiring market data'
            # grab market data from quandl
            quandl.ApiConfig.api_key = 'zFCX5bmbwZvgGzHu5szi'
            snp_index = quandl.get("YAHOO/FUND_VFINX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
            mining_eft = quandl.get("YAHOO/FUND_VGPMX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
            total_bond = quandl.get("YAHOO/FUND_VBMFX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
        else:
            snp_index = pd.read_csv('snp_index.csv')
            mining_eft = pd.read_csv('mining_eft.csv')
            total_bond = pd.read_csv('total_bond.csv')

        self.snp_index = np.asarray(snp_index.Close)
        self.mining_eft = np.asarray(mining_eft.Close)
        self.total_bond = np.asarray(total_bond.Close)

        # trims the sparse data portion at beginning of mining_eft
        self.mining_eft = self.mining_eft[150:]
        self.snp_index = self.snp_index[150:]
        self.total_bond = self.total_bond[150:]

        # modeled each fund as a normal distribution
        self.loc1, self.scale1 = st.norm.fit(self.snp_index)
        self.loc2, self.scale2 = st.norm.fit(self.mining_eft)
        self.loc3, self.scale3 = st.norm.fit(self.total_bond)

        if LONGFORM: print 'market data acquired and modeled'

    def history(self, time, account):
        if account.name == 'mining':
            return self.mining_eft[time]
        if account.name == 'index':
            return self.snp_index[time]
        if account.name == 'bond':
            return self.total_bond[time]

    def update(self, env, account):
        percentchange = 0
        # if trial is using historical data
        percentchange = self.history(int(env.now+1)+log.market_start, account)


        # calculate change
        change = percentchange*account.level

        # update balance
        if change > 0.0:
            account.put(account, change)
        elif change < 0.0:
            change = -1*change
            account.get(account, change)

        # log balance change
        account.value.append([env.now+1, percentchange, change, account.level])

    def run(self, env, user, accounts, investor):
        while True:
            if LONGFORM: print('\n- - day: {0} - -').format(env.now+1)

            # selling interest logic
            if user.debt.level > 2500 and investor.total_value() > 0.2*user.debt.level:
                if LONGFORM: print '\n>> Gonna sell shares to pay off debt\n'
                user.sell = True

            # chance to lose investing interest
            if user.interest == True:
                if user.wallet.level < 1.5*user.fixed_costs and np.random.random()<0.5:
                    # lose investing interest
                    if LONGFORM: print '\n>> Imma stop investing for awhile..\n'
                    user.interest = False
            # chance to gain investing interest again
            elif user.interest == False:
                if user.wallet.level > 1.5*user.fixed_costs and np.random.random()<0.05:
                    if LONGFORM: print '\n>> Gonna Start Investing Again!\n'
                    # regain investing interest
                    user.interest = True

            for i in range(len(accounts)):
                # update the worth of each account
                self.update(env, accounts[i])
                if LONGFORM: print('> {0} closes at balance {1}').format(accounts[i].name, accounts[i].level)

            # store wallet history
            user.wallethistory.append([env.now+1, user.wallet.level])
            user.debthistory.append([env.now+1, user.debt.level])
            if LONGFORM: print('>> wallet level at: {0}').format(user.wallet.level)
            if LONGFORM: print('>> debt level at: {0}\n').format(user.debt.level)

            # wait until next day
            yield env.timeout(1)


class Investor(object):
    def __init__(self, env, client, accounts):
        self.env = env
        # formatted as [day, account name, amount bought, total account]
        self.buyhistory = []
        # formatted as [day, account name, amount sold, total account]
        self.sellhistory = []

        # built-in function for calculating the net worth at any time
        self.total_value = lambda: accounts[0].level+accounts[1].level+accounts[2].level

        # begin the processes for buying & selling assets
        env.process(self.invest(client, accounts))
        env.process(self.sell(client, accounts))

    def invest(self, user, accounts):
        while True:
            # every 15 days invest money
            # the '+1' is so that you buy on the same day as your paycheck
            # which makes the data less jagged and weird
            if np.mod(self.env.now+1, 15) == 0.0 and user.interest:
                # amount set aside to invest every pay period
                amount = 0.1*user.wage
                user.wallet.get(amount)

                # allocation strategy
                strategy = [1./3, 1./3, 1./3]

                # place investments
                for i in range(len(accounts)):
                    if LONGFORM: print('> {0} invested in {1}').format(int(strategy[i]*amount), accounts[i].name)
                    accounts[i].put(accounts[i], int(strategy[i]*amount), 'fee')
                    self.buyhistory.append([self.env.now+1, accounts[i].name, int(strategy[i]*amount), accounts[i].level])

                user.has_stocks = True

            yield env.timeout(1)

    def sell(self, user, accounts):
        while True:
            # only if there is investing interest
            if user.sell:
                # total amount needed
                debt = user.debt.level

                valuation = self.total_value()
                if LONGFORM: print('total valuation is {0}\n').format(valuation)

                strategy = [1./3, 1./3, 1./3]
                amounts = [0, 0, 0]

                sales = 0
                for i in range(len(accounts)):
                    amounts[i] = 0.5*accounts[i].level
                    sales += amounts[i]

                if LONGFORM: print('total sold is {0}\n').format(sales)

                for i in range(len(accounts)):
                    if LONGFORM: print('> ${0} sold of {1}\n').format(int(amounts[i]), accounts[i].name)
                    accounts[i].get(accounts[i], int(amounts[i]), 'fee')
                    self.sellhistory.append([self.env.now+1, accounts[i].name, int(amounts[i]), accounts[i].level])
                if LONGFORM: print('\n')

                # pay off debt
                user.debt.get(sales)

                user.sell = False

            yield env.timeout(1)


class Account(simpy.Container):
    def __init__(self, env, cap, init, name, buyin, buyfeerate, sellfeerate):
        simpy.Container.__init__(self, env, cap, init)
        self.env = env
        self.name = name
        self.buyin = buyin
        self.buyfeerate = buyfeerate
        self.sellfeerate = sellfeerate

        # formatted as [day, % change, $ change, total account value]
        self.value = []
        # formatted as [day, fee paid, buying/selling]
        self.fees = []

        self.put(self, self.buyin)

    def put(self, *args, **kwargs):
        # some amount subtracted off the top for buying fees
        if 'fee' in args:
            amount = args[1]
            fee = self.buyfeerate*amount
            amount -= fee
            newargs = args[0], amount
            self.fees.append([self.env.now+1, fee, 'buy'])
        else:
            newargs = args
        return simpy.Container.put(*newargs, **kwargs)

    def get(self, *args, **kwargs):
        # some amount subtracted off the top for selling fees
        if 'fee' in args:
            amount = args[1]
            fee = self.sellfeerate*amount
            amount -= fee
            newargs = args[0], amount
            self.fees.append([self.env.now+1, fee, 'sell'])
        else:
            newargs = args
        return simpy.Container.get(*newargs, **kwargs)


class logbook(object):
    def __init__(self):
        # the current trial
        self.trial = 0

        # the market window for the current trial
        self.market_start = 0
        self.market_stop = 0

        # formatted as [trial]
        #   investments bought in each account: [day, bought $]
        self.mining_bought = []
        self.index_bought = []
        self.bond_bought = []

        # formatted as [trial]
        #   investments sold in each account: [day, sold $]
        self.mining_sold = []
        self.index_sold = []
        self.bond_sold = []

        # formatted as [trial]
        #   total amount in each account: [day, amount $]
        self.mining_amount = []
        self.index_amount = []
        self.bond_amount = []

        # formatted as [trial]
        #   amount of fees for both bought and sold transactions: [day, fee $]
        self.mining_fees = []
        self.index_fees = []
        self.bond_fees = []

        # formatted as [trial]
        #   store the market behavior for a each trial: [day, asset value]
        self.mining_market = []
        self.index_market = []
        self.bond_market = []

        # formatted as [trial]
        #   wallet balance over trial duration: [day, wallet $]
        self.wallet_history = []
        self.net_worth = []

        # formatted as [trial]
        #   return on Investment for accounts: [day, ROI]
        self.mining_ROI = []
        self.index_ROI = []
        self.bond_ROI = []

        # formatted as [trial]
        #   'trend' for accounts (accumulators): [day, value]
        self.mining_trend = []
        self.index_trend = []
        self.bond_trend = []

        self.debt_history = []

        ## short-form stats (JUST for ROI)
        self.initialWorth = []
        self.finalWorth = []
        self.final_debt = []
        self.total_ROI = []
        self.total_bought = []
        self.total_value = []

        self.num_bad = []

    def record_long(self, investor, accounts, user):
        # records all data from trial and stores in dataframes of more simple structure
        # ie. each metric has a corresponding dataframe object in its list for each trial

        self.wallet_history.append(pd.DataFrame(user.wallethistory, columns=['Day', 'Amount']).set_index('Day'))
        self.debt_history.append(pd.DataFrame(user.debthistory, columns=['Day', 'Amount']).set_index('Day'))

        # market performance
        self.mining_market.append(pd.Series(market.mining_eft[self.market_start:self.market_stop]))
        self.index_market.append(pd.Series(market.snp_index[self.market_start:self.market_stop]))
        self.bond_market.append(pd.Series(market.total_bond[self.market_start:self.market_stop]))

        # total account value
        col = ['Day', 'PercentChange', 'AmountChange', 'Value']
        self.mining_amount.append(pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value)
        self.index_amount.append(pd.DataFrame(accounts[1].value, columns=col).set_index('Day').Value)
        self.bond_amount.append(pd.DataFrame(accounts[2].value, columns=col).set_index('Day').Value)

        # account investments fees
        col = ['Day', 'FeeAmount', 'FeeType']
        self.mining_fees.append(pd.DataFrame(accounts[0].fees, columns=col).set_index('Day').FeeAmount)
        self.index_fees.append(pd.DataFrame(accounts[1].fees, columns=col).set_index('Day').FeeAmount)
        self.bond_fees.append(pd.DataFrame(accounts[2].fees, columns=col).set_index('Day').FeeAmount)

        # convert stored data to dataframe for access
        col = ['Day', 'Account', 'AmountBought', 'AccountTotal']
        bh = pd.DataFrame(investor.buyhistory, columns=col).set_index('Day')

        col = ['Day', 'Account', 'AmountSold', 'AccountTotal']
        sh = pd.DataFrame(investor.sellhistory, columns=col).set_index('Day')

        # account investments bought
        self.mining_bought.append(bh[bh.Account == 'mining'].AmountBought)
        self.index_bought.append(bh[bh.Account == 'index'].AmountBought)
        self.bond_bought.append(bh[bh.Account == 'bond'].AmountBought)

        # account investments sold
        self.mining_sold.append(sh[sh.Account == 'mining'].AmountSold)
        self.index_sold.append(sh[sh.Account == 'index'].AmountSold)
        self.bond_sold.append(sh[sh.Account == 'bond'].AmountSold)

        # calculate Trend
        self.mining_trend.append(self.AccountTrend(self.mining_market[self.trial]))
        self.index_trend.append(self.AccountTrend(self.index_market[self.trial]))
        self.bond_trend.append(self.AccountTrend(self.bond_market[self.trial]))

        # calculate Net Worth
        self.net_worth.append(self.NetWorth(self.wallet_history[self.trial], self.mining_amount[self.trial],
                                            self.index_amount[self.trial], self.bond_amount[self.trial]))

        self.total_ROI.append(self.ROI(investor, accounts, user))

        self.num_bad.append(user.bad_things)

        col = ['Day', 'PercentChange', 'AmountChange', 'Value']
        mining = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        index = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        bond = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value

        initial = mining[1] + index[1] + bond[1] + user.wallethistory[0][1] - user.debthistory[-1][1]
        final = mining[RUNTIME - 1] + index[RUNTIME - 1] + bond[RUNTIME - 1] + user.wallethistory[-1][1] - \
                user.debthistory[-1][1]

        self.finalWorth.append(final)
        self.initialWorth.append(initial)

        self.trial += 1
        if LONGFORM: print('Logbook Recorded for Trial {0}').format(self.trial)

    def record_short(self, investor, accounts, user):

        col = ['Day', 'PercentChange', 'AmountChange', 'Value']
        mining = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        index = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        bond = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value

        initial = mining[1] + index[1] + bond[1] + user.wallethistory[0][1]
        final = mining[RUNTIME-1] + index[RUNTIME-1] + bond[RUNTIME-1] + user.wallethistory[-1][1] - user.debthistory[-1][1]

        self.finalWorth.append(final)
        self.initialWorth.append(initial)

        total_bought = initial_investment*len(accounts)
        for i in range(len(investor.buyhistory)):
            if investor.buyhistory[i][2] < 0.0:
                print('huh, negative buy history found: {0}\n').format(investor.buyhistory[i][2])
            total_bought += investor.buyhistory[i][2]

        self.final_debt.append(user.debthistory[-1][1])
        total_value = mining[RUNTIME-1]+index[RUNTIME-1]+bond[RUNTIME-1]

        ROI = 100*(total_value/total_bought)

        self.total_ROI.append(ROI)
        self.total_bought.append(total_bought)
        self.total_value.append(total_value)

        self.num_bad.append(user.bad_things)

    def store(self, env, investor, accounts, user):
        while True:
            # stores all data at the last second
            yield env.timeout(RUNTIME-1)
            if LONGFORM: self.record_long(investor, accounts, user)
            else: self.record_short(investor, accounts, user)

    def AccountTrend(self, account):
        # calculates the cumulative change in market value at each point in present trial
        trend = np.empty((len(account), 2))
        trend[:, 0] = account.index
        for i in range(len(account)):
            trend[i, 1] = np.sum(account[:i])
        return pd.DataFrame(trend, columns=['Day', 'Value']).set_index('Day')

    def ROI(self, investor, accounts, user):

        col = ['Day', 'PercentChange', 'AmountChange', 'Value']
        mining = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        index = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value
        bond = pd.DataFrame(accounts[0].value, columns=col).set_index('Day').Value

        total_bought = 0.0
        for i in range(len(investor.buyhistory)):
            total_bought += investor.buyhistory[i][2]
        if total_bought < 1.0:
            total_bought = 0.1

        # calculates percent return on investment
        total_value = mining[RUNTIME-1]+index[RUNTIME-1]+bond[RUNTIME-1]
        ROI = 100*(total_value/total_bought)
        return ROI

    def NetWorth(self, wallet, mining, index, bond):
        # calculates the net worth at each time point in present trial
        nw = np.empty((len(wallet), 2))
        nw[:, 0] = wallet.index
        for i in range(len(wallet)):
            nw[i, 1] = wallet.iloc[i]+mining.iloc[i]+index.iloc[i]+bond.iloc[i]
        return pd.DataFrame(nw, columns=['Day', 'Amount']).set_index('Day')


def graph(trial):
    # plots some relevant plots for one trial

    # plot wallet & net value over time
    f1, (plot1, plot2, plot3) = plt.subplots(3, sharex=True)
    plot1.set_title('Worth over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Wallet (dollars)')
    plot1.plot(log.wallet_history[trial])
    plot2.plot(log.net_worth[trial].subtract(log.wallet_history[trial]))
    plot3.plot(log.net_worth[trial])
    f1.savefig('worth.png')
    plt.show()

    # plot comparison of value of all three accounts over time
    f2, (plot1, plot2, plot3) = plt.subplots(3, sharex=True)
    plt.xlabel('Time (days)')
    plt.ylabel('Value (dollars)')
    # plot the value of the investments in the mining account
    plot1.plot(log.mining_amount[trial])
    # plot the value of the investments in the index account
    plot2.plot(log.index_amount[trial])
    # plot the value of the investments in the bond account
    plot3.plot(log.bond_amount[trial])
    plt.title('Investment Account Value over Time')
    f2.savefig('accountvalues.png')
    plt.show()

    # plot comparison of buying and selling of all three accounts over time
    f3, (plot1, plot2) = plt.subplots(2, sharex=True)
    plot1.set_title('Investments Sold')
    plot2.set_title('Investments Bought')
    plt.xlabel('Time (days)')
    plt.ylabel('Investments (dollars)')
    try:
        plot1.plot(log.mining_sold[trial], c='red')
        plot1.plot(log.index_sold[trial], c='green')
        plot1.plot(log.bond_sold[trial], c='blue')
    except ZeroDivisionError:   # i.e. no sales logged
        pass
    try:
        plot2.plot(log.mining_bought[trial], c='red')
        plot2.plot(log.index_bought[trial], c='green')
        plot2.plot(log.bond_bought[trial], c='blue')
    except ZeroDivisionError:   # i.e. no purchases logged
        pass
    f3.savefig('buysell.png')
    plt.show()

def extra_stats(log):
    # shows the relative accumulated market value of each account
    plt.figure(6)
    plt.plot(log.mining_trend[0], 'red')
    plt.plot(log.index_trend[0], 'blue')
    plt.plot(log.bond_trend[0], 'green')
    plt.show()

    # dataframe of all buying and selling activity (only first trial)
    BoughtSold = {'B_Mining':log.mining_bought[0], 'S_Mining':log.mining_sold[0],
          'B_Index':log.index_bought[0], 'S_Index':log.index_sold[0],
          'B_Bond':log.bond_bought[0], 'S_Bond':log.bond_sold[0]}
    BoughtSold = pd.DataFrame(BoughtSold, index=np.arange(1, RUNTIME)).fillna(0)

    # dataframe of all activity in the mining account (only first trial)
    MiningSummary = {'Bought':log.mining_bought[0], 'Sold':log.mining_sold[0], 'Fees':log.mining_fees[0],
                     'Amount':log.mining_amount[0]}
    MiningSummary = pd.DataFrame(MiningSummary, index=np.arange(1, RUNTIME)).fillna(0)

    # dataframe of all account values (only first trial)
    AccountSummary = {'Mining':log.mining_amount[0], 'Index':log.index_amount[0], 'Bond':log.bond_amount[0]}
    AccountSummary = pd.DataFrame(AccountSummary, index=np.arange(1, RUNTIME)).fillna(0)
    AccountSummary.tail()

def hist(data):
    plt.hist(data, 10)
    return plt.show()

def box(group, values):
    unis = np.unique(group)
    box = []
    for i in range(len(unis)):
        tmp = []
        for j in range(len(values)):
            if unis[i] == group[j]: tmp.append(values[j])
        box.append(tmp)

    plt.axes().set_xticklabels(unis)
    return plt.boxplot(box)

def setup(env, trial):
    if LONGFORM: print 'setting up...\n'

    # Jack Attack
    Jack = user(env, 1000)

    # investment accounts to open
    # env, max cap, initial amount, name, buy in, buy fee rate, sell fee rate
    mining = Account(env, 100000, 0, 'mining', initial_investment, BUYRATE, SELLRATE)
    index = Account(env, 100000, 0, 'index', initial_investment, BUYRATE, SELLRATE)
    bond = Account(env, 100000, 0, 'bond', initial_investment, BUYRATE, SELLRATE)

    accounts = [mining, index, bond]

    for i in range(len(accounts)):
        if LONGFORM: print('{0} opening balance: {1}').format(accounts[i].name, accounts[i].level)

    # Rich Chambers (actual name of an accountant I knew)
    Rich = Investor(env, Jack, accounts)

    # initialize and prepare logbook to store data
    env.process(log.store(env, Rich, accounts, Jack))

    # market process
    env.process(market.run(env, Jack, accounts, Rich))


# initialize market to be used for all trials
market = Market()

# initialize logbook
log = logbook()

# run all trials
for i in range(1, NUMTRIALS+1):
    print('\n * * * * * * Trial {0} * * * * * * \n'.format(i))

    random.seed(lambda: random.randint(1, 99)+i)

    # sample from historical data
    log.market_start = np.random.randint(0, len(market.total_bond)-(RUNTIME+1))
    log.market_stop = log.market_start+RUNTIME+1

    # defining the environment
    env = simpy.Environment()

    # trigger startup process
    setup(env, i)

    # execute simulation!
    env.run(until=RUNTIME)
    del env


if LONGFORM == True:
    plt.figure(5)
    plt.title('Financial Worth Over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Dollar Value')
    Savings, = plt.plot(log.wallet_history[0])
    Debt, = plt.plot(log.debt_history[0])
    Investments, = plt.plot(log.mining_amount[0]+log.bond_amount[0]+log.index_amount[0])
    plt.legend(handles=[Savings, Debt, Investments], labels=['Savings', 'Debt', 'Investments'], loc=0)
else:
    # Net change in total worth over trial
    plt.figure(1)
    net = []
    for i in range(NUMTRIALS):
        net.append(log.finalWorth[i] - log.initialWorth[i])
    plt.title('Net Worth at End of Trial')
    plt.ylabel('Value in Dollars')
    plt.xlabel('Trial Index')
    plt.plot(net[:])
    plt.plot([0, NUMTRIALS], [0, 0])

    # calculte values for ROI odds table
    pos, neg, ratio = 0, 0, 0
    for i in range(NUMTRIALS):
        if net[i] > 0:
            pos += 1
        if net[i] < 0:
            neg += 1
    if neg == 0:
        ratio = 'inf'
    else:
        ratio = float(pos) / neg
    print('Pos: {0}, Neg: {1}, Ratio: {2}\n').format(pos, neg, ratio)

    # ROI for trial
    plt.figure(2)
    plt.xlabel('Percent ROI Outcome')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of ROI in All-Weathers Strategy')
    hist(log.total_ROI[:])

    # relationship between outcome and bad events
    plt.figure(3)
    # rois = log.total_ROI[:]
    nets = net[:]
    bads = log.num_bad[:]
    plt.xlabel('Number of Bad Events')
    # plt.ylabel('Percent ROI')
    plt.ylabel('Total Worth Net Change (dollars)')
    plt.title('Investment Outcome Given Bad Luck')
    # plt.scatter(x=bads, y=rois)
    # box(bads, rois)
    box(bads, nets)
    plt.xticks(np.arange(1, len(np.unique(log.num_bad)) + 1), np.unique(log.num_bad))
    plt.plot([0, NUMTRIALS], [0, 0])

    # plots the distribution of outcomes in final net worth
    plt.figure(4)
    hist(log.finalWorth[:])
    plt.title('Net Worth at End of Trial')
    plt.ylabel('Value in Dollars')
    plt.xlabel('Trial Index')
    plt.plot([0, NUMTRIALS], [0, 0])

print '\npeace'
