# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:52:14 2016

@author: kramerPro
"""
# %%
## use pip install quandl

import quandl
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

# my api key --> hope no-one abuses this
quandl.ApiConfig.api_key = 'zFCX5bmbwZvgGzHu5szi'

# can copy and paste commands after finding what you want on the website
# easy to get date ranges: could mix and overlap 
# help for python commands:
# https://www.quandl.com/tools/python
snp_index = quandl.get("YAHOO/FUND_VFINX", authtoken="zFCX5bmbwZvgGzHu5szi")
mining_eft = quandl.get("YAHOO/FUND_VGPMX", authtoken="zFCX5bmbwZvgGzHu5szi")
total_bond = quandl.get("YAHOO/FUND_VBMFX", authtoken="zFCX5bmbwZvgGzHu5szi")

snp_index_rdiff = quandl.get("YAHOO/FUND_VFINX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
mining_eft_rdiff = quandl.get("YAHOO/FUND_VGPMX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
total_bond_rdiff = quandl.get("YAHOO/FUND_VBMFX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")

# visualize distributions
f1, (plot1, plot2, plot3) = plt.subplots(3, sharex=True, sharey=True)
plot1.hist(snp_index.Close, 20)
plot2.hist(mining_eft.Close, 20)
plot3.hist(total_bond.Close, 20)
f1.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f1.axes[:-1]], visible=False)
plt.savefig('MarketRBP.png')
plt.show()

f2, (plot1, plot2, plot3) = plt.subplots(3, sharex=True, sharey=True)
plot1.hist(snp_index_rdiff.Close, 20)
plot2.hist(mining_eft_rdiff.Close, 20)
plot3.hist(total_bond_rdiff.Close, 20)
f2.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f2.axes[:-1]], visible=False)
plt.savefig('MarketDiffRBP.png')
plt.show()

a = np.asarray(snp_index_rdiff.Close)
b = np.asarray(mining_eft_rdiff.Close)
c = np.asarray(total_bond_rdiff.Close)

loc1, scale1 = st.norm.fit(a)
loc2, scale2 = st.norm.fit(b)
loc3, scale3 = st.norm.fit(c)

print loc1, scale1
print loc2, scale2
print loc3, scale3

#%%
# grab and store data for grab

quandl.ApiConfig.api_key = 'zFCX5bmbwZvgGzHu5szi'
snp_index = quandl.get("YAHOO/FUND_VFINX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
mining_eft = quandl.get("YAHOO/FUND_VGPMX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")
total_bond = quandl.get("YAHOO/FUND_VBMFX", authtoken="zFCX5bmbwZvgGzHu5szi", transform="rdiff")

snp_index.to_csv('snp_index.csv')
mining_eft.to_csv('mining_eft.csv')
total_bond.to_csv('total_bond.csv')
#%%