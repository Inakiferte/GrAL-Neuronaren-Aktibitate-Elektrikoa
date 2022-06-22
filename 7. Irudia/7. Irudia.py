#!/usr/bin/env python
# coding: utf-8

# Erabiliko diren moduloak inportartu:

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


# Aldagaien Laburbilduma:
# 
# |Aldagaia|Definizioa|Unitatea|
# |--------|-----------|-------|
# |$t_{max}$ |Simulazioa denbora maximoa|$ms$|
# |$\Delta t$|Denboraren diskretizazioa |$ms$|
# |$C_{m}$   |Mintz kapazitatea         |$\frac{\mu F}{cm^{2}}$|
# |$E_{Na}$  |Inbertsio potentziala, sodio kanaletarako|$mV$|
# |$E_{K}$   |Inbertsio potentziala, potasio kanaletarako|$mV$|
# |$E_{L}$   |Inbertsi potentziala|$mV$|
# |$g_{Na}$  |Konduktantzia espezifikoa, sodio kanaletarako|$\frac{mS}{cm^{2}}$|
# |$g_{K}$   |Konduktantzia espezifikoa, potasio kanaletarako|$\frac{mS}{cm^{2}}$|
# |$g_{L}$   |Konduktantzia espezifikoa|$\frac{mS}{cm^{2}}$|
# |$i_{mean}$|Batez-besteko intentsitate bat finkatu|$$\frac{\mu A}{cm^{2}}$$|
# |$V_{rest}$|Egoera egonkorreko mintz potentziala|$mV$|

# In[2]:


#Parametroak finkatu#####
t_max=100      #msegundu, Simulazio denbora osoa
delta_t=0.01   #msegundu, Diskretizazio denbora
C_m=0.01       #micro F/cm^2
E_Na=55.17     # mV
E_K=-72.14     # mV
E_L=-49.42     # mV
g_Na=1.2       # mS/cm^2
g_K=0.36       # mS/cm^2
g_L=0.003      # mS/cm^2
i_mean=0.1     # microA/cm^2
V_rest=-60.0   # mV
##########################


# Erabiliko diren funtzio laguntzaileak:
# 
# $$\alpha_{n}=\frac{0.01(v+50)}{1-exp(\frac{-(v+50)}{10})}$$
# 
# $$\beta_{n}=0.125exp(\frac{-(v+60)}{80})$$
# 
# $$\alpha_{m}=\frac{0.1(v+35)}{1-exp(\frac{-(v+35)}{10})}$$
# 
# $$\beta_{m}=4.0exp(-0.0556(v+60))$$
# 
# $$\alpha_{h}=0.07exp(-0.05(v+60))$$
# 
# $$\beta_{h}=\frac{1}{1+exp(-0.1(v+30))}$$

# In[3]:


#sarrera mvolt-etan izan behar da
def alpha_n(v):
    return 0.01*(v+50.0)/(1.0-np.exp(-(v+50.0)/10.0))

def beta_n(v):
    return 0.125*np.exp(-(v+60.0)/80.0)

def alpha_m(v):
    return 0.1*(v+35.0)/(1.0-np.exp(-(v+35.0)/10.0))

def beta_m(v):
    return 4.0*np.exp(-0.0556*(v+60.0))

def alpha_h(v):
    return 0.07*np.exp(-0.05*(v+60.0))

def beta_h(v):
    return 1/(1+np.exp(-0.1*(v+30.0)))


# Bektoreak finkatu:

# In[4]:


#Denbora eremu osoa finkatzen dugu. 0 s-tik hasiz, delta_t diskretizazio denbora aldiunez t_max aldiunerarte
t_eremua=np.arange(0,t_max,delta_t)

#Luzera finkatu
step=len(t_eremua)

#Vt bektorea finkatu
Vt=np.ones([step])

#Fase espaziorako bektoreaa finkatu
V_dot=np.ones([step])

#Intentsitatea bektore baten gorde
It=i_mean*np.ones([step])

#nt, mt, ht bektoreak finkatu
nt=np.ones([step])
mt=np.ones([step])
ht=np.ones([step])

#Fase espaziorako bektoreak finkatu
n_dot=np.ones([step])
m_dot=np.ones([step])
h_dot=np.ones([step])

#Hasierako balioak finkatu#
Vt[0]=V_rest
nt[0]=alpha_n(V_rest) / (alpha_n(V_rest) + beta_n(V_rest))
mt[0]=alpha_m(V_rest) / (alpha_m(V_rest) + beta_m(V_rest))
ht[0]=alpha_h(V_rest) / (alpha_h(V_rest) + beta_h(V_rest))


# Ebatzi beharreko ekuazioa:
# 
# $$\frac{dv}{dt}=\frac{1}{C_{m}}[I-g_{Na}m^{3}h(v-E_{Na})-g_{K}n^{4}(v-E_{K})-g_{L}(v-E_{L})]$$
# 
# $$\frac{dn}{dt}=\alpha_{n}(v)(1-n)-\beta_{n}(v)n$$
# 
# $$\frac{dm}{dt}=\alpha_{m}(v)(1-m)-\beta_{m}(v)m$$
# 
# $$\frac{dh}{dt}=\alpha_{h}(v)(1-h)-\beta_{h}(v)h$$

# Euler-en aurrerazko formula erabiliz diskretizazioa denboran egiteko:
# 
# $$v^{i+1}=v^{i}+\frac{\Delta t}{C_{m}}[I^{i}-g_{Na}(m^{3})^{i}h^{i}(v^{i}-E_{Na})-g_{K}(n^{4})^{i}(v^{i}-E_{K})-g_{L}(v^{i}-E_{L})]$$
# 
# $$n^{i+1}=n^{i}+\Delta t[\alpha_{n}(v^{i})(1-n^{i})-\beta_{n}(v^{i})n^{i}]$$
# 
# $$m^{i+1}=m^{i}+\Delta t[\alpha_{m}(v^{i})(1-m^{i})-\beta_{m}(v^{i})m^{i}]$$
# 
# $$h^{i+1}=h^{i}+\Delta t[\alpha_{h}(v^{i})(1-h^{i})-\beta_{h}(v^{i})h^{i}]$$

# In[5]:


for i in range(0, step-1) :
    mt[i + 1] = mt[i] + delta_t * (alpha_m(Vt[i]) * (1 - mt[i]) - beta_m(Vt[i]) * mt[i])
    ht[i + 1] = ht[i] + delta_t * (alpha_h(Vt[i]) * (1 - ht[i]) - beta_h(Vt[i]) * ht[i])
    nt[i + 1] = nt[i] + delta_t * (alpha_n(Vt[i]) * (1 - nt[i]) - beta_n(Vt[i]) * nt[i])
    Vt[i + 1] = Vt[i] + (delta_t/C_m) * (It[i]-g_Na * mt[i] ** 3 * ht[i] * (Vt[i]-E_Na) - g_K * nt[i] ** 4 * (Vt[i]-E_K) - g_L * (Vt[i]-E_L)) 


# Ploteatu:

# In[6]:


#################################  ### 00 ###

plt.rc('text', usetex=True)

#################################

fig = plt.figure();

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#################################  ### 01 ###
fontsize = 25
labelsize = 20
#################################


fig, ax1 = plt.subplots(figsize=(11, 8))
ax1 = plt.subplot(221)
ax1.set_xlim(0,100)
ax1.plot(t_eremua, nt, 'r', label= "$n$")
ax1.plot(t_eremua, mt, 'b', label= "$m$")
ax1.plot(t_eremua, ht, 'g', label= "$h$")
ax1.set_ylabel(r'$n, \textrm{m} \textrm{ eta } \textrm{h}$',fontsize=fontsize)
ax1.set_xlabel(r'$t \quad \textrm{(ms)}$',fontsize=fontsize)
fig, ax1.set_title(r'$\textrm{Aktibazio-aldagaiak}$',fontsize=fontsize)
plt.legend()


ax2 = plt.subplot(222)
ax2.set_xlim(0,10)
ax2.plot(t_eremua, nt, 'r', label= r"$n$")
ax2.plot(t_eremua, mt, 'b', label= r"$m$")
ax2.plot(t_eremua, ht, 'g', label= r"$h$")
#ax2.set_ylabel('$n$, $m$ eta $h$',fontsize=15)
ax2.set_xlabel(r'$t \quad \textrm{(ms)}$',fontsize=fontsize)
fig, ax2.set_title(r'$\textrm{Aktibazio-aldagaiak}$',fontsize=fontsize)
plt.legend()
######################################  ### 04 ###
axes_fig = [ax1,ax2] # irudiak dazkanaren arabera

for i in axes_fig:
    i.tick_params(axis='y', labelsize=labelsize, pad=5, length=10);
    i.tick_params(axis='x', labelsize=labelsize, pad=5, length=10);

plt.savefig('H-H_Eredua_Aktibazio_Aldagaiak.pdf', format='pdf', dpi=180,bbox_inches="tight")

plt.show()


# In[ ]:




