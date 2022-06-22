#!/usr/bin/env python
# coding: utf-8

# IB eredua dugu kasu honetan. IB ereuduaren ebazpen zuzen bat ikusteko jo 19. Irudiko kodera.
# 
# Erabiliko diren moduloak inportatu:

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


# Erabilido diren aldagaien laburbilduma
# 
# |Aldagaia|Definizioa|Unitatea|
# |--------|-----------|-------|
# |$t_{max}$ |Simulazioa denbora maximoa|$ms$|
# |$\Delta t$|Denboraren diskretizazioa |$ms$|
# |$\tau_{max}$ |Parametro esperimentala|$s$|
# |$C_{m}$   |Mintz kapazitatea         |$\frac{\mu F}{cm^{2}}$|
# |$E_{Na}$  |Inbertsio potentziala, sodio kanaletarako|$mV$|
# |$E_{K}$   |Inbertsio potentziala, potasio kanaletarako|$mV$|
# |$E_{L}$   |Inbertsi potentziala|$mV$|
# |$E_{Ca}$   |Inbertsi potentziala, kaltzio kanaletarako|$mV$|
# |$g_{Na}$  |Konduktantzia espezifikoa, sodio korronte azkarrerako|$\frac{mS}{cm^{2}}$|
# |$g_{K}$   |Konduktantzia espezifikoa, potasio korronte iraukorrerako|$\frac{mS}{cm^{2}}$|
# |$g_{M}$   |Konduktantzia espezifikoa, potasio korronte geldorako|$\frac{mS}{cm^{2}}$|
# |$g_{L}$   |Konduktantzia espezifikoa|$\frac{mS}{cm^{2}}$|
# |$g_{Ca}$   |Konduktantzia espezifikoa, kaltzio korronte leherketarako|$\frac{mS}{cm^{2}}$|
# |$i_{mean}$|Batez-besteko intentsitate bat finkatu|$\frac{\mu A}{cm^{2}}$|
# |$V_{rest}$|Egoera egonkorreko mintz potentziala|$mV$|
# |$V_{Th}$|Atari potentziala|$mV$|

# In[2]:


#Parametroak finkatu#####
t_max=400      # ms
delta_t=0.04   # ms
tau_max=500.0  # s
C_m=3.14       # micro F/cm^2 (c_m/(L*d)=C_m)
E_Na=50.00     # mV
E_K=-90.00     # mV
E_L=-70.00     # mV
E_Ca=120.0     # mV
g_Na=50.0      # mS/cm^2
g_K=5.0        # mS/cm^2
g_M=0.07       # mS/cm^2
g_L=0.1        # mS/cm^2
g_Ca=0.1       # mS/cm^2  
i_mean=5.8     # microA/cm^2
V_rest=-65.0   # mV
v_th=-40.0     # mV


# ### Funtzio laguntzaileen inplementazioa
# 
# $$\alpha_{m}=\frac{-0.32(v-v_{Th}-13)}{\exp [-(v-v_{Th}-13)/4]-1}$$
# 
# $$\beta_{m}=\frac{0.28(v-v_{Th}-40)}{\exp [(v-v_{Th}-40)/5]-1}$$
# 
# $$\alpha_{h}=0.128\exp [-(v-v_{Th}-17)/18]$$
# 
# $$\beta_{h}=\frac{4}{1+ exp[-(v-v_{Th}-40)/5]}$$
# 
# $$\alpha_{n}=\frac{-0.032(v-v_{Th}-15)}{\exp [-(v-v_{Th}-15)/5]-1}$$
# 
# $$\beta_{n}=0.5 \exp[-(v-v_{Th}-10)/40]$$
# 
# $$p_{\infty}(v)=\frac{1}{1+\exp[-(v+35)/10]}$$
# 
# $$\tau_{p}(v)=\frac{\tau_{max}}{3.3\exp[(v+35)/20]+\exp[-(v+35)/20]}$$
# 
# $$\alpha_{q}=\frac{0.055(-27-v)}{\exp [(-27-v)/3.9]-1}$$
# 
# $$\beta_{q}=0.94 \exp[(-75-v)/17]$$
# 
# $$\alpha_{r}=0.000457 \exp[(-13-v)/50]$$
# 
# $$\beta_{r}=\frac{0.0065}{1+ exp[(-15-v)/28]}$$

# In[3]:


def alpha_m(v):
    return -0.32 * (v - v_th - 13.0) / (np.exp(-(v - v_th - 13.0) / 4.0) - 1.0)

def beta_m(v):
    return 0.28 * (v - v_th - 40.0) / (np.exp((v - v_th - 40.0) / 5.0) - 1.0)

def alpha_h(v):
    return 0.128 * np.exp(-(v - v_th - 17.0)/18.0)

def beta_h(v):
    return 4.0 / (1.0 + np.exp(-(v - v_th - 40.0) / 5.0))

def alpha_n(v):
    return -0.032 * (v - v_th - 15.0) / (np.exp(-(v - v_th - 15.0) / 5.0) - 1.0)

def beta_n(v):
    return 0.5 * np.exp(-(v - v_th - 10.0) / 40.0)

def p_infty(v):
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

def tau_p(v):
    return tau_max / (3.3 * np.exp((v + 35.0) / 20.0) + np.exp(-(v + 35.0) / 20.0))

def alpha_q(v):
    return 0.055 * (-27.0 - v) / (np.exp((-27.0 - v) / 3.8) - 1.0)

def beta_q(v):
    return 0.94 * np.exp((-75.0 - v) / 17.0)

def alpha_r(v):
    return 0.000457 * np.exp((-13.0 - v) / 50.0)

def beta_r(v):
    return 0.0065 / (np.exp((-15.0 - v) / 28.0) + 1.0)


# ### Bektoreak eta hasierako balioak finkatu

# In[4]:


#Denbora eremu osoa finkatzen dugu. 0 s-tik hasiz, delta_t diskretizazio denbora aldiunez t_max aldiunerarte
t_eremua=np.arange(0,t_max,delta_t)
#Luzera finkatu
step=len(t_eremua)

#Vt bektorea finkatu
Vt=np.ones([step])

#nt, mt, ht bektoreak finkatu
nt=np.ones([step])
mt=np.ones([step])
ht=np.ones([step])
pt=np.ones([step])
qt=np.ones([step])
rt=np.ones([step])


# Hasierako egoerak finkatu
# 
# $$n(0)=\frac{\alpha_{n}}{\alpha_{n} + \beta_{n}}|_{V_{rest}}$$
# 
# $$m(0)=\frac{\alpha_{m}}{\alpha_{m} + \beta_{m}}|_{V_{rest}}$$
# 
# $$h(0)=\frac{\alpha_{h}}{\alpha_{h} + \beta_{h}}|_{V_{rest}}$$
# 
# $$p(0)=p_{\infty}|_{V_{rest}}$$
# 
# $$q(0)=\frac{\alpha_{q}}{\alpha_{q} + \beta_{q}}|_{V_{rest}}$$
# 
# $$r(0)=\frac{\alpha_{r}}{\alpha_{r} + \beta_{r}}|_{V_{rest}}$$

# In[5]:


# Hasierako balioak
Vt[0]=V_rest
nt[0]= alpha_n(V_rest) / (alpha_n(V_rest) + beta_n(V_rest))
mt[0]= alpha_m(V_rest) / (alpha_m(V_rest) + beta_m(V_rest))
ht[0]= alpha_h(V_rest) / (alpha_h(V_rest) + beta_h(V_rest))
pt[0]= p_infty(V_rest)
qt[0]= alpha_q(V_rest) / (alpha_q(V_rest) + beta_q(V_rest))
rt[0]= alpha_r(V_rest) / (alpha_r(V_rest) + beta_r(V_rest))


# Hainbat gauza definituko dira: 
# 
# 1- Firing-rate-a emango digun bektorea definiftuko dugu. FR_vi motakoa izango da. i-k zenbatgarren spike-a den adieraziko digu. Bektoreko puntu bakoitzean intentsitate baterako i spike-eko FR-a gordeko da.
# 
# 2- Inter-spike interval emango digun bektorea definituko dugu. ISI_vi motakoa izango da. i-k zenbatgarren spike-a den adieraziko digu. Bektoreko puntu bakoitzean intentsitate baterako i spike-eko ISI-a gordeko da. 
# 
# 3- Noiz gertatu den spike-a finkatuko digu bektore honek. Luzerak berdin digu. Iñoiz ez dira step baino gehiagoko spike-ak emango. Bektoreko puntu bakoitzean spike bat noiz eman den gordeko da.

# In[6]:


t=np.ones([step])     #Noiz gertatu den spike-a finkatuko digu bektore honek. Luzerak berdin digu. Iñoiz ez dira step baino gehiagoko spike-ak emango.

FR_v1=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v1=np.ones([35])  #inter-spike interval emango digu.

FR_v2=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v2=np.ones([35])  #inter-spike interval emango digu.

FR_v3=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v3=np.ones([35])  #inter-spike interval emango digu.

FR_v4=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v4=np.ones([35])  #inter-spike interval emango digu.

FR_v5=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v5=np.ones([35])  #inter-spike interval emango digu.

FR_v6=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v6=np.ones([35])  #inter-spike interval emango digu.

FR_v7=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v7=np.ones([35])  #inter-spike interval emango digu.

FR_v8=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v8=np.ones([35])  #inter-spike interval emango digu.

FR_v9=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v9=np.ones([35])  #inter-spike interval emango digu.

FR_v10=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v10=np.ones([35])  #inter-spike interval emango digu.

FR_v11=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v11=np.ones([35])  #inter-spike interval emango digu.

FR_v12=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v12=np.ones([35])  #inter-spike interval emango digu.

FR_v13=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v13=np.ones([35])  #inter-spike interval emango digu.

FR_v14=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v14=np.ones([35])  #inter-spike interval emango digu.

FR_v15=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v15=np.ones([35])  #inter-spike interval emango digu.

FR_v16=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v16=np.ones([35])  #inter-spike interval emango digu.

FR_v17=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v17=np.ones([35])  #inter-spike interval emango digu.

FR_v18=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v18=np.ones([35])  #inter-spike interval emango digu.

FR_v19=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v19=np.ones([35])  #inter-spike interval emango digu.

FR_v20=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v20=np.ones([35])  #inter-spike interval emango digu.

FR_v21=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v21=np.ones([35])  #inter-spike interval emango digu.

FR_v22=np.ones([35])   #Firing-rate-a emango digu, spike bakoitzeko.
ISI_v22=np.ones([35])  #inter-spike interval emango digu.


# ### Gobernu ekuazioen ebazpena
# 
# Euler-en aurrerazko formula erabiliz diskretizazioa denboran egiteko:
# 
# $$v^{i+1}=v^{i}+\frac{\Delta t}{C_{m}}[I^{i}-\overline{g_{Na}}(m^{3})^{i}h^{i}(v^{i}-E_{Na})-\overline{g_{K}}(n^{4})^{i}(v^{i}-E_{K})-\overline{g_{M}}p^{i}(v^{i}-E_{K})-\overline{g_{Ca}}(q^{2})^{i}r^{i}(v^{i}-E_{Ca})-\overline{g_{L}}(v^{i}-E_{L})]$$
# 
# $$n^{i+1}=n^{i}+\Delta t[\alpha_{n}(v^{i})(1-n^{i})-\beta_{n}(v^{i})n^{i}]$$
# 
# $$m^{i+1}=m^{i}+\Delta t[\alpha_{m}(v^{i})(1-m^{i})-\beta_{m}(v^{i})m^{i}]$$
# 
# $$h^{i+1}=h^{i}+\Delta t[\alpha_{h}(v^{i})(1-h^{i})-\beta_{h}(v^{i})h^{i}]$$
# 
# $$p^{i+1}=p^{i}+\Delta t[\frac{p_{\infty}(v^{i})-p^{i}}{\tau_{p}(v^{i})}]$$
# 
# $$q^{i+1}=q^{i}+\Delta t[\alpha_{q}(v^{i})(1-q^{i})-\beta_{q}(v^{i})q^{i}]$$
# 
# $$r^{i+1}=r^{i}+\Delta t[\alpha_{r}(v^{i})(1-r^{i})-\beta_{r}(v^{i})r^{i}]$$
# 
# Estrategia zuzenekoa da. I izeneko intentsitate bat erabiliko dugu. k bakoitzerako i_mean + 0.1*k intentsitatearekin RS neurona ebatziko da. Hau 35 aldiz egingo da.
# 
# k bakoitzerako, ebazpenean zehar Vt>0 bada, ekintza potentzial bat gauzatu dela esan nahiko du. If-ean sartuko da gure kodea. j indizearekin zenbatgarren spike-a den gordeko da. Beti ere 1. spike-a 0 indizea izango du, 2. spike-ak 1 indizea... Momentu horretan t bektorean j spike-a eman den aldiunea gordeko da i*delta_t balioarekin. Momentu honetan, Vt>0 izaten jarraituko duenez ekintza potentziala bukatu arte, m=1 definituko da, lehenengo if-a blokeatuz. Vt<0-ra itzultzen denean, bigarren if-ean sartuko gara, m=0 finkatu eta lehenengo if-a desblokeatuko da. Honela berriz ere prozesua hasiz.
# 
# k baterako, t_max denbora osorako Vt-ren ebazpen egin denean  ISI_vi eta FR_Vi-ren balioak gordetzen joango dara. Kasu honetan lehenego 22 spike-ak gorde dira.

# In[7]:


for k in range(0,35):
    j=0
    m=0
    I=i_mean + 0.1*k
    for i in range(0, step-1) :
        nt[i + 1] = nt[i] + delta_t * (alpha_n(Vt[i]) * (1.0 - nt[i]) - beta_n(Vt[i]) * nt[i])
        mt[i + 1] = mt[i] + delta_t * (alpha_m(Vt[i]) * (1.0 - mt[i]) - beta_m(Vt[i]) * mt[i])
        ht[i + 1] = ht[i] + delta_t * (alpha_h(Vt[i]) * (1.0 - ht[i]) - beta_h(Vt[i]) * ht[i])
        pt[i + 1] = pt[i] + delta_t * ((p_infty(Vt[i]) - pt[i]) / tau_p(Vt[i]))
        qt[i + 1] = qt[i] + delta_t * (alpha_q(Vt[i]) * (1.0 - qt[i]) - beta_q(Vt[i]) * qt[i])
        rt[i + 1] = rt[i] + delta_t * (alpha_r(Vt[i]) * (1.0 - rt[i]) - beta_r(Vt[i]) * rt[i])
        Vt[i + 1] = Vt[i] + (delta_t/C_m) * (I-g_Na * mt[i] ** 3 * ht[i] * (Vt[i]-E_Na) - g_K * nt[i] ** 4 * (Vt[i]-E_K)- g_Ca * qt[i]** 2 * rt[i] * (Vt[i] - E_Ca) - g_M * pt[i] * (Vt[i] - E_K) - g_L * (Vt[i]-E_L))
        
        #Fire-egiteko aldiunea kalkulatu
        if Vt[i+1]>=0.0 and m==0: # Fire egin duela zihurtatu.
            t[j]=i * delta_t      # i*delta_t-ek gure aldiunea ematen du milisegundutan.
            j=j+1                 # Spike zenbakia handitu hurrengo spike-reako.
            m=1                   # m=1 bihurtu.

        #spike-a-ren amaiera lortu eta parametroak ezarri
        if Vt[i+1]<0.0 and m==1:  # m parametroak berrezarri.
            m=0                   # m=0 berrezarri.
                                  #spike-a amaitu da, m parametroa hasierako egoerara eman
    
    #ISI eta FR intentsitate bakoitzerako (k bakoitzerako) guk nahi dugun spike-an ISI_vi moduan zenbatuz non i spike-aren ordena den.
    
    ISI_v1[k]= (t[1]-t[0]) * 0.001 #Segundutara bihurtu 
    FR_v1[k]= 1.0/ISI_v1[k] 
    
    ISI_v2[k]= (t[2]-t[1]) * 0.001 #Segundutara bihurtu 
    FR_v2[k]= 1.0/ISI_v2[k]
    
    ISI_v3[k]= (t[3]-t[2]) * 0.001 #Segundutara bihurtu 
    FR_v3[k]= 1.0/ISI_v3[k]
    
    ISI_v4[k]= (t[4]-t[3]) * 0.001 #Segundutara bihurtu 
    FR_v4[k]= 1.0/ISI_v4[k]
    
    ISI_v5[k]= (t[5]-t[4]) * 0.001 #Segundutara bihurtu 
    FR_v5[k]= 1.0/ISI_v5[k]
    
    ISI_v6[k]= (t[6]-t[5]) * 0.001 #Segundutara bihurtu 
    FR_v6[k]= 1.0/ISI_v6[k]
    
    ISI_v7[k]= (t[7]-t[6]) * 0.001 #Segundutara bihurtu 
    FR_v7[k]= 1.0/ISI_v7[k]
    
    ISI_v8[k]= (t[8]-t[7]) * 0.001 #Segundutara bihurtu 
    FR_v8[k]= 1.0/ISI_v8[k]
    
    ISI_v9[k]= (t[9]-t[8]) * 0.001 #Segundutara bihurtu 
    FR_v9[k]= 1.0/ISI_v9[k]
    
    ISI_v10[k]= (t[10]-t[9]) * 0.001 #Segundutara bihurtu 
    FR_v10[k]= 1.0/ISI_v10[k]
    
    ISI_v11[k]= (t[11]-t[10]) * 0.001 #Segundutara bihurtu 
    FR_v11[k]= 1.0/ISI_v10[k]
    
    ISI_v12[k]= (t[12]-t[11]) * 0.001 #Segundutara bihurtu 
    FR_v12[k]= 1.0/ISI_v10[k]
    
    ISI_v13[k]= (t[13]-t[12]) * 0.001 #Segundutara bihurtu 
    FR_v13[k]= 1.0/ISI_v10[k]
    
    ISI_v14[k]= (t[14]-t[13]) * 0.001 #Segundutara bihurtu 
    FR_v14[k]= 1.0/ISI_v10[k]
    
    ISI_v15[k]= (t[15]-t[14]) * 0.001 #Segundutara bihurtu 
    FR_v15[k]= 1.0/ISI_v10[k]
    
    ISI_v16[k]= (t[16]-t[15]) * 0.001 #Segundutara bihurtu 
    FR_v16[k]= 1.0/ISI_v10[k]
    
    ISI_v17[k]= (t[17]-t[16]) * 0.001 #Segundutara bihurtu 
    FR_v17[k]= 1.0/ISI_v10[k]
    
    ISI_v18[k]= (t[18]-t[17]) * 0.001 #Segundutara bihurtu 
    FR_v18[k]= 1.0/ISI_v10[k]
    
    ISI_v19[k]= (t[19]-t[18]) * 0.001 #Segundutara bihurtu 
    FR_v19[k]= 1.0/ISI_v10[k]
    
    ISI_v20[k]= (t[20]-t[19]) * 0.001 #Segundutara bihurtu 
    FR_v20[k]= 1.0/ISI_v10[k]
    
    ISI_v21[k]= (t[21]-t[20]) * 0.001 #Segundutara bihurtu 
    FR_v21[k]= 1.0/ISI_v10[k]
    
    ISI_v22[k]= (t[22]-t[21]) * 0.001 #Segundutara bihurtu 
    FR_v22[k]= 1.0/ISI_v10[k]


# Ploteatzeko intentsitate bektore bat sortu.

# In[8]:


Inten=np.ones([35])
for i in range(0,35):
    Inten[i]= i_mean + i*0.1


# Ploteatu:

# In[9]:


#################################  ### 00 ###

plt.rc('text', usetex=True)

#################################
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#################################  ### 01 ###
fontsize = 25
labelsize = 20
#################################

fig = plt.figure(constrained_layout=True, figsize=(labelsize,labelsize))

spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax1.plot(Inten, FR_v1, 'r', label= r"$\textrm{spike}_1$")
f_ax1.plot(Inten, FR_v2, 'g', label= r"$\textrm{spike}_2$")
f_ax1.plot(Inten, FR_v3, 'b', label= r"$\textrm{spike}_3$")
f_ax1.plot(Inten, FR_v4, 'y', label= r"$\textrm{spike}_4$")
f_ax1.plot(Inten, FR_v5, 'purple', label= r"$\textrm{spike}_5$")
f_ax1.plot(Inten, FR_v6, 'c', label= r"$\textrm{spike}_6$")
#f_ax1.plot(Inten, FR_v7, 'm', label= "spike_7")
#f_ax1.plot(Inten, FR_v8, 'yellow', label= "spike_8")
#f_ax1.plot(Inten, FR_v9, 'olive', label= "spike_9")
#f_ax1.plot(Inten, FR_v10, 'grey', label= "spike_10")
#f_ax1.plot(Inten, FR_v11, 'silver', label= "spike_11")
#f_ax1.plot(Inten, FR_v12, 'aqua', label= "spike_12")
#f_ax1.plot(Inten, FR_v13, 'gold', label= "spike_13")
#f_ax1.plot(Inten, FR_v14, 'tan', label= "spike_14")
#f_ax1.plot(Inten, FR_v15, 'lightgray', label= "spike_15")
#f_ax1.plot(Inten, FR_v16, 'lime', label= "spike_16")
f_ax1.plot(Inten, FR_v17, 'darkgreen', label= r"$\textrm{spike}_{17}$")
f_ax1.plot(Inten, FR_v18, 'lavender', label= r"$\textrm{spike}_{18}$")
f_ax1.plot(Inten, FR_v19, 'royalblue', label= r"$\textrm{spike}_{19}$")
f_ax1.plot(Inten, FR_v20, 'magenta', label= r"$\textrm{spike}_{20}$")
f_ax1.plot(Inten, FR_v21, 'skyblue', label= r"$\textrm{spike}_{21}$")
f_ax1.plot(Inten, FR_v22, 'brown', label= r"$\textrm{spike}_{22}$")

f_ax1.set_xlabel(r'$i_{e} \textrm{ (mA/cm}^{2})$', fontsize=fontsize)
f_ax1.set_ylabel(r'$FR \textrm{ (Hz)}$',fontsize=fontsize)
f_ax1.set_title(r'$\textrm{FR vs } i_{e}$',fontsize=fontsize)

f_ax1.legend(bbox_transform=f_ax1.transData, bbox_to_anchor=(9, 110), ncol=2, borderaxespad=0, 
             frameon=False, fontsize=23)

axes_fig = [f_ax1] # irudiak dazkanaren arabera

for i in axes_fig:
    i.tick_params(axis='y', labelsize=labelsize, pad=5, length=10);
    i.tick_params(axis='x', labelsize=labelsize, pad=5, length=10);


plt.savefig('IB_Ereduko_Firing_Rate.pdf', format='pdf', dpi=180,bbox_inches="tight")


# In[ ]:




