# India air pollution types: K-means clustering (Tanmay Gautam, 24151020)
# Paste this whole file into one Google Colab cell and run. Data loads from GitHub.

RAW = 'https://raw.githubusercontent.com/tanmayg122/india-air-pollution-tracker/main/'

# # Air pollution types across 50 Indian cities (2022 to 2026) with unsupervised learning
# **Tanmay Gautam, 24151020 | DEC 351 Fundamentals of AI/ML**
# 
# AQI gives one number. This notebook groups city-days by their pollutant mix with K-means, then uses weather (wind, rain, humidity) and season to explain each group, so a city can see what kind of pollution it has, when and where, and act on the cause.
# 
# Data:
# 1. India Daily Weather and Air Quality (Kaggle): daily air quality and weather for 50 cities, 5 Aug 2022 to 30 Oct 2026. Air quality values are model estimates (Open-Meteo, Copernicus CAMS), weather from Open-Meteo.
# 2. CPCB hourly station AQI (Vonter/india-cpcb-aqi, from the CPCB AQI repository), 2017 to 2023. Real station readings, used to check the estimates.

# %%
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, os, json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
warnings.filterwarnings('ignore'); os.makedirs('figs', exist_ok=True)
RS=42; INK,INK2,GRID='#111111','#5c5c5c','#e6e6e6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#9a9a9a','xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK2,'axes.grid':True,'grid.color':GRID,
 'axes.axisbelow':True,'legend.frameon':False,'savefig.dpi':200,'savefig.bbox':'tight'})
def save(fig,n): fig.savefig(f'figs/{n}.png'); plt.show()
import urllib.request; INDIA=json.load(urllib.request.urlopen(RAW+'india_outline.json'))
def outline(ax):
    for p in INDIA: a=np.array(p); ax.fill(a[:,0],a[:,1],color='#f3f3f3',ec='#9a9a9a',lw=.7,zorder=0)
POL=['pm2_5_mean','pm10_mean','no2_mean','so2_mean','o3_8h_max','co_8h_max','dust_mean']
LBL={'pm2_5_mean':'PM2.5','pm10_mean':'PM10','no2_mean':'NO2','so2_mean':'SO2','o3_8h_max':'O3','co_8h_max':'CO','dust_mean':'Dust','ratio':'PM2.5/PM10'}
MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ## 1. Load and join the two files

# %%
aq=pd.read_csv(RAW+'data/air_quality_daily.csv',parse_dates=['date']); wx=pd.read_csv(RAW+'data/weather_daily.csv',parse_dates=['date']); cs=pd.read_csv(RAW+'data/cities.csv')
print('air quality', aq.shape, aq.date.min().date(), aq.date.max().date(), aq.city.nunique(), 'cities')
print('weather', wx.shape, wx.date.min().date(), wx.date.max().date())
d=aq.merge(wx,on=['date','city'],how='left').merge(cs,on='city',how='left'); print('joined', d.shape)
print(d[['date','city']+POL+['wind_speed_10m_max','wind_direction_10m_dominant','precipitation_sum']].head())

# ## 2. Data quality and cleaning

# %%
log={'City-days in file':len(d)}
log['Duplicate city-days']=int(d.duplicated(['city','date']).sum()); d=d.drop_duplicates(['city','date'])
log['Days with fewer than 24 hourly values']=int((d.hours_available<24).sum())
log['Missing weather after join']=int(d.wind_speed_10m_max.isna().sum())
log['Missing pollutant values']=int(d[POL].isna().sum().sum())
log['PM2.5 above PM10 (impossible)']=int((d.pm2_5_mean>d.pm10_mean).sum())
cap=d[POL].quantile(.999); over=(d[POL]>cap).any(axis=1); log['Extreme values above 99.9th percentile, capped']=int(over.sum())
d[POL]=d[POL].clip(upper=cap,axis=1)
log['Final city-days']=len(d)
for k,v in log.items(): print(f'{k:50s} {v}')

# %%
# Check the model estimates against real CPCB station readings (Aug 2022 to Dec 2023)
b=pd.read_parquet(RAW+'data/cpcb-aqi.parquet'); hrs=[c for c in b.columns if c.endswith(':00:00')]
b['aqi']=b[hrs].astype('float').mean(axis=1); b=b.dropna(subset=['aqi']); b['Date']=pd.to_datetime(b.Date)
b=b[b.Date>='2022-08-05']; b['ym']=b.Date.dt.to_period('M')
real=b.groupby(['City','ym']).aqi.mean().rename('real')
d['ym']=d.date.dt.to_period('M'); est=d.groupby(['city','ym']).india_aqi.mean().rename('est'); est.index.names=['City','ym']
chk=pd.concat([real,est],axis=1,join='inner').reset_index(); print(chk.City.nunique(),'cities,',len(chk),'city-months overlap')
r_all=chk[['real','est']].corr().iloc[0,1]; print('correlation of monthly AQI, real vs estimate:', round(r_all,2))
print('median real / estimate', round((chk.real/chk.est).median(),2))
fig,ax=plt.subplots(figsize=(5.2,4.6)); ax.scatter(chk.est,chk.real,s=10,alpha=.5,color=INK2)
m=max(chk.real.max(),chk.est.max()); ax.plot([0,m],[0,m],color='#C8372D',lw=1,ls='--')
ax.set_xlabel('Estimated AQI (dataset 1)'); ax.set_ylabel('Real CPCB station AQI'); ax.set_title(f'Monthly AQI per city, r = {r_all:.2f}',loc='left'); save(fig,'validate_real')

# ## 3. EDA

# %%
fig,axs=plt.subplots(2,4,figsize=(13,5.4))
for ax,p in zip(axs.flat,POL):
    ax.hist(d[p],bins=60,color=INK2); ax.axvline(d[p].median(),color='#C8372D',lw=1.5); ax.set_title(f'{LBL[p]} (median {d[p].median():.0f})',loc='left'); ax.set_yticks([])
axs.flat[-1].hist(d.wind_speed_10m_max,bins=60,color='#2A78D6'); axs.flat[-1].set_title('Wind speed max (km/h)',loc='left'); axs.flat[-1].set_yticks([])
fig.tight_layout(); save(fig,'dist'); print(d[POL].skew().round(1).to_dict())

# %%
med=d.groupby('city').pm2_5_mean.median().sort_values()
fig,ax=plt.subplots(figsize=(7,9)); ax.barh(med.index,med.values,color=['#C8372D' if v>=40 else '#9a9a9a' for v in med.values])
ax.axvline(40,color=INK,ls='--',lw=1); ax.text(41,1,'NAAQS annual limit 40',fontsize=9); ax.set_xlabel('Median PM2.5 (µg/m³)')
ax.tick_params(axis='y',labelsize=8); ax.set_title('Median PM2.5 by city, 2022 to 2026',loc='left'); save(fig,'cities')
cm=d.groupby('city').agg(pm=('pm2_5_mean','mean'),lat=('latitude','first'),lon=('longitude','first'))
fig,ax=plt.subplots(figsize=(5.6,6.2)); outline(ax); s=ax.scatter(cm.lon,cm.lat,c=cm.pm,s=cm.pm*4,cmap='YlOrRd',edgecolor=INK,lw=.4)
for c,r in cm.iterrows():
    if r.pm>48 or c in ['Mumbai','Chennai','Bengaluru','Kolkata','Leh','Port Blair']: ax.text(r.lon+.4,r.lat,c,fontsize=7.5)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [sp.set_visible(False) for sp in ax.spines.values()]; plt.colorbar(s,ax=ax,shrink=.5,label='Mean PM2.5'); ax.set_title('Where PM2.5 is highest',loc='left'); save(fig,'map_pm')

# %%
mon=d.groupby(['city',d.date.dt.month]).pm2_5_mean.median().unstack(0)
fig,ax=plt.subplots(figsize=(9,3.8))
for c,col in zip(['Delhi','Lucknow','Kolkata','Jodhpur','Mumbai','Bengaluru','Chennai'],['#C8372D','#E8A33D','#6250D6','#A0652A','#2A78D6','#1BAF7A','#5c5c5c']):
    ax.plot(mon.index,mon[c],marker='o',color=col,label=c,lw=2)
ax.set_xticks(range(1,13)); ax.set_xticklabels(MON); ax.set_ylabel('Median PM2.5'); ax.legend(ncol=4,fontsize=9)
ax.set_title('Season: north India spikes Nov to Jan, south and coast stay low',loc='left'); save(fig,'season')
cr=d[POL+['wind_speed_10m_max','relative_humidity_2m_mean','precipitation_sum']].corr()
lab=[LBL.get(c,c) for c in POL]+['Wind','Humidity','Rain']
fig,ax=plt.subplots(figsize=(7,5.6)); im=ax.imshow(cr,cmap='RdGy_r',vmin=-1,vmax=1); ax.grid(False)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=45,ha='right'); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(len(lab)):
    for j in range(len(lab)): ax.text(j,i,f'{cr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(cr.iloc[i,j])>.5 else INK)
ax.set_title('Correlation: pollutants and weather',loc='left'); save(fig,'corr')
print(cr.round(2).iloc[:,-3:])

# ## 4. Features and PCA

# %%
d['ratio']=d.pm2_5_mean/d.pm10_mean
X=np.log1p(d[POL]); X['ratio']=d.ratio; FEAT=list(X.columns)
sc=StandardScaler().fit(X); Z=sc.transform(X)
pca=PCA(2,random_state=RS).fit(Z); P=pca.transform(Z); print('PCA', pca.explained_variance_ratio_.round(3), pca.explained_variance_ratio_.sum().round(3))
fig,ax=plt.subplots(figsize=(6,4.5)); ax.scatter(P[:,0],P[:,1],s=1,alpha=.15,color=INK2); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('77,400 city-days on 2 PCA axes, before clustering',loc='left'); save(fig,'pca_raw')

# ## 5. Choosing k

# %%
idx=np.random.RandomState(0).choice(len(Z),6000,replace=False); rows=[]
for k in range(2,10):
    m=KMeans(k,n_init=10,random_state=RS).fit(Z); rows.append([k,m.inertia_,silhouette_score(Z[idx],m.labels_[idx]),davies_bouldin_score(Z,m.labels_)])
ks=pd.DataFrame(rows,columns=['k','inertia','silhouette','davies_bouldin'])
def stab(k,n=8):
    base=KMeans(k,n_init=10,random_state=RS).fit(Z); o=[]
    for s in range(n):
        r=np.random.RandomState(s).choice(len(Z),int(.8*len(Z)),replace=False); m=KMeans(k,n_init=10,random_state=s).fit(Z[r]); o.append(adjusted_rand_score(base.predict(Z[r]),m.labels_))
    return np.mean(o)
ks['stability']=[stab(k) for k in ks.k]; print(ks.round(3).to_string())
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
axs[0].plot(ks.k,ks.inertia/1000,marker='o',color=INK); axs[0].axvline(4,color='#C8372D',ls='--'); axs[0].set_title('Elbow: inertia (thousands)',loc='left'); axs[0].set_xlabel('k')
axs[1].plot(ks.k,ks.silhouette,marker='o',color=INK,label='Silhouette'); axs[1].plot(ks.k,ks.stability/3,marker='s',color='#9a9a9a',label='Stability / 3')
axs[1].axvline(4,color='#C8372D',ls='--'); axs[1].set_xlabel('k'); axs[1].legend(); axs[1].set_title('Silhouette and stability',loc='left'); save(fig,'k_select')

# ## 6. Final model: K-means, k = 4

# %%
K=4; km=KMeans(K,n_init=10,random_state=RS).fit(Z); d['raw_t']=km.labels_
prof=d.groupby('raw_t')[POL+['ratio']].median()
names={}
names[prof.no2_mean.idxmax()]='Winter smog'; names[prof.dust_mean.idxmax()]='Dust'; names[prof.pm2_5_mean.idxmin()]='Clean, rain washed'
for i in range(K): names.setdefault(i,'Moderate haze')
print(names)
ORDER=['Clean, rain washed','Moderate haze','Dust','Winter smog']
COL={'Clean, rain washed':'#1BAF7A','Moderate haze':'#2A78D6','Dust':'#E8A33D','Winter smog':'#C8372D'}
d['type']=d.raw_t.map(names)
g=d.groupby('type')
summary=pd.DataFrame({'days':g.size(),'share_%':(g.size()/len(d)*100).round(1)})
for p in POL+['ratio']: summary[LBL[p]]=g[p].median().round(2)
summary['AQI']=g.india_aqi.median(); summary['wind']=g.wind_speed_10m_max.median(); summary['humidity']=g.relative_humidity_2m_mean.median(); summary['rain']=g.precipitation_sum.mean().round(1)
for nm,mm in [('winter_%',[11,12,1]),('summer_%',[3,4,5]),('monsoon_%',[6,7,8,9])]: summary[nm]=(g.date.apply(lambda s: s.dt.month.isin(mm).mean())*100).round(0)
summary=summary.loc[ORDER]; print(summary.T.to_string())

# %%
fig,ax=plt.subplots(figsize=(7,4.6))
for t in ORDER:
    m=(d.type==t).values; ax.scatter(P[m,0],P[m,1],s=1,alpha=.25,color=COL[t],label=t)
ax.legend(markerscale=8,fontsize=9); ax.set_xlabel('PC1: overall pollution'); ax.set_ylabel('PC2'); ax.set_title('4 pollution types on the PCA map',loc='left'); save(fig,'types_pca')
zp=pd.DataFrame(Z,columns=FEAT).groupby(d.type.values).mean().loc[ORDER]
fig,ax=plt.subplots(figsize=(8.5,3)); ax.imshow(zp.values,cmap='RdGy_r',vmin=-1.6,vmax=1.6,aspect='auto'); ax.grid(False)
ax.set_xticks(range(len(FEAT))); ax.set_xticklabels([LBL[f] for f in FEAT]); ax.set_yticks(range(K)); ax.set_yticklabels(ORDER)
for i in range(K):
    for j in range(len(FEAT)): ax.text(j,i,f'{zp.values[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(zp.values[i,j])>.9 else INK)
ax.set_title('Type profile: above (red) or below (grey) the average day',loc='left'); save(fig,'types_heat')

# ## 7. Baseline: the official AQI category

# %%
bi=np.random.RandomState(1).choice(len(d),6000,replace=False)
sil_base=silhouette_score(Z[bi],d.india_aqi_category.values[bi]); print('AQI category silhouette',round(sil_base,3))
cats=[c for c in ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe'] if c in d.india_aqi_category.unique()]
ct=pd.crosstab(d.india_aqi_category,d.type,normalize='index').loc[cats,ORDER]*100; print(ct.round(0))
fig,ax=plt.subplots(figsize=(9,3.4)); left=np.zeros(len(ct))
for t in ORDER: ax.barh(ct.index,ct[t],left=left,color=COL[t],label=t); left+=ct[t].values
ax.invert_yaxis(); ax.set_xlabel('% of days'); ax.legend(ncol=4,fontsize=8.5,loc='lower center',bbox_to_anchor=(.5,-.42)); ax.set_title('Same AQI category, different pollution type',loc='left'); save(fig,'baseline')

# ## 8. Comparing methods

# %%
si=np.random.RandomState(2).choice(len(Z),6000,replace=False); Zs=Z[si]; k4=km.labels_[si]; res=[]
res.append(['Baseline: AQI category',sil_base,'6 fixed bands','-'])
res.append(['K-means (k = 4)',silhouette_score(Zs,k4),' / '.join(map(str,np.bincount(km.labels_))),'1.00'])
w=AgglomerativeClustering(4,linkage='ward').fit_predict(Zs); res.append(['Hierarchical (Ward)',silhouette_score(Zs,w),' / '.join(map(str,sorted(np.bincount(w)*len(Z)//len(Zs),reverse=True))),f'{adjusted_rand_score(k4,w):.2f}'])
gm=GaussianMixture(4,random_state=RS).fit(Z).predict(Zs); res.append(['Gaussian mixture',silhouette_score(Zs,gm),' / '.join(map(str,sorted(np.bincount(gm)*len(Z)//len(Zs),reverse=True))),f'{adjusted_rand_score(k4,gm):.2f}'])
db=DBSCAN(eps=0.7,min_samples=20).fit_predict(Zs); core=db!=-1
res.append(['DBSCAN (eps 0.7, min 20)',silhouette_score(Zs[core],db[core]) if len(set(db[core]))>1 else np.nan,f'{len(set(db[core]))} groups, {(~core).mean()*100:.0f}% noise',f'{adjusted_rand_score(k4,db):.2f}'])
cmp=pd.DataFrame(res,columns=['Method','Silhouette','Group sizes','ARI vs K-means']); print(cmp.round(3).to_string())

# ## 9. Do the types make sense? Season, Diwali, wind

# %%
mt=pd.crosstab(d.date.dt.month,d.type,normalize='index')[ORDER]*100
fig,ax=plt.subplots(figsize=(9,3.6)); bot=np.zeros(12)
for t in ORDER: ax.bar(mt.index,mt[t],bottom=bot,color=COL[t],label=t,width=.8); bot+=mt[t].values
ax.set_xticks(range(1,13)); ax.set_xticklabels(MON); ax.set_ylabel('% of city-days'); ax.legend(ncol=4,fontsize=8.5,loc='lower center',bbox_to_anchor=(.5,-.3))
ax.set_title('Pollution type by month, all 50 cities',loc='left'); save(fig,'month_types'); print(mt.round(0).T)
diw=pd.to_datetime(['2022-10-24','2023-11-12','2024-10-31','2025-10-20']); north=['Delhi','Lucknow','Kanpur','Agra','Ludhiana','Amritsar','Chandigarh','Patna','Varanasi','Prayagraj','Jaipur','Gwalior']
dn=d[d.city.isin(north)]
for dt in diw:
    a=dn[dn.date.between(dt-pd.Timedelta(days=7),dt-pd.Timedelta(days=3))]; b2=dn[dn.date.between(dt+pd.Timedelta(days=1),dt+pd.Timedelta(days=3))]
    print(dt.date(),'north India smog share week before', round((a.type=='Winter smog').mean()*100), '% , 3 days after', round((b2.type=='Winter smog').mean()*100),'%')

# %%
ts=ORDER
fig,axs=plt.subplots(1,3,figsize=(13,3.6))
for ax,(col,lab) in zip(axs,[('wind_speed_10m_max','Max wind speed (km/h)'),('relative_humidity_2m_mean','Humidity (%)'),('precipitation_sum','Rain (mm)')]):
    bp=ax.boxplot([d[d.type==t][col].dropna() for t in ts],vert=False,patch_artist=True,showfliers=False)
    for p,t in zip(bp['boxes'],ts): p.set_facecolor(COL[t])
    ax.set_yticklabels(ts if ax is axs[0] else ['']*4); ax.set_xlabel(lab)
axs[0].set_title('Weather on each type of day',loc='left'); fig.tight_layout(); save(fig,'weather_types')
dl=d[(d.city=='Delhi')&(d.date.dt.month.isin([10,11,12,1]))]; lab=['N','NE','E','SE','S','SW','W','NW']
sector=lambda x: (((x+22.5)%360)//45).astype(int)
fig,axs=plt.subplots(1,2,figsize=(9,4.2),subplot_kw={'projection':'polar'})
for ax,t in zip(axs,['Winter smog','Moderate haze']):
    s=sector(dl[dl.type==t].wind_direction_10m_dominant.dropna()).value_counts(normalize=True).reindex(range(8),fill_value=0)
    ax.bar(np.deg2rad(np.arange(0,360,45)),s.values,width=np.deg2rad(40),color=COL[t]); ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0,360,45))); ax.set_xticklabels(lab); ax.set_yticklabels([]); ax.set_title(f'Delhi, Oct to Jan: {t}',fontsize=10)
save(fig,'windrose')
print('Delhi smog days wind from:', sector(dl[dl.type=='Winter smog'].wind_direction_10m_dominant.dropna()).value_counts(normalize=True).rename(index=dict(enumerate(lab))).round(2).to_dict())

# ## 10. City profiles and winter trend

# %%
cp=pd.crosstab(d.city,d.type,normalize='index')[ORDER]*100; cp=cp.sort_values(['Winter smog','Dust'])
fig,ax=plt.subplots(figsize=(8,10)); left=np.zeros(len(cp))
for t in ORDER: ax.barh(cp.index,cp[t],left=left,color=COL[t],label=t); left+=cp[t].values
ax.tick_params(axis='y',labelsize=8); ax.set_xlabel('% of days'); ax.legend(ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.08)); ax.set_title('Pollution type mix for each city',loc='left'); save(fig,'city_types')
dom=cp.idxmax(axis=1); cm=d.groupby('city')[['latitude','longitude']].first()
fig,ax=plt.subplots(figsize=(5.6,6.2)); outline(ax)
for t in ORDER:
    m=cp.index[(cp['Winter smog']>=35) if t=='Winter smog' else (cp['Dust']>=25)&(cp['Winter smog']<35) if t=='Dust' else (dom==t)&(cp['Winter smog']<35)&(cp['Dust']<25)]
    ax.scatter(cm.loc[m,'longitude'],cm.loc[m,'latitude'],s=70,color=COL[t],edgecolor=INK,lw=.4,label=t)
for c in ['Delhi','Mumbai','Chennai','Kolkata','Jodhpur','Bengaluru','Patna','Leh','Ahmedabad','Guwahati']: ax.text(cm.loc[c,'longitude']+.5,cm.loc[c,'latitude'],c,fontsize=7.5)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [sp.set_visible(False) for sp in ax.spines.values()]; ax.legend(fontsize=8,loc='lower left'); ax.set_title('Main problem per city',loc='left'); save(fig,'map_types')
d['winter']=np.where(d.date.dt.month>=11,d.date.dt.year,d.date.dt.year-1); ww=d[d.date.dt.month.isin([11,12,1])]
tr=ww[ww.city.isin(north)].groupby('winter').agg(smog=('type',lambda s:(s=='Winter smog').mean()*100),pm=('pm2_5_mean','mean'))
print('North India, Nov to Jan:'); print(tr.round(1))

# ## 11. Export for the dashboard

# %%
d['ymd']=d.date.dt.strftime('%Y-%m-%d'); d['ym2']=d.date.dt.strftime('%Y-%m'); ti={t:i for i,t in enumerate(ORDER)}
cities={}
for c,gc in d.groupby('city'):
    m=gc.groupby('ym2').agg(pm25=('pm2_5_mean','mean'),pm10=('pm10_mean','mean'),no2=('no2_mean','mean'),so2=('so2_mean','mean'),o3=('o3_8h_max','mean'),dust=('dust_mean','mean'),aqi=('india_aqi','mean'),wind=('wind_speed_10m_max','mean'),rain=('precipitation_sum','sum'))
    tc=pd.crosstab(gc.ym2,gc.type).reindex(columns=ORDER,fill_value=0)
    last=gc.sort_values('date').tail(60)
    cities[c]={'lat':float(gc.latitude.iloc[0]),'lon':float(gc.longitude.iloc[0]),'state':gc.state.iloc[0],'region':gc.region.iloc[0],
      'm':{ym:[round(float(v),1) for v in r.values]+tc.loc[ym].astype(int).tolist() for ym,r in m.iterrows()},
      'wd':{t:sector(gc[gc.type==t].wind_direction_10m_dominant.dropna()).value_counts(normalize=True).reindex(range(8),fill_value=0).round(3).tolist() for t in ORDER},
      'daily':[[r.ymd,round(r.pm2_5_mean,1),round(r.pm10_mean,1),round(r.no2_mean,1),round(r.dust_mean,1),int(r.india_aqi),ti[r.type],round(r.wind_speed_10m_max,1),int(r.wind_direction_10m_dominant) if pd.notna(r.wind_direction_10m_dominant) else None] for r in last.itertuples()]}
model={'features':FEAT,'mean':sc.mean_.round(5).tolist(),'scale':sc.scale_.round(5).tolist(),'centroids':km.cluster_centers_.round(4).tolist(),'names':[names[i] for i in range(K)]}
out={'order':ORDER,'colors':[COL[t] for t in ORDER],'mcols':['pm25','pm10','no2','so2','o3','dust','aqi','wind','rain'],'summary':summary.reset_index().rename(columns={'index':'type'}).to_dict('records'),'cities':cities,'model':model,'last_date':d.date.max().strftime('%Y-%m-%d'),'validation_r':round(float(r_all),2)}
json.dump(out,open('dash_data.json','w'),separators=(',',':')); print('dash_data.json saved,', os.path.getsize('dash_data.json')//1024,'KB')
