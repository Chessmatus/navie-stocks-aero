"""Static plots of measured and numerically reproduced turbine-wake velocity."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_experiment(directory, report):
    directory=Path(directory);a=dict(np.load(directory/'solution_primary.npz'))
    t=a['times'];points=a['points'];k=np.argmin(np.linalg.norm(points-[.060,0,0],axis=1))
    predicted=np.full_like(a['observed_points'],np.nan)
    predicted[1:10]=a['fit_prediction'];predicted[11:]=a['validation_prediction']
    figures={}
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for c,ax in enumerate(axes):
        ax.plot(t,a['observed_points'][:,k,c],'o-',color='#222222',lw=1.3,label='Измерения (VIC#)')
        ax.plot(t,predicted[:,k,c],'s--',color='#b04035',lw=1.4,label='Решение Навье–Стокса')
        ax.axvspan(0,.09,color='#bfd9ef',alpha=.3);ax.axvspan(.10,.17,color='#e5cbe8',alpha=.3)
        ax.set_ylabel(f'{"uvw"[c]}, м/с');ax.grid(alpha=.25)
    axes[0].legend(loc='best',fontsize=9)
    axes[0].set_title('Измерения и решение PDE в точке (60, 0, 0) мм\nПодбор: 0–0,09 с; контроль с новым начальным полем: 0,10–0,17 с')
    axes[-1].set_xlabel('Время от кадра 2732, с')
    figures['time_series']=fig

    zindex=np.argmin(abs(a['z']));measured=a['observed_grid'][-1,:,:,zindex,0].T
    calculated=a['validation_velocity'][-1,:,:,zindex,0].T;error=calculated-measured
    extent=[a['x'][0]*1000,a['x'][-1]*1000,a['y'][0]*1000,a['y'][-1]*1000]
    # Pixel edges coincide with the domain boundary (cell-centered fields).
    dx=(a['x'][1]-a['x'][0])*500;dy=(a['y'][1]-a['y'][0])*500
    extent=[extent[0]-dx,extent[1]+dx,extent[2]-dy,extent[3]+dy]
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),layout='constrained',sharey=True)
    vmin=min(measured.min(),calculated.min());vmax=max(measured.max(),calculated.max())
    for ax,field,title in zip(axes[:2],[measured,calculated],['Измерения (VIC#)','Решение PDE']):
        im=ax.imshow(field,origin='lower',extent=extent,cmap='viridis',vmin=vmin,vmax=vmax,aspect='equal')
        ax.set_title(title);ax.set_xlabel('x, мм')
    fig.colorbar(im,ax=list(axes[:2]),label='u, м/с',shrink=.75)
    im=axes[2].imshow(error,origin='lower',extent=extent,cmap='RdBu_r',vmin=-abs(error).max(),vmax=abs(error).max(),aspect='equal')
    axes[2].set_title('Расчёт − измерения');axes[2].set_xlabel('x, мм');axes[0].set_ylabel('y, мм')
    fig.colorbar(im,ax=axes[2],label='Разность u, м/с',shrink=.75)
    fig.suptitle(f'Контрольное окно: t = {t[-1]:.2f} с; сечение z = {a["z"][zindex]*1000:.1f} мм')
    figures['slice_comparison']=fig

    r=report['runs'][0];profile=r['profile'];nus=np.array([p['nu'] for p in profile]);j=np.array([p['J'] for p in profile])
    jref=r['fit_reference']['rmse_m_per_s']**2*(9*len(points)*3)/.4**2
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    ax.plot(nus/1e-6,j/jref,'o-',label=r'$J(\nu)/J(\nu_{ref})$')
    ax.axvline(r['nu_reference']/1e-6,ls='--',color='gray',label='Опорная вязкость')
    ax.plot(r['nu_fit']/1e-6,(r['fit_calibrated']['rmse_m_per_s']/r['fit_reference']['rmse_m_per_s'])**2,'*',ms=14,color='#b04035',label='Результат TRF (граница)')
    ax.set_xlabel(r'$\nu$, $10^{-6}$ м$^2$/с');ax.set_ylabel(r'$J(\nu)/J(\nu_{ref})$')
    ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(alpha=.25);ax.legend(fontsize=9)
    ax.set_title('Профиль ошибки: внутреннего минимума в выбранном диапазоне нет')
    figures['viscosity_profile']=fig
    for name,fig in figures.items():fig.savefig(directory/f'{name}.png',dpi=170,bbox_inches='tight')
    return figures
