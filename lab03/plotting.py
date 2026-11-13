"""Scientific figures for the executed notebook and Beamer presentation."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'tmp/lab03/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from .model import ROOT,TERMS,bits,graph

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
BLUE='#275d8c';RED='#a52c35';GREEN='#28734f'


def save(fig,name):
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=170);plt.close(fig)


def draw_tree(mask,ax,expanded=True,title=None,brief=False):
    if expanded:
        g=graph(mask);root='p_tt'
    else:
        g=nx.DiGraph();root='p_tt';g.add_node(root,label='p_tt')
        g.add_node('sum',label='+');g.add_edge('sum',root,position=0)
        short=['θ1 Fx','θ2 Fy','θ3 Fxy','−θ4 pt','−θ5 p','−θ6 p³']
        if brief:short=['Fx','Fy','Fxy','−pt','−p','−p³']
        for j in np.flatnonzero(bits(mask)):
            g.add_node(str(j),label=short[j]);g.add_edge(str(j),'sum',position=int(j))
    pos={};leaf=[0]
    def layout(n,depth):
        children=sorted(g.predecessors(n),key=lambda k:g.edges[k,n]['position'])
        if children:
            for c in children:layout(c,depth+1)
            x=np.mean([pos[c][0] for c in children])
        else:x=leaf[0];leaf[0]+=1
        pos[n]=(x,-depth)
    layout(root,0)
    labels={n:g.nodes[n]['label'] for n in g}
    labels={n:('×' if x=='*' else '−' if x=='neg' else x.replace('theta_','θ')) for n,x in labels.items()}
    colors=['#ffe3cf' if g.nodes[n].get('kind')=='parameter' else '#e4eef6' for n in g]
    nx.draw_networkx_edges(g,pos,ax=ax,arrows=True,arrowstyle='-|>',arrowsize=12,
        node_size=420 if expanded else 1000,node_shape='s',width=.9,edge_color='#65717a')
    nx.draw_networkx_nodes(g,pos,ax=ax,node_color=colors,node_size=420 if expanded else 1000,node_shape='s',edgecolors='#8092a0',linewidths=.5)
    nx.draw_networkx_labels(g,pos,labels,ax=ax,font_size=8 if expanded else 10)
    ax.set_axis_off();ax.margins(.1)
    if title:ax.set_title(title,fontsize=12)


def make_figures():
    r=json.loads((ROOT/'results/report.json').read_text())
    sample=r['evaluation']['well_test_sample']
    with np.load(ROOT/'data'/f'well_test_{sample}.npz') as d:rho=d['rho'];p=d['p'];t=d['t']
    fig,ax=plt.subplots(1,3,figsize=(11.5,3.4),layout='constrained')
    for a,z,title,cmap in zip(ax,[np.log10(rho),p[0],p[-1]],['Density log10(rho)','Initial pressure',f'Published pressure, t={t[-1]:.2f}'],['Greys','RdBu_r','RdBu_r']):
        im=a.imshow(z.T,origin='lower',extent=[-1,1,-1,1],cmap=cmap)
        a.set(xlabel='x',ylabel='y',title=title);fig.colorbar(im,ax=a,shrink=.8)
    save(fig,'well_data')
    for name in ('well','analytic'):
        mask=r[name]['selected']['mask']
        fig,ax=plt.subplots(figsize=(11.5,5.4));draw_tree(mask,ax,title=f'{name}: selected equation tree, mask {mask}')
        save(fig,f'{name}_tree')
        fig,ax=plt.subplots(figsize=(8,3.8));draw_tree(mask,ax,expanded=False,title=f'{name}: selected branches')
        save(fig,f'{name}_tree_short')
        g=graph(mask);order=list(nx.topological_sort(g));a=nx.to_numpy_array(g,nodelist=order)
        fig,ax=plt.subplots(figsize=(5.5,5));ax.imshow(a,cmap='Blues',vmin=0,vmax=1)
        ax.set(xlabel='Target node index',ylabel='Source node index',title=f'Adjacency: {name}, {len(order)} nodes')
        save(fig,f'{name}_adjacency')
        with np.load(ROOT/'results'/f'{name}_sampling.npz') as d:
            accepted=d['r0_proposal'][d['r0_accepted']]
            unique=list(dict.fromkeys(accepted.tolist()))
            choices=unique[:3]+[mask]
            fig,axes=plt.subplots(1,4,figsize=(13,3.6),layout='constrained')
            for ax,m in zip(axes,choices):draw_tree(m,ax,False,title=f'Mask {m}: {bits(m).sum()} branches',brief=True)
            save(fig,f'{name}_sequence')
        with np.load(ROOT/'results'/f'{name}_distribution_audit.npz') as d:
            expected=d['expected_pmf'];empirical=d['empirical_pmf']
        ids=np.flatnonzero(expected>0)
        fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained');x=np.arange(len(ids))
        ax.bar(x-.18,expected[ids],.36,label='Exact target',color=BLUE)
        ax.bar(x+.18,empirical[ids],.36,label='Accepted samples',color='#e5a965')
        ax.set(xticks=x,xticklabels=ids,xlabel='Graph mask',ylabel='Probability',title=f'{name}: rejection sampling distribution')
        ax.legend();save(fig,f'{name}_distribution')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name,color in [('well',BLUE),('analytic',RED)]:
        info=r[name]['rounds'];xs=np.arange(1,4)
        axes[0].plot(xs,[z['T'] for z in info],'o-',label=name,color=color)
        axes[1].plot(xs,[z['acceptance_rate']*100 for z in info],'o-',label=name,color=color)
    axes[0].set(xlabel='Round',ylabel='Temperature T',xticks=xs)
    axes[1].set(xlabel='Round',ylabel='Accepted proposals, %',xticks=xs)
    for a in axes:a.legend();a.grid(alpha=.2)
    save(fig,'sampling_rounds')
    with np.load(ROOT/'results/well_rollout.npz') as d:
        ref=d['reference'];times=d['times'];central=d['selected']
    with np.load(ROOT/'results'/f'clawpack_test{sample}_original.npz') as d:original=d['p']
    with np.load(ROOT/'results'/f'clawpack_test{sample}_selected.npz') as d:selected=d['p']
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.5),layout='constrained')
    vmax=float(np.max(abs(ref[-1])))
    for ax,z,title in zip(axes,[ref[-1],selected[-1],selected[-1]-ref[-1]],
                         ['Published test pressure','Selected graph + Clawpack','Prediction minus reference']):
        im=ax.imshow(z.T,origin='lower',extent=[-1,1,-1,1],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        ax.set(xlabel='x',ylabel='y',title=title);fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'well_approximation')
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.6),layout='constrained')
    norm=np.linalg.norm(ref.reshape(len(ref),-1),axis=1)
    for label,z,color in [('Original physics + Clawpack',original,BLUE),('Selected graph + Clawpack',selected,RED),('Selected + centered FV',central,'#8b9297')]:
        e=np.linalg.norm((z-ref).reshape(len(ref),-1),axis=1)/norm
        axes[0].plot(times,e,label=label,color=color)
    point=tuple(int(i) for i in np.unravel_index(np.argmax(np.mean(ref**2,axis=0)),ref.shape[1:]))
    for label,z,color in [('Reference',ref,'black'),('Original + Clawpack',original,BLUE),('Selected + Clawpack',selected,RED)]:
        axes[1].plot(times,z[:,point[0],point[1]],label=label,color=color)
    axes[0].set(xlabel='Time',ylabel='Relative L2 pressure error');axes[1].set(xlabel='Time',ylabel='Pressure',title=f'Test probe, grid {point}')
    for a in axes:a.legend(fontsize=8);a.grid(alpha=.2)
    save(fig,'well_error')
    with np.load(ROOT/'results/analytic_rollout.npz') as d:
        ref=d['reference'];pred=d['selected'];times=d['times']
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.5),layout='constrained')
    for ax,z,title in zip(axes,[ref[-1],pred[-1],pred[-1]-ref[-1]],['Fresh exact case, t=5','Selected solution','Numerical error']):
        im=ax.imshow(z.T,origin='lower',extent=[0,1,0,1],cmap='RdBu_r')
        ax.set(xlabel='x',ylabel='y',title=title);fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'analytic_approximation')
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    v=r['verification']['standing_wave'];ax.loglog([z['n']-1 for z in v],[z['relative_L2'] for z in v],'o-',label='Measured error',color=BLUE)
    ax.loglog([32,64,128],[v[0]['relative_L2'],v[0]['relative_L2']/4,v[0]['relative_L2']/16],'--',label='Second order',color=RED)
    ax.set(xlabel='Grid intervals per axis',ylabel='Relative L2',title='Independent solver verification');ax.legend();ax.grid(alpha=.2)
    save(fig,'convergence')
    fig,ax=plt.subplots(figsize=(9.5,3.7),layout='constrained')
    for i,name in enumerate(('well','analytic','analytic_noise')):
        selected=bits(r[name]['selected']['mask']).astype(float)
        ax.scatter(np.arange(6),np.full(6,2-i),s=250,c=np.where(selected>0,BLUE,'#eeeeee'),edgecolors='#888')
    ax.set(yticks=[2,1,0],yticklabels=['The Well','Exact CNO','CNO + 1% moment noise'],
        xticks=np.arange(6),xticklabels=TERMS,xlim=(-.6,5.6),ylim=(-.5,2.5),title='Retained branches (blue)')
    ax.tick_params(axis='x',rotation=20);save(fig,'selected_branches')
    rows=json.loads((ROOT/'results/well_catalog.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    labels=['Original structure','With damping'];ids=[3,11]
    axes[0].bar(labels,[rows[i]['validation_pressure']*100 for i in ids],color=[BLUE,RED])
    axes[0].set(ylabel='Validation pressure error, %',title='Same validation trajectory and Clawpack')
    axes[1].bar(labels,[rows[i]['J'] for i in ids],color=[BLUE,RED])
    axes[1].set(ylabel='Corrected objective J',title='Both structures are admissible')
    save(fig,'criterion_comparison')
    forward=json.loads((ROOT/'results/well_forward_validation.json').read_text())['records']
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.5),layout='constrained')
    for row,index in enumerate(r['evaluation']['well_transfer_samples']):
        with np.load(ROOT/'data'/f'well_valid_{index}.npz') as d:
            fields=[np.log10(d['rho']),d['p'][0]]
        for ax,z,title,cmap in zip(axes[row],fields,['Density log10(rho)','Initial pressure'],['Greys','RdBu_r']):
            im=ax.imshow(z.T,origin='lower',extent=[-1,1,-1,1],cmap=cmap)
            ax.set(title=f'Validation sample {index}: {title}',xlabel='x',ylabel='y')
            fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'transfer_data')
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    x=np.arange(3)
    for offset,mask,label,color in [(-.18,3,'Original structure',BLUE),(.18,11,'With damping',RED)]:
        row=forward[str(mask)]
        values=[row['pressure_error']]+[z['pressure_error'] for z in row['transfer_errors']]
        ax.bar(x+offset,np.array(values)*100,.36,label=label,color=color)
    ax.set(xticks=x,xticklabels=['Base validation / sample 0','Transfer / sample 1','Transfer / sample 2'],
        ylabel='Relative pressure L2 error, %',title='Fixed fitted coefficients; separate stored reference trajectories')
    maximum=max([forward[str(i)]['pressure_error'] for i in (3,11)]+
        [z['pressure_error'] for i in (3,11) for z in forward[str(i)]['transfer_errors']])
    ax.set_ylim(0,maximum*100*1.3)
    ax.legend();save(fig,'transfer_errors')
    print('Scientific figures generated',flush=True)


if __name__=='__main__':make_figures()
