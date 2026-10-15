"""Generate figures/table/report from saved matched NPT runs; no simulations.
Run: .venv/Scripts/python.exe src/prepare_matched_npt_report.py
"""
from pathlib import Path
import csv, hashlib, html, json, subprocess, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/waterbox_npt_preflight_zbl_bonded_ext70_matched_configs/npt'
OUT = ROOT/'results/advisor_matched_npt_2026_10_09'
CONDITIONS = ('water_absolute', 'water_absolute+momentum')
SEEDS = (0, 3, 5)
EVENTS = [('first_hoh_angle_below_80_deg','hoh_angle_min_deg',80,'H–O–H < 80°'),('first_intramolecular_hh_below_1_2a','intramolecular_hh_min_a',1.2,'intra H–H < 1.2 Å'),('first_intramolecular_hh_below_1_0a','intramolecular_hh_min_a',1.,'intra H–H < 1.0 Å'),('first_intermolecular_oo_below_2_3a','intermolecular_oo_min_a',2.3,'inter O–O < 2.3 Å')]
def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    provenance={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'command':'.venv/Scripts/python.exe src/prepare_matched_npt_report.py','python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'script_sha256':digest(Path(__file__)),'inputs_sha256':{},'pressure_formula':'Ppotential = Ptotal - 2 Ekin / (3 V); eV/A^3 to bar = 1602176.6208','gate_evidence':'Passes reported in handoff; six checkpoint-specific gate logs not found locally.'}
    audit_path=ROOT/'results/waterbox_property_diagnostics_zbl_bonded_ext70/topology_audit/topology_audit.csv'
    for name in ['paper/main.tex','paper/literature_review_candidates.md','paper/advisor_matched_npt_report_template.html','src/run_waterbox_ensemble.py','src/verify_waterbox_stress.py','src/analyze_waterbox_property_trajectory.py','results/waterbox_study_zbl_bonded_ext70/summary_table.md']:
        provenance['inputs_sha256'][name]=digest(ROOT/name)
    audit=read_csv(audit_path)
    provenance['inputs_sha256'][str(audit_path.relative_to(ROOT))]=digest(audit_path)
    runs,table,geometry={},[],None
    for seed in SEEDS:
        selected=next(r for r in audit if int(r['train_seed'])==seed and int(r['raw_dataset_index'])==1175)
        assert selected['topology_clean']=='1'
        for condition in CONDITIONS:
            folder=SOURCE/condition/f'seed{seed}'/'vseed0'
            for name in ['run_manifest.json','thermodynamic_history.csv','structural_history.csv','structural_summary.json','equilibration.xyz','production.xyz']:
                p=folder/name
                provenance['inputs_sha256'][str(p.relative_to(ROOT))]=digest(p)
            manifest=json.loads((folder/'run_manifest.json').read_text())
            summary=json.loads((folder/'structural_summary.json').read_text())
            rows=read_csv(folder/'thermodynamic_history.csv')
            structure=read_csv(folder/'structural_history.csv')
            assert manifest['status']=='complete' and manifest['data_seed']==seed
            assert manifest['test_config_index']==int(selected['test_config_index'])
            expected={'velocity_seed':0,'dt_fs':.1,'temperature_k':300.,'pressure_bar':1.,'tdamp_fs':100.,'pdamp_fs':1000.,'stress_mode':'hydrostatic_fd','stress_fd_epsilon':.003,'steps_completed':200,'equilibration_ps_requested':.01,'production_ps_requested':.01}
            assert all(manifest[k]==v for k,v in expected.items())
            assert summary['topology_source']=='dataset_sample_0_matching_training'
            a={k:np.array([float(r[k]) for r in rows]) for k in ['time_fs','temperature_k','epot_ev','ekin_ev','volume_a3','pressure_bar']}
            assert len(rows)==len(structure)==21 and np.array_equal(a['time_fs'],np.arange(21))
            assert all(np.isfinite(v).all() for v in a.values())
            assert np.array_equal(a['time_fs'],[float(r['time_fs']) for r in structure])
            lines=(folder/'equilibration.xyz').read_text().splitlines()
            atoms=[r.split() for r in lines[2:2+int(lines[0])]]
            symbols=[r[0] for r in atoms]; pos=np.array([[float(v) for v in r[1:4]] for r in atoms])
            if geometry is None: geometry=symbols,pos
            assert symbols==geometry[0] and np.allclose(pos,geometry[1],rtol=0,atol=1e-8)
            a['delta_epot_ev']=a['epot_ev']-a['epot_ev'][0]
            a['potential_pressure_bar']=a['pressure_bar']-2*a['ekin_ev']/(3*a['volume_a3'])*1602176.6208
            for k,v in [('final_temperature_k',a['temperature_k'][-1]),('maximum_temperature_k',a['temperature_k'].max()),('potential_energy_change_ev',a['delta_epot_ev'][-1])]:
                assert np.isclose(summary['thermodynamics'][k],v,rtol=0,atol=1e-8)
            events=[]
            for k,metric,cutoff,label in EVENTS:
                first=next((r for r in structure if float(r[metric])<cutoff),None)
                assert (first is None)==(summary[k] is None)
                if first is not None:
                    assert float(first['time_fs'])==summary[k]['time_fs']
                    events.append((float(first['time_fs']),label))
            event=min(events) if events else None
            table.append({'training_seed':seed,'condition':condition,'initial_potential_pressure_bar':float(a['potential_pressure_bar'][0]),'initial_total_pressure_bar':float(a['pressure_bar'][0]),'final_temperature_k':float(a['temperature_k'][-1]),'maximum_temperature_k':float(a['temperature_k'].max()),'delta_potential_energy_ev':float(a['delta_epot_ev'][-1]),'delta_volume_a3':float(a['volume_a3'][-1]-a['volume_a3'][0]),'delta_volume_percent':float(100*(a['volume_a3'][-1]/a['volume_a3'][0]-1)),'first_structural_event':event[1] if event else 'None observed through 20 fs','first_structural_event_time_fs':event[0] if event else '', 'raw_dataset_index':1175,'test_config_index':manifest['test_config_index'],'source_directory':str(folder.relative_to(ROOT))})
            runs[seed,condition]=a
    provenance['checks']=['Shared manifest protocol verified for all six runs','Raw configuration 1175 held out and topology-clean for each seed','Initial coordinates identical to 1e-8 A','Twenty-one finite rows on common 0–20 fs grid','Thermodynamic summaries match histories','Four structural event records match histories']
    with (OUT/'presentation_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    metrics=[('temperature_k','Temperature (K)','temperature',300),('delta_epot_ev','Potential-energy change (eV / box)','potential_energy_change',0),('pressure_bar','Total pressure (kbar)','pressure',.001)]
    def plot(ax,seed,key,ylabel,reference):
        for condition,label,color,style in zip(CONDITIONS,['Absolute','Absolute + molecular regularizer'],['#0072B2','#D55E00'],['-','--']):
            a=runs[seed,condition]; ax.plot(a['time_fs'],a[key]/1000 if key=='pressure_bar' else a[key],color=color,ls=style,lw=2,label=label)
        ax.axhline(reference,color='#666666',ls=':',lw=.9); ax.set_xlim(0,20); ax.set_xticks([0,5,10,15,20]); ax.grid(alpha=.18)
    def save(fig,name):
        fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=2,frameon=False)
        for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white')
        plt.close(fig)
    for key,ylabel,name,reference in metrics:
        fig,axes=plt.subplots(1,3,figsize=(11.6,3.8),sharex=True,sharey=True,layout='constrained')
        for ax,seed in zip(axes,SEEDS): plot(ax,seed,key,ylabel,reference); ax.set(title=f'Training seed {seed}',xlabel='Time (fs)')
        axes[0].set_ylabel(ylabel); save(fig,name)
    fig,axes=plt.subplots(3,3,figsize=(11.6,9),sharex=True,sharey='row',layout='constrained')
    for i,(key,ylabel,name,reference) in enumerate(metrics):
        for j,seed in enumerate(SEEDS):
            plot(axes[i,j],seed,key,ylabel,reference)
            if i==0: axes[i,j].set_title(f'Training seed {seed}')
            if i==2: axes[i,j].set_xlabel('Time (fs)')
        axes[i,0].set_ylabel(ylabel)
    save(fig,'comparison_overview')
    cells=[]
    for r in table:
        event=r['first_structural_event']+(f" at {r['first_structural_event_time_fs']:g} fs" if r['first_structural_event_time_fs']!='' else '')
        vals=[r['training_seed'],'Absolute' if r['condition']==CONDITIONS[0] else '+ regularizer',f"{r['initial_potential_pressure_bar']/1000:.2f}",f"{r['final_temperature_k']:.1f}",f"{r['maximum_temperature_k']:.1f}",f"{r['delta_potential_energy_ev']:.3f}",f"{r['delta_volume_a3']:.3f} ({r['delta_volume_percent']:.3f}%)",event]
        cells.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>')
    fig, ax = plt.subplots(figsize=(13.5, 3.5), layout='constrained')
    ax.axis('off')
    compact=[]
    for r in table:
        compact.append([r['training_seed'], 'Absolute' if r['condition']==CONDITIONS[0] else '+ regularizer', f"{r['initial_potential_pressure_bar']/1000:.2f}",f"{r['final_temperature_k']:.1f}",f"{r['maximum_temperature_k']:.1f}",f"{r['delta_potential_energy_ev']:.3f}",f"{r['delta_volume_a3']:.3f}", 'H–H < 1.2 Å at 16 fs' if r['first_structural_event_time_fs']!='' else 'None through 20 fs'])
    tab=ax.table(cellText=compact,colLabels=['Seed','Condition','Initial potential\npressure (kbar)','Final T\n(K)','Max T\n(K)','ΔEpot\n(eV / box)','ΔV\n(Å³)','First structural threshold'],colWidths=[.05,.13,.15,.09,.09,.1,.09,.3],cellLoc='center',loc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1,1.8)
    for (i,j), cell in tab.get_celld().items():
        cell.set_edgecolor('#ccd4db'); cell.set_linewidth(.5)
        if i==0: cell.set_facecolor('#edf2f5'); cell.set_text_props(weight='bold'); cell.set_height(.20)
    ax.set_title('Matched 20 fs NPT preflight · configuration 1175',pad=10,fontsize=14)
    for ext in ['png','pdf','svg']: fig.savefig(OUT/f'presentation_table.{ext}',dpi=400,facecolor='white')
    plt.close(fig)
    report=(ROOT/'paper/advisor_matched_npt_report_template.html').read_text(encoding='utf-8-sig')
    (OUT/'advisor_report.html').write_text(report.replace('TABLE_ROWS',''.join(cells)),encoding='utf-8')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(f'Prepared {len(table)} verified rows and four comparison figures plus table in PNG/PDF/SVG: {OUT}')
if __name__=='__main__': main()
