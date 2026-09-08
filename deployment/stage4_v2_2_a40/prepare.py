"""Build and verify in isolation before publishing any new sealed Stage-4 files."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile
from research.stage4_v2_2 import core as c,a40
from research.stage4_v2_2.verify_selection import run_checks

HISTORICAL='research/results/stage4_adversarial_method_v2_1/stage4_adversarial_method_manifest.json'
HISTORICAL_SHA='a830e155f0f4052a4b8f3a7d9492bc27877513b11cee885dc93ab6fa1cee7d92'
IMPL='research/results/causal_history_implementation_v2_2/implementation_manifest.json'
LABELS='processed/protocol_v2_2/development_labels/retrospective_label_manifest.json'


def encoded(v):return (json.dumps(c.canonicalize(v),indent=2,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()


def build(report_path):
    c.require(not c.path(c.CONTRACT).exists() and not c.path(c.JOBMAP).exists() and not c.path(c.MANIFEST).exists(),'Never replace an existing Stage-4 seal or package')
    c.require(not c.path(c.OUT).exists(),'Local scientific output directory must not exist')
    report_raw=Path(report_path).read_bytes();report=json.loads(report_raw)
    c.require(report['status']=='PASS' and report['check_count']>=45 and report['real_scientific_optimizer_updates']==report['real_scientific_evaluations']==0,'Completed synthetic implementation gate')
    for p,h in report['tested_code_sha256'].items():c.require(c.sha(p)==h,'Code changed after implementation test: '+p)
    run_checks();clar=c.read(c.CLARIFICATION);old=c.read(HISTORICAL);stage3=c.read(c.STAGE3)
    pins={**c.UPSTREAM,**c.PINNED_DATA,**clar['upstream_sha256'],HISTORICAL:HISTORICAL_SHA,
          IMPL:'a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d',
          LABELS:'1868d6a95641e43ef3ad779f8a4f779aaf94fc3cb938b3db13b77e2eb0dd3469',
          'RESEARCH_PROTOCOL_V2_2_AMENDMENT.md':'02a44e3aa8bf1d656d2f9de833b45072d564052e2ab3c1426bda2a6456c558d0'}
    pins.update(stage3['joint_upstream_closure'] and {stage3['joint_upstream_closure']['path']:stage3['joint_upstream_closure']['sha256']})
    # Historical design text/code is provenance. Never open the historical panel in its map.
    pins.update({p:h for p,h in old['governing_input_sha256'].items() if p.endswith('.md') or p in ('research/data/budget_sampler.py','research/models/common.py','research/models/graph_gru.py','research/models/heads.py','research/training/trainer.py')})
    pins.update({p:h for p,h in old['artifact_integrity'].items() if p.endswith('.md') or p=='research/governance/stage4_adversarial_method_seal.py'})
    for p,h in pins.items():c.require(c.sha(p)==h,'Preserved authoritative binding: '+p)
    checks=[]
    def gate(name,ok):
        c.require(ok,name);checks.append({'check':name,'status':'PASS'})
    gate('prospective_not_historical',not clar['historically_defined'] and clar['created_before_stage4_scientific_execution'])
    gate('eight_independent_ranking_tests',run_checks()['synthetic_tests']==8)
    gate('stage3_rebound_not_search',stage3['status']=='FROZEN_REBOUND_POLICY_ONLY')
    gate('frozen_graph_backbone',stage3['selected_graph_gru']=={'config_id':'GGRU_K04_H032_D00','graph_k':4,'hidden_size':32,'dropout':0.0})
    gate('frozen_LOG1P',stage3['selected_scale']=='LOG1P' and old['backbone']['scale_loss_formulation']=='LOG1P_TARGET_MAE')
    gate('historical_method_preserved',old['method']['domain_ce_forward_weight']==1.0 and old['method']['grl_encoder_backward_multiplier']=='-lambda_t' and not old['method']['canonical_source_target_dann_claim'])
    gate('historical_grid_preserved',old['candidate_grid']=={'configurations':6,'lambda_max':list(c.LAMBDAS),'pseudo_target_city_ids':list(c.FOLDS),'schedule':list(c.SCHEDULES),'seeds':list(c.SEEDS)})
    gate('source_budget_12000',old['lambda_schedule']['source_updates']==12000 and c.SOURCE_UPDATES==12000)
    gate('seven_day_policy_300',next(x for x in stage3['budget_schedule'] if x['budget']=='7')['updates']==300 and c.ADAPTATION_UPDATES==300)
    gate('no_early_stopping',not stage3['policy']['early_stopping'] and old['method']['adaptation']['fresh_reset_adamw'])
    gate('source_and_adaptation_optimizers',c.metadata_equal(asdict(c.optimizer_config('source')),c.read('research/results/stage2_graphgru_selection_v2_2/scientific_contract.json')['source_policy']['optimizer']) and c.optimizer_config('adaptation').learning_rate==stage3['policy']['learning_rate'])
    gate('current_fit_mask_permission','domain pooling in Stage 4 may use the permitted training target mask at C' in c.path('RESEARCH_PROTOCOL_V2_2_AMENDMENT.md').read_text(encoding='utf-8'))
    jobs=c.registered_jobs();c.validate_jobs(jobs)
    gate('72_source_72_adaptation',len(jobs)==144 and sum(j['phase']=='source' for j in jobs)==72)
    gate('seven_sources_and_target_exclusion',all(len(j['source_cities'])==7 and j['pseudo_target'] not in j['source_cities'] for j in jobs))
    gate('local_scientific_execution_zero',not c.path(c.OUT).exists() and report['real_scientific_optimizer_updates']==report['real_scientific_evaluations']==0)
    gate('final_scope_unauthorized',not report['final_target_labels_accessed'] and not report['final_experiment_started'])
    dependencies=[]
    for folder in ('research/models','research/training','research/evaluation','research/v2_2'):
        names={'research/models':['__init__.py','common.py','graph_gru.py','heads.py','vanilla_gru.py'],
               'research/training':['__init__.py','trainer.py','losses.py','reproducibility.py'],
               'research/evaluation':['__init__.py','metrics.py'],
               'research/v2_2':['__init__.py','contract.py','history.py','snapshots.py','artifacts.py','labels.py']}[folder]
        dependencies.extend(folder+'/'+n for n in names)
    dependencies+=['research/stage1_v2_2/__init__.py','research/stage1_v2_2/core.py','research/development_data_v2_2/__init__.py','research/development_data_v2_2/registry.py','research/data/budget_sampler.py']
    source_files=set(pins)|set(dependencies)|set(report['tested_code_sha256'])
    source_files.update(p.relative_to(c.ROOT).as_posix() for p in c.path('research/results/stage4_selection_clarification_v2_2').iterdir() if p.is_file())
    source_files.update('research/results/causal_history_spec_v2_2/'+n for n in ('v2_2_specification.json','causal_history_spec_seal.json','fixed_cohort_static_manifest.json'))
    source_files.update([c.PACKAGE+'prepare.py',c.PACKAGE+'README.md',c.PACKAGE+'requirements-ts9.txt'])
    for entry in c.read(c.CACHE)['cities'].values():source_files.update(e['path'] for e in entry['arrays'].values())
    source_files.update(g['path'] for g in c.read(c.GRAPH)['graphs'] if g['k']==4)
    source_files.update(j['snapshot']['path'] for j in jobs)
    source_files.update(e['path'] for e in c.read(LABELS)['labels'] if e['city_id'] in c.FOLDS)
    code_hashes={p:c.sha(p) for p in sorted(set(dependencies)|set(report['tested_code_sha256']))}
    audit={'status':'PASS','scientific_ambiguities_remaining':[],'previous_omissions_resolved_prospectively':['primary_tie_definition','fold_dispersion_definition'],
           'checks':checks,'check_count':len(checks),'upstream_sha256':pins,'implementation_gate_tests':report['check_count'],
           'execution_authorization':'UNIVERSITY_A40_AFTER_STRICT_PREFLIGHT_PASS_ONLY','default_workers':4,
           'real_local_optimizer_updates':0,'real_local_scientific_evaluations':0,'final_target_labels_accessed':False,'final_experiment_started':False,
           'downstream_blockers':c.BLOCKERS,'timestamp_utc':c.utc()}
    contract={'schema_version':'1.0','protocol_version':'2.2','stage':4,'status':'SEALED_SCIENTIFIC_AND_IMPLEMENTATION_CONTRACT',
        'method':'GRL-based multi-source source-domain-invariant pretraining; not canonical target-domain DANN',
        'backbone_id':'GGRU_K04_H032_D00','graph_k':4,'architecture':c.model_config().to_dict(),'target':'LOG1P',
        'pseudo_targets':list(c.FOLDS),'seeds':list(c.SEEDS),'lambda_max':list(c.LAMBDAS),'schedules':list(c.SCHEDULES),
        'source_domains':'exactly the other seven development cities; pseudo-target excluded from all source batches, domain labels, and reload probes',
        'source_updates':12000,'source_optimizer':asdict(c.optimizer_config('source')),'batch_size_full_city_graphs':16,'global_clip_all_parameters':1.0,
        'domain_representation':'mean final station hidden states weighted only by current training-example target-observation mask M_i(t;C) from registered sealed fitting snapshot at cutoff C=HD',
        'pooling_exclusions':['historical observation mask','raw demand as pooling weights/input','future evaluation mask','pseudo-target source rows'],
        'domain_classifier':['Linear(32,64)','ReLU','Dropout(0.1)','Linear(64,7)'],
        'loss':'LOG1P_TARGET_MAE + ordinary unit-weight unweighted seven-source-class cross entropy',
        'grl':{'forward':'identity','encoder_backward':'-lambda_t','forecast_head_direct_domain_gradient':False,'separate_alpha':False},
        'schedule':{'j':'0..11999','constant':'lambda_max','linear':'lambda_max * min(1, j / 2999)'},
        'source_sampler':'exact hash-pinned EqualCitySampler class compiled from original AST without importing legacy data readers; 12000 balanced shuffled city tokens; eligible anchors uniformly with replacement',
        'initialization':'registered seed initializes unchanged GraphGRU first, then discriminator; each immutable source fit starts at update 0',
        'data_rng':'candidate-independent, ordinary Stage-2 source schedule/anchor namespace SHA256(stage2-v2.1|label|target|seed), first 4 bytes big endian; fine-tune uses inherited stage1-v2.1 namespace; namespace text is RNG provenance, never predictor or checkpoint version',
        'adaptation':{'discriminator_discarded':True,'source_dependency':'matching validated Stage-4 V2.2 A40 source completion only','fresh_optimizer':asdict(c.optimizer_config('adaptation')),
                      'window':'[HD - 168 elapsed UTC hours, HD)','updates':300,'full_network':True,'batch_size':16,'gradient_clip':1.0,'window_extension':False,'early_stopping':False},
        'causal_predictors':'unchanged V2.2 features frozen at each original origin t; fitting cutoff C only admits targets/masks; never recertify predictors at C',
        'cache_binding':c.entry(c.CACHE),'dataset_binding':c.entry(c.DATA),'static_graph_binding':c.entry(c.GRAPH),
        'stage3':c.entry(c.STAGE3),'prospective_selection_clarification':c.entry(c.CLARIFICATION),
        'selection_fields':{k:clar[k] for k in ('primary','primary_tie','dispersion','numerical_implementation','hierarchy','excluded_selection_inputs')},
        'evaluation':{'window':'[HD,HF)','forecast_grid_hours':1440,'post_adaptation_records':72,'zero_shot_records':0,'prediction_committed_before_label_read':True,'labels':'pseudo-target development retrospective labels only'},
        'workload':{'source_fits':72,'adaptation_fits':72,'total_scientific_fits':144},
        'precision':'float32 CUDA; no AMP or compilation','execution_environment':c.ENVIRONMENT,'deterministic_runtime':a40.SETTINGS,
        'workers':4,'partial_policy':'preserve every attempt; no partial checkpoint resume; new attempt from update 0; completed jobs skipped only after hash and semantic validation',
        'completion_policy':'final budget checkpoint + exact roundtrip + phase-appropriate evaluation + immutable completion hash commitment + worker exit 0',
        'corrupt_policy':'fail closed; retain evidence; no automatic replacement','metadata':'recursive tuple/list, NumPy/native scalar, Path/string representation normalization only; no rounding, defaults, missing keys or ignored differences',
        'upstream_sha256':pins,'implementation_sha256':code_hashes,'final_experiment_authorized':False,'downstream_blockers':c.BLOCKERS,
        'timestamp_utc':c.utc()}
    jm={'schema_version':'1.0','protocol_version':'2.2','stage':4,'status':'IMMUTABLE_JOB_MAP','source_fits':72,'adaptation_fits':72,'total_fits':144,'jobs':jobs}
    auth={'status':'AUTHORIZED_AFTER_STRICT_A40_PREFLIGHT_PASS','scope':'144 Stage-4 development fits from initialization; postrun validation; frozen prospective selection; freeze decision; STOP',
          'scientific_contract_sha256':digest(encoded(contract)),'job_map_sha256':digest(encoded(jm)),'clarification_sha256':c.CLARIFICATION_SHA,
          'local_real_training_or_evaluation_authorized':False,'final_experiment_authorized':False,'workers':4,'timestamp_utc':c.utc()}
    provenance={'purpose':'MINIMAL_SELF_CONTAINED_STAGE4_V2_2_A40_PACKAGE','upstream_hashes':pins,'prior_blocked_gate_preserved':True,
       'prospective_clarification_unchanged':True,'reused_cache':'accepted immutable Stage-1 V2.2 fitting cache, not any Stage-1 fit/checkpoint',
       'legacy_data_exception':'only reconciled raw geographic k4 adjacency; k8 cache adjacency retained solely to preserve cache closure',
       'not_packaged':['historical fit checkpoints/predictions/results payloads','raw source databases','final labels','final experiment runner'],
       'resumption_state':{'ranking_tests_already_passed':8,'latest_prior_implementation_checks':111,'a40_package_binding_checks_changed_since_prior_report':True,'no_prior_contract_jobmap_zip':True},
       'same_source_code_reused':dependencies,'final_target_labels_accessed':False,'final_experiment_started':False}
    generated={c.CONTRACT:encoded(contract),c.JOBMAP:encoded(jm),c.GOV+'scientific_reaudit.json':encoded(audit),c.GOV+'implementation_test_report.json':report_raw,
               c.GOV+'execution_authorization.json':encoded(auth),c.GOV+'provenance.json':encoded(provenance)}
    buildroot=c.ROOT/'tmp'/('s4build_'+uuid.uuid4().hex[:8]);buildroot.mkdir()
    for rel in sorted(source_files):
        dest=buildroot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c.path(rel),dest)
    for rel,raw in generated.items():
        dest=buildroot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    hashes={p:c.sha(p) for p in sorted(source_files)};hashes.update({p:digest(raw) for p,raw in generated.items()})
    manifest={'schema_version':'1.0','protocol_version':'2.2','stage':4,'execution_environment':c.ENVIRONMENT,'default_workers':4,
              'scientific_contract_sha256':digest(generated[c.CONTRACT]),'job_map_sha256':digest(generated[c.JOBMAP]),
              'implementation_gate_passed':True,'implementation_report_sha256':digest(report_raw),'final_experiment_authorized':False,'files':hashes}
    mraw=encoded(manifest);mh=digest(mraw);(buildroot/c.MANIFEST).write_bytes(mraw)
    env=os.environ.copy();env['PYTHONPATH']=str(c.ROOT/'tmp/neural_build/pydeps');env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    evidence=[]
    def verify_at(root,label):
        for mode in ('package-check','validate-existing','preflight'):
            cmd=[sys.executable,'-X','utf8','-B','-m','research.stage4_v2_2.a40',mode,'--manifest-sha256',mh]
            r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
            if mode=='preflight':c.require(r.returncode!=0 and 'Required USERPROFILE' in r.stderr,'Local A40 gate must fail before any scientific action')
            else:c.require(r.returncode==0,label+' '+mode+'\n'+r.stderr)
            evidence.append({'root':str(root),'gate':mode,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'expected_local_A40_rejection':mode=='preflight'})
        probe="import json,sys; from pathlib import Path; import research.stage4_v2_2.controller; root=Path.cwd(); mods={k:str(Path(v.__file__).resolve()) for k,v in sys.modules.items() if k.startswith('research.') and getattr(v,'__file__',None)}; assert all(root in Path(v).parents for v in mods.values()); print(json.dumps(mods,indent=2))"
        r=subprocess.run([sys.executable,'-X','utf8','-B','-c',probe],cwd=root,env=env,capture_output=True,text=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        c.require(r.returncode==0,'No live-tree module fallback: '+r.stderr);evidence.append({'gate':label+'_isolated_import_closure','modules':json.loads(r.stdout)})
        c.require(not (root/c.OUT).exists(),'No scientific outputs from local package verification')
    verify_at(buildroot,'staging')
    zipname='stage4_v2_2_a40_'+mh[:12]+'.zip';zippath=c.path(c.PACKAGE+zipname)
    tempzip=buildroot/zipname
    with zipfile.ZipFile(tempzip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in sorted([*hashes,c.MANIFEST]):z.write(buildroot/rel,rel)
    extract=c.ROOT/'tmp'/('s4extract_'+uuid.uuid4().hex[:8]);extract.mkdir()
    with zipfile.ZipFile(tempzip) as z:
        c.require(set(z.namelist())==set(hashes)|{c.MANIFEST},'ZIP exact member set')
        for rel,h in {**hashes,c.MANIFEST:mh}.items():c.require(digest(z.read(rel))==h,'ZIP payload SHA256: '+rel)
        z.extractall(extract)
    verify_at(extract,'clean_zip_extraction')
    # Publish immutable artifacts only now, after the actual ZIP passed isolated checks.
    for rel,raw in {**generated,c.MANIFEST:mraw}.items():
        p=c.path(rel);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    with zippath.open('xb') as f:
        with tempzip.open('rb') as source:shutil.copyfileobj(source,f)
    result={'status':'PASS','zip':c.entry(c.PACKAGE+zipname),'manifest':c.entry(c.MANIFEST),'scientific_contract':c.entry(c.CONTRACT),'job_map':c.entry(c.JOBMAP),
            'clarification':c.entry(c.CLARIFICATION),'scientific_audit_checks':len(checks),'implementation_checks':report['check_count'],'package_member_count':len(hashes)+1,
            'uncompressed_payload_bytes':sum((buildroot/p).stat().st_size for p in [*hashes,c.MANIFEST]),'isolated_verifications':evidence,
            'real_local_optimizer_updates':0,'real_local_scientific_evaluations':0,'final_target_labels_accessed':False,'final_experiment_started':False,
            'A40_preflight_status':'NOT_RUN_ON_A40; strict local rejection verified','timestamp_utc':c.utc()}
    for rel,raw in {c.GOV+'package_validation_report.json':encoded(result),c.PACKAGE+'DELIVERY_SHA256.json':encoded({k:result[k] for k in ('zip','manifest','scientific_contract','job_map','clarification')})}.items():
        with c.path(rel).open('xb') as f:f.write(raw)
    print(json.dumps({k:result[k] for k in ('status','zip','manifest','scientific_contract','job_map','implementation_checks','package_member_count')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test-report',required=True);build(p.parse_args().test_report)
