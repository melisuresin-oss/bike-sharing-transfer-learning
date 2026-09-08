# Stage 2 and Stage 2B V2.2 scientific-definition verification

Status: **PASS**. No governing contradiction was found.

- Stage 1 binding: `8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed`; selected formulation `LOG1P / log1p_target`.
- Stage 2: 12 Graph-GRU configurations x 4 pseudo-targets x 3 seeds = 144 source fits and 144 zero-shot evaluations.
- Stage 2B: 4 genuine Vanilla-GRU configurations x 4 pseudo-targets x 3 seeds = 48 source fits and 48 zero-shot evaluations.
- Both use 12,000 source updates, 16 city-hour anchors per update, AdamW 1e-3, weight decay 1e-4, global clipping 1.0, final-update checkpoints, and no early stopping or adaptation.
- Candidate-independent data-order seeds retain the registered `stage2-v2.1` namespace as historical deterministic design provenance.
- V2.1 fitted checkpoints, predictions, metrics, and winners are excluded from V2.2 selection.
- Final-target labels are inaccessible. Stage 3 and Stage 4 remain unstarted.

Artifacts:

- Stage-2 contract: `research/results/stage2_graphgru_selection_v2_2/scientific_contract.json` (`55ebfa06ca49d77e5ae0b7b2e531d6bb14c26a2b93f478da7754c75fb1aae52f`)
- Stage-2B contract: `research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json` (`2eb1a2a14dd1412f1286248c547814a2c7e04fad94edaace2084c7d78be0cad3`)
- Stage-2 job map: `research/results/stage2_graphgru_selection_v2_2/job_map.json` (`fd34f71b46eb6f9cfc9adb5ebfd44001a2e26df049b9bb197aff1d272e7a074a`)
- Stage-2B job map: `research/results/stage2b_vanilla_fairness_v2_2/job_map.json` (`0cf72a4f66797bcc5625bd438046562c5fd444677676971e2697e1f9a3d0f23d`)
- Machine verification: `deployment/stage2_v2_2_a40/SCIENTIFIC_DEFINITION_VERIFICATION.json` (`b79d7922f4e35af92d31c7cbc8cbb97c4da822457990a41ed2317102f85b2948`)
