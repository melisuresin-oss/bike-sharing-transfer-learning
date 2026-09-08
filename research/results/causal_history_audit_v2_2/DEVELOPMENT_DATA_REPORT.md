# V2.2 development data gate report

PASS: 960 read-only verification checks; all 18 full-development data checks pass.

Scope: eight development cities, 523 fixed stations, 5,537 origins per city, [H0,HF). The global cohort remains exactly 799. Count availability is IDEALIZED_COUNT_FEED_BENCHMARK, Q=1[e<=a]. No final-target outcome construction occurred.

Full history validation checked 72,396,275 station-lag cells, with zero future certificates and zero boundary violations. No exact-12h equality occurred in this raw development build; exact equality and 12h+1us rejection are established by the hash-bound accepted synthetic implementation tests. Full-data checks use exact microsecond comparisons.

All 24 geographic adjacency arrays (eight cities x k=4,8,16) are byte-identical to the reconciled frozen-coordinate calculation and reused in place. No k selection or new adjacency copies occurred.

Lag 1 is 100% unknown in every development city. Persistence is unavailable on every pseudo-target evaluation key. These descriptive findings do not amend the protocol or introduce a fallback.

## Lag availability

Percent observed across all feature cells in [H0,HF), including H0 history padding. Equal-city is the arithmetic mean of eight city percentages. The accompanying lag_availability.csv and JSON contain all total/positive/zero/unknown counts and shares (denominator = all cells).

| Lag | Dortmund | Heidelberg | Marburg | Giessen | Cardiff | Bilbao | Freiburg | Goeteborg | Equal-city |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 2 | 39.2125% | 50.0913% | 36.5999% | 46.0254% | 40.2755% | 68.2311% | 46.2487% | 35.0127% | 45.2121% |
| 3 | 53.2159% | 65.1013% | 49.8763% | 59.5481% | 55.3556% | 75.3922% | 61.8877% | 48.7154% | 58.6366% |
| 4 | 61.1558% | 73.0336% | 57.1702% | 66.7845% | 63.7373% | 80.7359% | 70.1745% | 56.4802% | 66.1590% |
| 5 | 66.7003% | 78.2365% | 62.0931% | 71.7995% | 69.4761% | 85.1905% | 75.6819% | 61.9559% | 71.3917% |
| 6 | 70.8749% | 81.9866% | 65.7971% | 75.6824% | 73.8014% | 88.9874% | 79.6875% | 66.2941% | 75.3889% |
| 7 | 74.1352% | 84.7755% | 68.6641% | 78.7404% | 77.2225% | 92.1858% | 82.7009% | 69.9165% | 78.5426% |
| 8 | 76.7478% | 86.8725% | 70.9443% | 81.2128% | 79.9434% | 94.7772% | 84.9969% | 72.9580% | 81.0566% |
| 9 | 78.8623% | 88.4014% | 72.7453% | 83.2484% | 82.1225% | 96.8541% | 86.7628% | 75.5056% | 83.0628% |
| 10 | 80.5562% | 89.5573% | 74.1439% | 84.9512% | 83.8721% | 97.3888% | 88.0856% | 77.5919% | 84.5184% |
| 11 | 81.9468% | 90.4357% | 75.2326% | 86.3296% | 85.2747% | 97.5656% | 89.0887% | 79.2485% | 85.6403% |
| 12 | 83.0970% | 91.1128% | 76.0907% | 87.4590% | 86.3956% | 97.6354% | 89.8466% | 80.5486% | 86.5232% |
| 13 | 84.0665% | 91.6519% | 76.7837% | 88.3950% | 87.3165% | 97.6790% | 90.4367% | 81.5631% | 87.2366% |
| 14 | 84.0461% | 91.6331% | 76.7643% | 88.3782% | 87.2944% | 97.6610% | 90.4139% | 81.5307% | 87.2152% |
| 15 | 84.0254% | 91.6135% | 76.7459% | 88.3614% | 87.2712% | 97.6429% | 90.3909% | 81.4918% | 87.1929% |
| 16 | 84.0035% | 91.5931% | 76.7274% | 88.3446% | 87.2554% | 97.6249% | 90.3672% | 81.4488% | 87.1706% |
| 17 | 83.9814% | 91.5701% | 76.7081% | 88.3272% | 87.2393% | 97.6068% | 90.3421% | 81.3937% | 87.1461% |
| 18 | 83.9591% | 91.5447% | 76.6875% | 88.3021% | 87.2232% | 97.5887% | 90.3251% | 81.3223% | 87.1191% |
| 19 | 83.9451% | 91.5278% | 76.6669% | 88.2834% | 87.2071% | 97.5707% | 90.3085% | 81.2409% | 87.0938% |
| 20 | 83.9306% | 91.5109% | 76.6455% | 88.2550% | 87.1910% | 97.5526% | 90.2914% | 81.1707% | 87.0685% |
| 21 | 83.9161% | 91.4863% | 76.6241% | 88.2337% | 87.1747% | 97.5346% | 90.2746% | 81.0901% | 87.0418% |
| 22 | 83.9011% | 91.4690% | 76.6022% | 88.2124% | 87.1586% | 97.5165% | 90.2578% | 81.0285% | 87.0183% |
| 23 | 83.8866% | 91.4517% | 76.5808% | 88.1892% | 87.1418% | 97.4984% | 90.2410% | 80.9911% | 86.9976% |
| 24 | 83.8719% | 91.4190% | 76.5598% | 88.1660% | 87.1257% | 97.4804% | 90.2244% | 80.9585% | 86.9757% |
| 168 | 81.5248% | 88.8137% | 74.0049% | 85.6349% | 84.8756% | 94.8801% | 87.7748% | 78.2850% | 84.4742% |

## Source fitting rows at HD

| City | V2.2 | V2.1 | Difference | Difference % |
|---|---:|---:|---:|---:|
| Dortmund 129 | 259,662 | 259,856 | -194 | -0.074657% |
| Heidelberg 194 | 174,959 | 175,055 | -96 | -0.054840% |
| Marburg 438 | 127,427 | 127,529 | -102 | -0.079982% |
| Giessen 467 | 101,550 | 101,627 | -77 | -0.075767% |
| Cardiff 476 | 259,371 | 259,531 | -160 | -0.061650% |
| Bilbao 532 | 172,035 | 172,083 | -48 | -0.027894% |
| Freiburg 619 | 320,516 | 320,710 | -194 | -0.060491% |
| Goeteborg 658 | 415,062 | 414,337 | +725 | +0.174978% |

The as-of city denominator is recalculated from prefix lifetime, so row-count changes need not be uniformly negative. No elapsed fitting window was extended.

## Pseudo-target adaptation rows

| City | Budget | V2.2 | V2.1 | Difference | Difference % |
|---|---|---:|---:|---:|---:|
| Cardiff | zero | 0 | 0 | +0 | N/A (zero denominator) |
| Cardiff | 1 | 1,518 | 1,643 | -125 | -7.608034% |
| Cardiff | 7 | 11,041 | 11,168 | -127 | -1.137178% |
| Cardiff | 30 | 47,393 | 47,526 | -133 | -0.279847% |
| Cardiff | full | 259,371 | 259,531 | -160 | -0.061650% |
| Bilbao | zero | 0 | 0 | +0 | N/A (zero denominator) |
| Bilbao | 1 | 963 | 1,008 | -45 | -4.464286% |
| Bilbao | 7 | 7,011 | 7,056 | -45 | -0.637755% |
| Bilbao | 30 | 30,316 | 30,362 | -46 | -0.151505% |
| Bilbao | full | 172,035 | 172,083 | -48 | -0.027894% |
| Freiburg | zero | 0 | 0 | +0 | N/A (zero denominator) |
| Freiburg | 1 | 1,739 | 1,908 | -169 | -8.857442% |
| Freiburg | 7 | 13,291 | 13,460 | -169 | -1.255572% |
| Freiburg | 30 | 57,479 | 57,651 | -172 | -0.298347% |
| Freiburg | full | 320,516 | 320,710 | -194 | -0.060491% |
| Goeteborg | zero | 0 | 0 | +0 | N/A (zero denominator) |
| Goeteborg | 1 | 2,470 | 2,783 | -313 | -11.246856% |
| Goeteborg | 7 | 17,856 | 18,171 | -315 | -1.733531% |
| Goeteborg | 30 | 74,495 | 74,699 | -204 | -0.273096% |
| Goeteborg | full | 415,062 | 414,337 | +725 | +0.174978% |

## Seven-source folds

| Excluded pseudo-target | V2.2 source rows | V2.1 rows | Difference | Source-city distribution |
|---|---:|---:|---:|---|
| Cardiff | 1,571,211 | 1,571,197 | +14 | 129: 259,662, 194: 174,959, 438: 127,427, 467: 101,550, 532: 172,035, 619: 320,516, 658: 415,062 |
| Bilbao | 1,658,547 | 1,658,645 | -98 | 129: 259,662, 194: 174,959, 438: 127,427, 467: 101,550, 476: 259,371, 619: 320,516, 658: 415,062 |
| Freiburg | 1,510,066 | 1,510,018 | +48 | 129: 259,662, 194: 174,959, 438: 127,427, 467: 101,550, 476: 259,371, 532: 172,035, 658: 415,062 |
| Goeteborg | 1,415,520 | 1,416,391 | -871 | 129: 259,662, 194: 174,959, 438: 127,427, 467: 101,550, 476: 259,371, 532: 172,035, 619: 320,516 |

## Baseline availability only

Period [HD,HF). Both all immutable prediction keys and retrospectively observed evaluation keys are reported. No prediction values or forecasting-performance metrics were computed.

| City | Baseline | Key domain | Available | Unavailable | Coverage |
|---|---|---|---:|---:|---:|
| Cardiff | PERSISTENCE_T_MINUS_1 | all_prediction_keys | 0 | 105,120 | 0.000000% |
| Cardiff | PERSISTENCE_T_MINUS_1 | retrospectively_observed_evaluation_keys | 0 | 94,138 | 0.000000% |
| Cardiff | SEASONAL_NAIVE_T_MINUS_168 | all_prediction_keys | 94,702 | 10,418 | 90.089422% |
| Cardiff | SEASONAL_NAIVE_T_MINUS_168 | retrospectively_observed_evaluation_keys | 87,453 | 6,685 | 92.898723% |
| Bilbao | PERSISTENCE_T_MINUS_1 | all_prediction_keys | 0 | 61,920 | 0.000000% |
| Bilbao | PERSISTENCE_T_MINUS_1 | retrospectively_observed_evaluation_keys | 0 | 60,960 | 0.000000% |
| Bilbao | SEASONAL_NAIVE_T_MINUS_168 | all_prediction_keys | 60,877 | 1,043 | 98.315568% |
| Bilbao | SEASONAL_NAIVE_T_MINUS_168 | retrospectively_observed_evaluation_keys | 60,617 | 343 | 99.437336% |
| Freiburg | PERSISTENCE_T_MINUS_1 | all_prediction_keys | 0 | 125,280 | 0.000000% |
| Freiburg | PERSISTENCE_T_MINUS_1 | retrospectively_observed_evaluation_keys | 0 | 115,830 | 0.000000% |
| Freiburg | SEASONAL_NAIVE_T_MINUS_168 | all_prediction_keys | 115,603 | 9,677 | 92.275702% |
| Freiburg | SEASONAL_NAIVE_T_MINUS_168 | retrospectively_observed_evaluation_keys | 110,020 | 5,810 | 94.984028% |
| Goeteborg | PERSISTENCE_T_MINUS_1 | all_prediction_keys | 0 | 181,440 | 0.000000% |
| Goeteborg | PERSISTENCE_T_MINUS_1 | retrospectively_observed_evaluation_keys | 0 | 149,327 | 0.000000% |
| Goeteborg | SEASONAL_NAIVE_T_MINUS_168 | all_prediction_keys | 148,959 | 32,481 | 82.098214% |
| Goeteborg | SEASONAL_NAIVE_T_MINUS_168 | retrospectively_observed_evaluation_keys | 133,907 | 15,420 | 89.673669% |

## Historical-average fitting availability

Only fitting keys and support counts are constructed; no lookup mean is fitted. One permitted observation suffices. Every seven-source pool supports all 168 local hour-of-week bins and the global fallback, so HA_SOURCE and the complete registered HA_TARGET hierarchy are defined for every development prediction key. Some earlier target-specific fallback levels remain unavailable, as recorded below. Equal-station/equal-city aggregation and the registered hierarchy are unchanged.

| Target | Budget | Fitting rows | Missing station x local-HOW cells | Stations without overall support | Missing target-city bins |
|---|---|---:|---:|---:|---:|
| Cardiff | 1 | 1,518 | 10,746 | 0 | 145 |
| Cardiff | 7 | 11,041 | 1,223 | 0 | 1 |
| Cardiff | 30 | 47,393 | 107 | 0 | 0 |
| Cardiff | full | 259,371 | 0 | 0 | 0 |
| Bilbao | 1 | 963 | 6,261 | 1 | 145 |
| Bilbao | 7 | 7,011 | 213 | 1 | 1 |
| Bilbao | 30 | 30,316 | 0 | 0 | 0 |
| Bilbao | full | 172,035 | 0 | 0 | 0 |
| Freiburg | 1 | 1,739 | 12,877 | 4 | 145 |
| Freiburg | 7 | 13,291 | 1,325 | 2 | 1 |
| Freiburg | 30 | 57,479 | 432 | 2 | 0 |
| Freiburg | full | 320,516 | 169 | 1 | 0 |
| Goeteborg | 1 | 2,470 | 18,698 | 7 | 145 |
| Goeteborg | 7 | 17,856 | 3,312 | 7 | 1 |
| Goeteborg | 30 | 74,495 | 1,743 | 7 | 0 |
| Goeteborg | full | 415,062 | 948 | 2 | 0 |

## Effective-pass preview

Effective passes = frozen updates x 16 / eligible city-hour anchors, sampled with replacement. Station-hour label counts are not sampler units. The 7d row applies to Stage 1 and Stage 3; other rows preview the frozen Stage-3 policy. More than 10 repeated passes is marked descriptively, without changing updates.

| Target | Budget | Eligible anchors | Updates | Anchor draws | Effective passes |
|---|---|---:|---:|---:|---:|
| Cardiff | 1 | 23 | 100 | 1,600 | 69.565217 |
| Cardiff | 7 | 167 | 300 | 4,800 | 28.742515 |
| Cardiff | 30 | 719 | 600 | 9,600 | 13.351878 |
| Cardiff | full | 4,023 | 1200 | 19,200 | 4.772558 |
| Bilbao | 1 | 23 | 100 | 1,600 | 69.565217 |
| Bilbao | 7 | 167 | 300 | 4,800 | 28.742515 |
| Bilbao | 30 | 719 | 600 | 9,600 | 13.351878 |
| Bilbao | full | 4,024 | 1200 | 19,200 | 4.771372 |
| Freiburg | 1 | 23 | 100 | 1,600 | 69.565217 |
| Freiburg | 7 | 167 | 300 | 4,800 | 28.742515 |
| Freiburg | 30 | 719 | 600 | 9,600 | 13.351878 |
| Freiburg | full | 4,024 | 1200 | 19,200 | 4.771372 |
| Goeteborg | 1 | 23 | 100 | 1,600 | 69.565217 |
| Goeteborg | 7 | 167 | 300 | 4,800 | 28.742515 |
| Goeteborg | 30 | 705 | 600 | 9,600 | 13.617021 |
| Goeteborg | full | 3,897 | 1200 | 19,200 | 4.926867 |

## Preservation and authorization

All 1,182 previously inventoried historical files are byte-preserved; accepted V2.2 implementation file bindings also match. Final evaluation label payloads were never opened, modified or newly hashed. Their existing provenance records remain preserved.

NO_MODEL_SELECTION_OCCURRED; NO_MODEL_TRAINING_OCCURRED; NO_FORECASTING_METRICS_COMPUTED; NO_SCIENTIFIC_PREDICTIONS_GENERATED. No HA lookup values were fitted. No final-target adaptation labels were constructed. No Stage 1/2/2B or Stage 4 operation occurred.

There is no remaining listed data-gate failure. Stage 1 requires separate authorization.

## Exact artifact paths and SHA256 hashes

Complete machine-readable index: artifact_index.json; SHA256 a0c3d179e1c9136612f304e08687215b7e87a5504780275aab62c5e315ec640b

| Artifact | SHA256 |
|---|---|
| [processed/protocol_v2_2/development/city_129_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_129_predictors.npz) | ee60c48205f29f5d1c2ab8fba106e673a9dcecd6a765612191d1edee3046a17b |
| [processed/protocol_v2_2/development/city_194_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_194_predictors.npz) | 823f3c5d07a816b125e82bbd74a922bd28d0281b0869e2ef15e9f3db24734e3c |
| [processed/protocol_v2_2/development/city_438_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_438_predictors.npz) | be7279a5e72d06b75b842d79b426a285bff5334da0b4520d482546034c1abdc3 |
| [processed/protocol_v2_2/development/city_467_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_467_predictors.npz) | c6e02eeba63d9fc3eb2fd841f5554c77ebb95ae918d4ac73f9a39e62715a7363 |
| [processed/protocol_v2_2/development/city_476_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_476_predictors.npz) | 8bc84d1357d8eb99e287ebedcc21edb45749091f6cc4fe056e76f4c527babf31 |
| [processed/protocol_v2_2/development/city_532_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_532_predictors.npz) | 87ded7633788ab9c217980abcd4a3b3810e8a1739109dd892a3ad84c93f0e625 |
| [processed/protocol_v2_2/development/city_619_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_619_predictors.npz) | 635a39151e857bdf586b967d86ccf304bfee4bd2dfcf822ba6b70d3460d732a5 |
| [processed/protocol_v2_2/development/city_658_predictors.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development/city_658_predictors.npz) | 4c4cfee8d688fee0c4d220f5cb5fba64968fbced558a2fa43cf0a14a223ccbc3 |
| [processed/protocol_v2_2/development_labels/city_129_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_129_retrospective.npz) | bc866c46488dfcddd1c7b4259bc65295e228f2c302092c6975d04d5d17ec09dc |
| [processed/protocol_v2_2/development_labels/city_194_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_194_retrospective.npz) | a6c81c9b2dc07d2e334d78274f89f2aa2ed3070b06480ca7b0edec1cbebc99b9 |
| [processed/protocol_v2_2/development_labels/city_438_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_438_retrospective.npz) | db96a0a15219fbd6eea2bb39d46a2436dcb217e15d2eabd3fe004d5326436f44 |
| [processed/protocol_v2_2/development_labels/city_467_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_467_retrospective.npz) | 8d7dd7072995ceb579d70d22c9a55d8c9a2507e9706eb06a17acf6d2a16a34a9 |
| [processed/protocol_v2_2/development_labels/city_476_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_476_retrospective.npz) | 215a495769a9fb5269560c9ef385492d04b4019f0ad292c13e312b8d3f142d4f |
| [processed/protocol_v2_2/development_labels/city_532_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_532_retrospective.npz) | 44a4ff791341157e694f14b903cd628c6790ae4c32f8436bea31e4621a8afe5e |
| [processed/protocol_v2_2/development_labels/city_619_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_619_retrospective.npz) | d1978729ea2ed3a2b53f6cb59b91eea1f08863176e5f65a66aea89f43ff853ff |
| [processed/protocol_v2_2/development_labels/city_658_retrospective.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/city_658_retrospective.npz) | 8a6561bcd1622357e5f7548321391b3b6eee3d78774621a07fa3a73e6d15a27d |
| [processed/protocol_v2_2/development_labels/evaluation_keys_city_476_HD_HF.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/evaluation_keys_city_476_HD_HF.npz) | de19639d0962231cb019de45a23fcc23c2b24ca0dc235eb28b6a3fee461fece9 |
| [processed/protocol_v2_2/development_labels/evaluation_keys_city_532_HD_HF.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/evaluation_keys_city_532_HD_HF.npz) | 3fe94ec991d114d955611baa808f03be625220ff8a0299957223124764bb8340 |
| [processed/protocol_v2_2/development_labels/evaluation_keys_city_619_HD_HF.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/evaluation_keys_city_619_HD_HF.npz) | 1fb78dcc825fa1c77580df20f3676d08e4878615fe32c101149b6fb30b29be2b |
| [processed/protocol_v2_2/development_labels/evaluation_keys_city_658_HD_HF.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/evaluation_keys_city_658_HD_HF.npz) | 8a54b8b6501965fa89e2f55791bf8e4292c07eb580016ed50c8db81a7e388db7 |
| [processed/protocol_v2_2/development_labels/retrospective_label_manifest.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/development_labels/retrospective_label_manifest.json) | 1868d6a95641e43ef3ad779f8a4f779aaf94fc3cb938b3db13b77e2eb0dd3469 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_1_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_1_HD.json) | 1f3a429d8108f4c9c00a36314868baa54c51e6eaeea613b3360ea76b3cbaf8e0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_1_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_1_HD.npz) | 20aa62c1447cd73dcdfd8b1968b2000368a8baaadc703146dc0f17c7c216cacc |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_30_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_30_HD.json) | 99ad7bbbb10da34145faf7d9ae256746392b18c8edef651dd8fac4cd60c91b10 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_30_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_30_HD.npz) | 814553d851f16b0fc22a37381642c20b2ee75ceffb91e523bd41c68f2079b392 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_7_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_7_HD.json) | 1046fa3acb1573ebee6a7cee1ba609058e49dc173ca888247eaf3937f6df9458 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_7_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_7_HD.npz) | ea5387c1830fdad32c64d58bfeede84b4b198a01459099892f95b347dd9b5364 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_full_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_full_HD.json) | d08b8f9e2e8314eb6f084e7f7c470e1d6e2ddf9b95504a821bda29ea78a55122 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_full_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_full_HD.npz) | ba541c4a21a44beb9454134ae0b1df248399690a34e3cb125856a5c985203a7b |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_zero_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_zero_HD.json) | 7a3a75852de9d9fab42a21968f9c7c82fdfbbc407267158ea05f0114d6ec49e2 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_476_zero_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_476_zero_HD.npz) | 2224e9d815c592b27d98b14cce20b2d0cb1eee79bee312f2f3fa43627c4a42b0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_1_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_1_HD.json) | ed0bffc46a1041b9c28df384bddb95c84d4c48d2d22b765fe2312e3f30e4a8eb |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_1_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_1_HD.npz) | 1853d0b75eea5db51d77c3ef3013aa1e9dd472008d06137f8df2911b35479b2b |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_30_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_30_HD.json) | 43ab4bea8d31566d3e4065e7a7ac018adb4cc21325a989250a367c8845833ff1 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_30_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_30_HD.npz) | 498f0e31beb335d2bdcc87d4f193c21fa4773a94d82f779d942d71a5855fa9da |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_7_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_7_HD.json) | 2cb83b64f4ca5ef8817248af8f40c60543d7afc816d2b9e18b504642a1f5f000 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_7_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_7_HD.npz) | 3c0037d9befb8a9597ab5bb412d9d0fe5d1c955cb190ce19da2319a7ddcb2846 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_full_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_full_HD.json) | dc822449f31f2c6317aa6eeee46cdc5da0d9d16cec02c545d9ab36dd85e6ceeb |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_full_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_full_HD.npz) | 7158057eb36b9bd668fbb577705ec2284f2e24ff188c7bf3b3d4e8527cb39510 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_zero_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_zero_HD.json) | 14bb3fa2fb5a7d3da8e8d32a73a1248f0366a0a576770f13998f87f70ae422f5 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_532_zero_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_532_zero_HD.npz) | 2224e9d815c592b27d98b14cce20b2d0cb1eee79bee312f2f3fa43627c4a42b0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_1_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_1_HD.json) | a474b0547b7064abcf04537521963af0b6d79ef9e5f56723a25505feb7a6626b |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_1_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_1_HD.npz) | 5f599357854926b6fec26356b65ae92c68129cc0d22de473502008872aff2651 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_30_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_30_HD.json) | 00d226ff2951c00f6551fa74a65b1a980bc2cf896224fbf0cacaa44f56003a4a |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_30_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_30_HD.npz) | a7602330073ea0d74cb451a554725e825ffec0f05b2abf5c5f9fcb6f60541833 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_7_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_7_HD.json) | 41726d68eb192a91e0770eccdc44d3f76c12970e6e39d1efd1c149c93d9cd2fc |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_7_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_7_HD.npz) | 1d74d7e46700d01bcd01ce1e169dc012d6e9c142494876cc65c02936b4780c24 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_full_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_full_HD.json) | 29994baee727d01e8eb2b7cdaa2badc7340e7169645ec36a5f9019ae4210365f |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_full_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_full_HD.npz) | 744a98c18f3d28188f3023e79f27cbf9e163addc5e2bfc4faa36b2d43d47be1b |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_zero_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_zero_HD.json) | 9b19b8ff9333404c7b6c9226e37a922470f8cc927b2d3c3ced04931116fd8ea0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_619_zero_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_619_zero_HD.npz) | 2224e9d815c592b27d98b14cce20b2d0cb1eee79bee312f2f3fa43627c4a42b0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_1_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_1_HD.json) | ddb8a38d1e248986a17d12cbdb168abab450ab6668e6e2df6d4da31f41d6dc4d |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_1_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_1_HD.npz) | 9e1fbd3609b38e7529130a6a5900269264666c5d1e1babe78dc9a2027a3ed4e0 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_30_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_30_HD.json) | d7d39c0289afb1e808063d0fc1a3065027c21b9c1cef3e5a07051804f064f46a |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_30_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_30_HD.npz) | 5922c1bd791bd51f9fdd6b872002df02cbc3bca48903254f5e3b7943cbaa9593 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_7_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_7_HD.json) | 1349ad6597934742361159fd3ad6a3f4009b9c6799d4b6ed99d6cb0a100f0103 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_7_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_7_HD.npz) | b8508f38642ff4a19b3d14eae29b7f5a4d84b1078199946888fa68b72c4dde9e |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_full_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_full_HD.json) | edf5cc2edc6ee2953f16a28e3abeedeaef16c0f4abe9a24a195966d42f046170 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_full_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_full_HD.npz) | 5051e2bb5e1d063ff80b15e3712994b98151bdb97cf8dccb3fbcc0569cdad740 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_zero_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_zero_HD.json) | 5eda952c811eb984bddb8e86fc49de39e13040ce272371f758d892134f9fd2b7 |
| [processed/protocol_v2_2/fit_snapshots/adaptation_city_658_zero_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/adaptation_city_658_zero_HD.npz) | 2224e9d815c592b27d98b14cce20b2d0cb1eee79bee312f2f3fa43627c4a42b0 |
| [processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json) | 467aefd170c789c6533a7f2f1ebc0f3c343944f8e2781c9e2e5e0927d200077d |
| [processed/protocol_v2_2/fit_snapshots/ha_fitting_pools.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/ha_fitting_pools.json) | 2533e9dbba7cb0b39961972739cee3a259ebbbba96e21ab52ad33e1bf1192926 |
| [processed/protocol_v2_2/fit_snapshots/source_city_129_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_129_HD.npz) | d32649fdc74e1266be5c417696462816e4344ab87839b6caf5ded54f3bed1095 |
| [processed/protocol_v2_2/fit_snapshots/source_city_194_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_194_HD.npz) | 9074a270bcb163ac74b434c43f0d56d0d569d55c20c092c0070091f9e106127e |
| [processed/protocol_v2_2/fit_snapshots/source_city_438_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_438_HD.npz) | e327cd79fefd91e403f59d9041282de4cbfc1890e5a814e7440378a1722828cf |
| [processed/protocol_v2_2/fit_snapshots/source_city_467_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_467_HD.npz) | 6f64d1e29a7011bfa92dd1d9cc92e65ef60e96d6e928335da2a62c0c27977093 |
| [processed/protocol_v2_2/fit_snapshots/source_city_476_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_476_HD.npz) | cfaa6c54d92b71dec95ae750a9ba0558e952ce6c4c0b06e9483b0d0227cb9521 |
| [processed/protocol_v2_2/fit_snapshots/source_city_532_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_532_HD.npz) | 0c5f823c0b3983c9cf0c4eb56a30e27431c92e1899ba950bb180b5da3f7d3ce8 |
| [processed/protocol_v2_2/fit_snapshots/source_city_619_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_619_HD.npz) | 39ea054d850939f337dcf7e64debcf434b210242287bab331f2bee4ae5d0f1f2 |
| [processed/protocol_v2_2/fit_snapshots/source_city_658_HD.npz](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_city_658_HD.npz) | 4fbd89d601bb03e21d3a1e40cd3d9f377c96e9bfb60b3f4dd5ac2f5ecc030ac2 |
| [processed/protocol_v2_2/fit_snapshots/source_fold_excluding_476_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_fold_excluding_476_HD.json) | 00adec7a75b3551ecabe7cb1e019f529473db80be37da47c35e9717a0ca61356 |
| [processed/protocol_v2_2/fit_snapshots/source_fold_excluding_532_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_fold_excluding_532_HD.json) | f5614e447625cf03d6136aefd32a00c4f6f5fedfcb4273f127ec2224a158b2e3 |
| [processed/protocol_v2_2/fit_snapshots/source_fold_excluding_619_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_fold_excluding_619_HD.json) | 2e59b547dc5562f591d0362b7d9d9d2a74f7daea1a028cdb8156f8467d0ee883 |
| [processed/protocol_v2_2/fit_snapshots/source_fold_excluding_658_HD.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/fit_snapshots/source_fold_excluding_658_HD.json) | 6adedc5889f683476ba9e49f8aa516f0fbfbbc8ea4b29862db182cb269bf123c |
| [processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json](D:/Deep/deepest/european-bike-sharing-dataset/processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json) | 140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6 |
| [research/results/causal_history_audit_v2_2/baseline_availability.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/baseline_availability.json) | bfb87b4ab693a228ef47c02f294bcb2a7a286ef8a0a410a583af971e25f12fc2 |
| [research/results/causal_history_audit_v2_2/causal_history_audit_manifest.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/causal_history_audit_manifest.json) | 90ce85d1bcb89ffd9b304bcc534146263805b733aa5fb04004cc8e2587d5b629 |
| [research/results/causal_history_audit_v2_2/city_129_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_129_audit.json) | 33cc3f3f7532aa2423676856da9d7c11d9981c8578ac6200b61590a1c9fb3929 |
| [research/results/causal_history_audit_v2_2/city_194_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_194_audit.json) | 845ac0deaa9f43ac61706835fa4a23f18bc475ec31f2236aa7b9526aff715d89 |
| [research/results/causal_history_audit_v2_2/city_438_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_438_audit.json) | a8ed49da89ffd530b4ff88ad7a39ef8bae69323b08124e8970edb48538edb03d |
| [research/results/causal_history_audit_v2_2/city_467_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_467_audit.json) | 4af2d4abdf1d01b7eb80108f3af5c326922286550837c33f46eccae7241a533b |
| [research/results/causal_history_audit_v2_2/city_476_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_476_audit.json) | 17c758019ada2eff698c37d25ffbcea03d0a93fefafce6e72f93c22135c7bb17 |
| [research/results/causal_history_audit_v2_2/city_532_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_532_audit.json) | 44fb7304af0486d32c81c996c3d567f034def6aa90c8bd563ad0db56d2c918f5 |
| [research/results/causal_history_audit_v2_2/city_619_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_619_audit.json) | 9668e7d4b19d174805fb5886ae781726bdbca65b040297c9ee50a6105b297114 |
| [research/results/causal_history_audit_v2_2/city_658_audit.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/city_658_audit.json) | 90fac33c0c589c8b41de24e6dc85e8d731ff5016167f33d4e8475dbec75a0b87 |
| [research/results/causal_history_audit_v2_2/development_verification_report.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/development_verification_report.json) | 040e46bdfde99ad70bcb70b9b8132aac6ef64e88545d239b38a9c259de7324f2 |
| [research/results/causal_history_audit_v2_2/effective_pass_preview.csv](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/effective_pass_preview.csv) | b46d34ff034820366f9a12729e0bbccd025b39cf2ca9c8abd929165e5eb262b5 |
| [research/results/causal_history_audit_v2_2/effective_pass_preview.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/effective_pass_preview.json) | ccea6018850b77146704f2d88e8b29eb6b0b981f3aad0ea0c023360b11f75038 |
| [research/results/causal_history_audit_v2_2/fit_pool_comparison.csv](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/fit_pool_comparison.csv) | 43a52a50b13fcd5eb9793f3219ed8acf85c183977acb6476a5abbb198421c506 |
| [research/results/causal_history_audit_v2_2/fit_pool_comparison.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/fit_pool_comparison.json) | 9c6ac84ed7089c6d0bf7510a3323c43d28bcf0878d052b6045ed2b6999f4b801 |
| [research/results/causal_history_audit_v2_2/graph_static_reconciliation.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/graph_static_reconciliation.json) | e425ce873da946a4973fb03a57ffd6ecf92027da82a4220f56b6231343ff9758 |
| [research/results/causal_history_audit_v2_2/lag_availability.csv](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/lag_availability.csv) | 6b8918fef96e90bf3886d2226dac9fc3053b7941bf40563857667293ad298e8b |
| [research/results/causal_history_audit_v2_2/lag_availability.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/lag_availability.json) | 1eba60d802981083385b061d5507a444c9fa15c3777005458a65a9b740b163df |
| [research/results/causal_history_audit_v2_2/preservation_report.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/preservation_report.json) | 96b7705b978d0aaf264e348b97b6098808bfec1d58a0298e404562b1d9e52b89 |
| [research/results/causal_history_audit_v2_2/source_bindings.json](D:/Deep/deepest/european-bike-sharing-dataset/research/results/causal_history_audit_v2_2/source_bindings.json) | 19ea2bad24060f6185abb58ca130956994e222f3432b1d53efbcdd65af6fd737 |
| [research/scripts/build_development_data_v2_2.py](D:/Deep/deepest/european-bike-sharing-dataset/research/scripts/build_development_data_v2_2.py) | ac0044f64b6b2435ea06eeaef82f7274093790f1c72c9428f9936d8990421226 |
| [research/scripts/verify_development_data_v2_2.py](D:/Deep/deepest/european-bike-sharing-dataset/research/scripts/verify_development_data_v2_2.py) | 68becfa546413e1354f872b2e698053765c0680dc5fd5b13c89e59d993575f54 |
| [research/development_data_v2_2/__init__.py](D:/Deep/deepest/european-bike-sharing-dataset/research/development_data_v2_2/__init__.py) | 2705419475c8a25b3ca063d89e8ca72bdf000210124c2b5e0803aaa87df550b9 |
| [research/development_data_v2_2/registry.py](D:/Deep/deepest/european-bike-sharing-dataset/research/development_data_v2_2/registry.py) | b5a142bbe1052cbc98b6fad8e927138b6cdbddf25ea6558b26e794d983b21051 |
| [research/development_data_v2_2/README.md](D:/Deep/deepest/european-bike-sharing-dataset/research/development_data_v2_2/README.md) | 77938a36ed3223442152f3b3e738b204fc2bb72584237b6909b9eea5a37a350e |

A. V2_2_DEVELOPMENT_DATA_VERIFIED_READY_FOR_STAGE1_AUTHORIZATION
