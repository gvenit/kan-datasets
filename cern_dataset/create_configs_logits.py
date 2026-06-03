#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the Ship Performance Clusterring Dataset.'
    )

    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='custom')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--patience', dest='patience', default=10)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch
    import torchmetrics
    import hashlib
    from collections import OrderedDict

    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.callbacks import *
    from kan_utils.models import *
    from kan_utils.utils import uses_momentum

    features = [
        'fj_jetNTracks', # Number of tracks associated with the AK8 jet
        'fj_nSV',  # Number of SVs associated with the AK8 jet (∆R < 0.7)
        'fj_tau0_trackEtaRel_0',# Smallest track pseudorapidity ∆η, relative to the jet axis, associated to the 1st N-subjettiness axis
        'fj_tau0_trackEtaRel_1', # Second smallest ...
        'fj_tau0_trackEtaRel_2', # Third smallest ...
        'fj_tau1_trackEtaRel_0', # Smallest track pseudorapidity ∆η, relative to the jet axis, associated to the 2nd N-subjettiness axis
        'fj_tau1_trackEtaRel_1', # Second smallest ...
        'fj_tau1_trackEtaRel_2', # Thrid smallest ...
        'fj_tau_flightDistance2dSig_0', # Transverse (2D) flight distance significance between the PV and the SV with the smallest uncertainty on the 3D flight distance associated to the first N-subjettiness axis
        'fj_tau_flightDistance2dSig_1', # ... associated to the second N-subjettiness axis
        'fj_tau_vertexDeltaR_0',  # Pseudoangular distance ∆R between the first N-subjettiness axis and SV direction
        'fj_tau_vertexEnergyRatio_0', # SV vertex energy ratio for the first N-subjettiness axis, defined as the total energy of all SVs associated with the first N-subjettiness axis divided by the total energy of all the tracks associated with the AK8 jet that are consistent with the PV
        'fj_tau_vertexEnergyRatio_1', # SV vertex energy ratio for the second N-subjettiness axis
        'fj_tau_vertexMass_0', 
        'fj_tau_vertexMass_1',
        'fj_trackSip2dSigAboveBottom_0',
        'fj_trackSip2dSigAboveBottom_1',
        'fj_trackSip2dSigAboveCharm_0',
        'fj_trackSipdSig_0',
        'fj_trackSipdSig_0_0',
        'fj_trackSipdSig_0_1',
        'fj_trackSipdSig_1',
        'fj_trackSipdSig_1_0',
        'fj_trackSipdSig_1_1',
        'fj_trackSipdSig_2',
        'fj_trackSipdSig_3',
        'fj_z_ratio'
    ]
    # spectators to define mass/pT window
    # remove_mass_pt_window = False
    remove_mass_pt_window = {
        'fj_sdmass' : ( 40,  200), # Soft drop mass of the AK8 jet
        'fj_pt'     : (300, 2000), # Transverse momentum of the AK8 jet
    }
    # 2 labels: QCD or Hbb
    labels = [
        'fj_isQCD*sample_isQCD',
        'fj_isH*fj_isBB'
    ]
    model_config = {}
    model_config['input']  = features
    model_config['output'] = labels
    model_config['remove_mass_pt_window'] = remove_mass_pt_window
    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            object_to_config(
                OrderedDict, [[
                    'kan', 
                    object_to_config(
                        FasterKAN,
                        hidden_layers     = [
                            len(model_config['input']),
                            *([] if args.hidden_layers is None else args.hidden_layers),
                            len(labels),
                        ],
                        num_grids         = args.num_grids,
                        grid_min          = args.grid_min,
                        grid_max          = args.grid_max,
                        inv_denominator   = args.scale,
                        mode              = args.mode if args.mode != 'custom' else object_to_config(
                                                # torch.nn.PReLU,
                                                # init = 0.01
                                                torch.nn.Sequential,
                                                RSWAFF,
                                                object_to_config(
                                                    Parameterizer,
                                                    module = type_to_config(RangeTransform),
                                                    data_min    = (False, 0.),
                                                    data_max    = (False, 1.),
                                                    target_min  = (False, 0.),
                                                    target_max  = ( True, 1.),
                                                )
                                            ),
                        residual          = args.residual,
                        dynamic           = args.dynamic,
                        normalize         = args.normalize,
                        dropout_rate      = object_to_config(
                            UpdatableFloat,
                            args.dropout,
                        )
                    ),
                ],
            ]),
            target_name       = 'model',
    ))
    model_config['outputs_logits'] = True
    train_config = get_default_training_config()
    train_config['task'] = 'multiclass'
    train_config['sampler'] = ['Label']
    # train_config['sample_weight'] = 'Weight'
    # train_config['splits'] = [0.66,0.09,0.25]  
    train_config.update(
        object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                CombinedLoss,
                object_to_config(
                    WeightedLoss,
                    torch.nn.CrossEntropyLoss,
                ),
                object_to_config(
                    WeightedLoss,
                    torch.nn.BCEWithLogitsLoss,
                ),
            ),
            targ_apply  = f'lambda targ: torch.nn.functional.one_hot(targ.to(torch.int64), {len(labels)}).float().squeeze(-2)',
            target_name = 'criterion',
    ))
    train_config['epochs'] = args.epochs
    train_config['patience'] = args.patience
    train_config.update(
        object_to_config(
            getattr(torch.optim, args.optimizer),
            target_name     = 'optimizer',
            weight_decay    = args.weight_decay,
            **({
                'momentum' : args.momentum
            } if uses_momentum(args.optimizer) else {})
    ))
    train_config.update(
        object_to_config(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            # factor      = 0.1,
            factor      = 0.5,
            # patience    = 10,
            patience    = 8,
            target_name = 'scheduler'
    ))
    train_config['lr'] = args.lr
    train_config['seed'] = args.seed
    train_config['batch_size'] = args.batch_size
    train_config['eval_criteria'] = {
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.Accuracy,
                task            = 'multiclass',
                num_classes     = len(labels),
            ),
            targ_apply  = 'lambda target: target.to(torch.int8).squeeze(-1)',
            target_name     = 'Accuracy',
        ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
            torchmetrics.F1Score,
                task            = 'multiclass',
                num_classes     = len(labels),
            ),
            targ_apply  = 'lambda target: target.to(torch.int8).squeeze(-1)',
            target_name     = 'F1Score',
        ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.PrecisionRecallCurve,
                task            = 'multiclass',
                num_classes     = len(labels),
                thresholds      = 100,
            ),
            targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
            target_name = 'PrecisionRecallCurve',
        ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.AUROC,
                task            = 'multiclass',
                num_classes     = len(labels),
                thresholds      = 100,
            ),
            targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
            target_name     = 'AUROC',
        ),
    }
    train_config['callbacks_arguments'].update({
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            task        = train_config['task'],
            export_path = os.path.join(THIS_DIR,'dataset','tr_statistics.csv'),
            target_name = 'train_gatherer',
        ),
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            task        = train_config['task'],
            export_path = os.path.join(THIS_DIR,'dataset','val_statistics.csv'),
            overwrite   = 1,
            target_name = 'val_gatherer',
        ),
    })
    train_config['callbacks']['train_iter_start'].extend([
        # object_to_config(
        #     'lambda *args, model=None, iteration=0, epoch=0, epochs=1, dataloader=None, **kwargs : model._modules["kan"].dropout_rate.set('
        #         f'{args.dropout} * torch.sigmoid( torch.tensor( ((epoch + (iteration / len(dataloader)) - {int(args.epochs) / 2}) / {int(args.epochs) / 4}) )).item()'
        #     ')'
        # ),
        object_to_config(
            'lambda *args, train_gatherer=None, **kwargs: train_gatherer(*args,**kwargs)',
        ),
    ])
    train_config['callbacks']['train_end'].extend([
        object_to_config(
            'lambda *args, train_gatherer=None, **kwargs: train_gatherer.finalize(*args,**kwargs)',
        ),
    ])
    train_config['callbacks']['eval_iter_start'].extend([
        object_to_config(
            'lambda *args, val_gatherer=None, **kwargs: val_gatherer(*args,**kwargs)',
        ),
    ])
    train_config['callbacks']['eval_metrics_start'].extend([
        object_to_config(
            'lambda *args, val_gatherer=None, **kwargs: val_gatherer.finalize(*args,**kwargs)',
        ),
    ])
    def build_test_dir(train_config, model_config, top_dir = None, test_version = None):
        pdir = os.path.join(
            '_'.join(['_'.join([key, str(val)]) for key, val in train_config.items()]),
            '_'.join(['_'.join([key, str(val)]) for key, val in model_config.items()]),
        )
        hashed = hashlib.sha1(pdir.encode()).hexdigest()
        pdir = hashed
        if top_dir is not None:
            pdir = os.path.join(top_dir,pdir)
        if test_version is not None:
            pdir = os.path.join(pdir,'_'.join(['test',test_version]))
        return pdir, hashed

    pdir, hashed = build_test_dir(train_config, model_config, top_dir=args.dest_top_dir, test_version=args.test_version)

    if not args.export :
        print(f'Test directory : {pdir}')

    if args.hash:
        print(hashed)

    if args.export :
        path = os.path.join(pdir,'config','train')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = save_config(train_config, path)
        
        path = os.path.join(pdir,'config','model')
        path = save_config(model_config, path)
        
    if not args.hash:
        print(os.path.dirname(pdir))
