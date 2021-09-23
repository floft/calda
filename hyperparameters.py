# Generated by hyperparameter_tuning_analysis.py. Changes will be overwritten.
def get(dataset, method, values, default_value=None):
    if dataset in values and method in values[dataset]:
        result = values[dataset][method]
    else:
        result = default_value

    return result


def get_hyperparameters_str(dataset, method):
    return get(dataset, method, hyperparameters_str, "")


def get_hyperparameters_tuple(dataset, method):
    return get(dataset, method, hyperparameters_tuple, None)


def get_hyperparameters_folder(dataset, method):
    return get(dataset, method, hyperparameters_folder, None)


hyperparameters_str = {
    "ucihar": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 60 --inv_alpha 0.0005 --inv_beta 2.5 --loss_weight 0.1",
        "calda_xs_h": "--lr=0.00001 --similarity_weight=1 --max_positives=5 --max_negatives=10 --temperature=0.05",
    },
    "ucihhar": {
        "none": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "upper": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 2.5 --loss_weight 0.3",
        "calda_xs_h": "--lr=0.001 --similarity_weight=1 --max_positives=10 --max_negatives=20 --temperature=0.5",
    },
    "wisdm_ar": {
        "none": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 60 --inv_alpha 0.0001 --inv_beta 2.25 --loss_weight 0.2",
        "calda_xs_h": "--lr=0.0001 --similarity_weight=10 --max_positives=5 --max_negatives=10 --temperature=0.1",
    },
    "wisdm_at": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 2.5 --loss_weight 0.3",
        "calda_xs_h": "--lr=0.00001 --similarity_weight=100 --max_positives=10 --max_negatives=20 --temperature=0.05",
    },
    "myo": {
        "none": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.001 --similarity_weight=1 --max_positives=10 --max_negatives=20 --temperature=0.5",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 1 --loss_weight 0.1",
    },
    "ninapro_db5_like_myo_noshift": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.001 --similarity_weight=10 --max_positives=5 --max_negatives=20 --temperature=0.5",
        "upper": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 1 --loss_weight 0.1",
    },
    "normal_n12_l3_inter0_intra1_5,0,0,0_sine": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.00001 --similarity_weight=1 --max_positives=5 --max_negatives=10 --temperature=0.05",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
    },
    "normal_n12_l3_inter2_intra1_5,0,0,0_sine": {
        "none": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.001 --similarity_weight=100 --max_positives=5 --max_negatives=20 --temperature=0.05",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 1 --loss_weight 0.1",
    },
    "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.00001 --similarity_weight=10 --max_positives=5 --max_negatives=10 --temperature=0.1",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.0001 --train_source_batch_size 60 --inv_alpha 0.001 --inv_beta 2 --loss_weight 0.1",
    },
    "normal_n12_l3_inter1_intra2_0,0,5,0_sine": {
        "none": "--lr=0.001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.001 --similarity_weight=10 --max_positives=10 --max_negatives=40 --temperature=0.5",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.001 --train_source_batch_size 30 --inv_alpha 0.0001 --inv_beta 1 --loss_weight 0.1",
    },
    "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine": {
        "none": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "codats": "--lr=0.0001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "calda_xs_h": "--lr=0.00001 --similarity_weight=100 --max_positives=10 --max_negatives=20 --temperature=0.1",
        "upper": "--lr=0.00001 --similarity_weight=0 --max_positives=1 --max_negatives=1 --temperature=0",
        "can": "--base_lr 0.0001 --train_source_batch_size 30 --inv_alpha 0.001 --inv_beta 1.5 --loss_weight 0.3",
    },
}

hyperparameters_tuple = {
    "ucihar": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 60, 0.0005, 2.5, 0.1],
        "calda_xs_h": [1e-05, 1.0, 5, 10, 0.05, 0, 0, 0, 0, 0],
    },
    "ucihhar": {
        "none": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "upper": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 2.5, 0.3],
        "calda_xs_h": [0.001, 1.0, 10, 20, 0.5, 0, 0, 0, 0, 0],
    },
    "wisdm_ar": {
        "none": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 60, 0.0001, 2.25, 0.2],
        "calda_xs_h": [0.0001, 10.0, 5, 10, 0.1, 0, 0, 0, 0, 0],
    },
    "wisdm_at": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 2.5, 0.3],
        "calda_xs_h": [1e-05, 100.0, 10, 20, 0.05, 0, 0, 0, 0, 0],
    },
    "myo": {
        "none": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [0.001, 1.0, 10, 20, 0.5, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 1, 0.1],
    },
    "ninapro_db5_like_myo_noshift": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [0.001, 10.0, 5, 20, 0.5, 0, 0, 0, 0, 0],
        "upper": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 1, 0.1],
    },
    "normal_n12_l3_inter0_intra1_5,0,0,0_sine": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [1e-05, 1.0, 5, 10, 0.05, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    },
    "normal_n12_l3_inter2_intra1_5,0,0,0_sine": {
        "none": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [0.001, 100.0, 5, 20, 0.05, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 1, 0.1],
    },
    "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [1e-05, 10.0, 5, 10, 0.1, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.0001, 60, 0.001, 2, 0.1],
    },
    "normal_n12_l3_inter1_intra2_0,0,5,0_sine": {
        "none": [0.001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [0.001, 10.0, 10, 40, 0.5, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.001, 30, 0.0001, 1, 0.1],
    },
    "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine": {
        "none": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "codats": [0.0001, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "calda_xs_h": [1e-05, 100.0, 10, 20, 0.1, 0, 0, 0, 0, 0],
        "upper": [1e-05, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "can": [0.0001, 30, 0.001, 1.5, 0.3],
    },
}

hyperparameters_folder = {
    "ucihar": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.0001_w0_p1_n1_t0",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb60_a0.0005_b2.5_w0.1",
        "calda_xs_h": "lr0.00001_w1_p5_n10_t0.05",
    },
    "ucihhar": {
        "none": "lr0.001_w0_p1_n1_t0",
        "codats": "lr0.001_w0_p1_n1_t0",
        "upper": "lr0.001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b2.5_w0.3",
        "calda_xs_h": "lr0.001_w1_p10_n20_t0.5",
    },
    "wisdm_ar": {
        "none": "lr0.001_w0_p1_n1_t0",
        "codats": "lr0.0001_w0_p1_n1_t0",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb60_a0.0001_b2.25_w0.2",
        "calda_xs_h": "lr0.0001_w10_p5_n10_t0.1",
    },
    "wisdm_at": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.00001_w0_p1_n1_t0",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b2.5_w0.3",
        "calda_xs_h": "lr0.00001_w100_p10_n20_t0.05",
    },
    "myo": {
        "none": "lr0.0001_w0_p1_n1_t0",
        "codats": "lr0.001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.001_w1_p10_n20_t0.5",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b1_w0.1",
    },
    "ninapro_db5_like_myo_noshift": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.001_w10_p5_n20_t0.5",
        "upper": "lr0.0001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b1_w0.1",
    },
    "normal_n12_l3_inter0_intra1_5,0,0,0_sine": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.00001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.00001_w1_p5_n10_t0.05",
        "upper": "lr0.00001_w0_p1_n1_t0",
    },
    "normal_n12_l3_inter2_intra1_5,0,0,0_sine": {
        "none": "lr0.0001_w0_p1_n1_t0",
        "codats": "lr0.00001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.001_w100_p5_n20_t0.05",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b1_w0.1",
    },
    "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.0001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.00001_w10_p5_n10_t0.1",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.0001_sb60_a0.001_b2_w0.1",
    },
    "normal_n12_l3_inter1_intra2_0,0,5,0_sine": {
        "none": "lr0.001_w0_p1_n1_t0",
        "codats": "lr0.0001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.001_w10_p10_n40_t0.5",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.001_sb30_a0.0001_b1_w0.1",
    },
    "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine": {
        "none": "lr0.00001_w0_p1_n1_t0",
        "codats": "lr0.0001_w0_p1_n1_t0",
        "calda_xs_h": "lr0.00001_w100_p10_n20_t0.1",
        "upper": "lr0.00001_w0_p1_n1_t0",
        "can": "lr0.0001_sb30_a0.001_b1.5_w0.3",
    },
}