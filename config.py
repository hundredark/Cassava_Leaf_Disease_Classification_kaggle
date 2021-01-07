class DefaultConfig:
    # dataset
    dataset_csv_path = '../dataset/merge/merged_data.csv'
    dataset_img_dir = '../dataset/merge/train_images'
    n_classes = 5
    resolution = 512
    fold_idx = 1

    # training
    SEED = 0
    num_workers = 4

    label_smooth_eps = 0.0 # 1 - > 1.0 - 0.5 * eps, 0 -> 0.5 * eps
    BATCH_SIZE = 4
    model_arch= 'tf_efficientnet_b4_ns'
    epochs = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    weight_decay= 1e-6
    num_workers = 0
    accum_iter = 2  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    verbose_step= 1
