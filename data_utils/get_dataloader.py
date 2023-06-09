import torch

from . import dataset, augmentations


# def get_dataloaders(config):
#     """
#     Function for creating training and validation dataloaders
#     :param config:
#     :return:
#     """
#     print("Preparing train reader...")
#     train_dataset = dataset.Product10KFS(
#         root=config.dataset.train_prefix,
#         annotation_file=config.dataset.train_list,
#         convert_file=config.dataset.cls_convert,
#         transforms=augmentations.get_train_aug(config),
#         n_pair=config.model.n_pair,
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config.dataset.batch_size,
#         shuffle=True,
#         num_workers=config.dataset.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     print("Done.")

#     print("Preparing valid reader...")
#     val_dataset = dataset.Product10KFS(
#         root=config.dataset.val_prefix,
#         annotation_file=config.dataset.val_list,
#         convert_file=config.dataset.cls_convert,
#         transforms=augmentations.get_val_aug(config),
#         is_val=True,
#     )
    

#     valid_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=config.dataset.batch_size,
#         shuffle=False,
#         num_workers=config.dataset.num_workers,
#         drop_last=False,
#         pin_memory=True,
#     )
#     print("Done.")
#     return train_loader, valid_loader

# DR
def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    train_dataset = dataset.Product10KDR(
        root=config.dataset.train_prefix,
        annotation_file=config.dataset.train_list,
        transforms=augmentations.get_train_aug(config),
        n_pair=config.model.n_pair,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print("Done.")

    print("Preparing valid reader...")
    val_dataset = dataset.Product10KDR(
        root=config.dataset.val_prefix,
        annotation_file=config.dataset.val_list,
        transforms=augmentations.get_val_aug(config),
        n_pair=config.model.n_pair
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    print("Done.")
    return train_loader, valid_loader
