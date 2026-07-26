import segmentation_models_pytorch as smp


def build_model(encoder_weights="imagenet"):
    """
    encoder_weights: "imagenet" downloads pretrained weights (training path).
        Pass None on the inference path — the checkpoint overwrites these
        weights anyway, and the download hard-fails in a no-internet Kaggle
        inference kernel.
    """
    model_class = smp.UnetPlusPlus

    model = model_class(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
    )

    return model
