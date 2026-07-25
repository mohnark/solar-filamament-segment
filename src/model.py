import segmentation_models_pytorch as smp


def build_model():
    model_class = smp.UnetPlusPlus

    model = model_class(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )

    return model
