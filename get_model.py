import timm

model = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=9)
#model = timm.create_model('deit3_base_patch16_224', pretrained=True, num_classes=9)
#model = timm.create_model('deit3_base_patch16_384', pretrained=True, num_classes=9)
#model = timm.create_model('swinv2_small_window16_256', pretrained=True, num_classes=9)