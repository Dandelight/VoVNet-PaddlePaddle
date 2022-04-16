# python3 tools/train.py -c ./ppcls/grm/VAN_tiny_patch16_224.yaml -o Arch.pretrained=False -o Global.device=gpu
python3 tools/train.py -c ./ppcls/grm/vovnet39_x1_0_lr_step.yaml -o Arch.pretrained=False -o Global.device=gpu
