# Train Generator (TiTok-B64 as example)
export PYTHONPATH=$PYTHONPATH:/home/ma-user/modelarts/work/jjw/SFT/lqf/1d-tokenizer
MACHINE_RANK=0
ROOT_IP=127.0.0.1
ROOT_PORT=24680
PATH_TO_STAGE1_or_STAGE2_WEIGHT="/home/ma-user/modelarts/work/jjw/SFT/lqf/pretrained_ckpt/tokenizer_titok_b64_imagenet/model.safetensors"
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=${MACHINE_RANK} --main_process_ip=${ROOT_IP}--main_process_port=${ROOT_PORT} --same_network scripts/train_maskgit.py config=configs/training/generator/maskgit.yaml \
    experiment.project="titok_generation" \
    experiment.name="titok_b64_maskgit" \
    experiment.output_dir="titok_b64_maskgit" \
    experiment.tokenizer_checkpoint=${PATH_TO_STAGE1_or_STAGE2_WEIGHT}