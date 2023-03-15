python3 entries/train.py --cfg ./configs/link_pred_gae_enron10.yaml --ptb_rate 0.1 --lr 0.0001 --model_name VGAE --device 1 &
python3 entries/train.py --cfg ./configs/link_pred_gae_enron10.yaml --ptb_rate 0.3 --lr 0.0001 --model_name VGAE --device 1 &
python3 entries/train.py --cfg ./configs/link_pred_gae_enron10.yaml --ptb_rate 0.5 --lr 0.0001 --model_name VGAE --device 0 &
python3 entries/train.py --cfg ./configs/link_pred_gae_enron10.yaml --ptb_rate 0.7 --lr 0.0001 --model_name VGAE --device 0 &
python3 entries/train.py --cfg ./configs/link_pred_gae_enron10.yaml --ptb_rate 1.0 --lr 0.0001 --model_name VGAE --device 0 