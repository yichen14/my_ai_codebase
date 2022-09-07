python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 0.0 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 0.1 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 0.3 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 0.5 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 0.7 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_dblp.yaml --attack_method 'random' --lr 0.0001 --device 0 --ptb_rate 1.0 &

python3 entries/train.py --cfg configs/link_pred_temporal_dblp.yaml --device 1