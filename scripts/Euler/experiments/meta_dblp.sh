python3 entries/train.py --cfg ./configs/link_pred_temporal_dblp.yaml --ptb_rate 0.1 --device 1 &
python3 entries/train.py --cfg ./configs/link_pred_temporal_dblp.yaml --ptb_rate 0.3 --device 1 &
python3 entries/train.py --cfg ./configs/link_pred_temporal_dblp.yaml --ptb_rate 0.5 --device 1 &
python3 entries/train.py --cfg ./configs/link_pred_temporal_dblp.yaml --ptb_rate 0.7 --device 0 &
python3 entries/train.py --cfg ./configs/link_pred_temporal_dblp.yaml --ptb_rate 1.0 --device 0 &